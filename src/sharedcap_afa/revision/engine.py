"""Torch-free evaluation engine for shared-capacity test allocation.

This module re-implements the frozen protocol of the original submission on
*exported probabilities* (one row per patient with ``p0`` = triage-only risk
and ``p1`` = risk after the panel is revealed).  Working from exported
probabilities has three advantages for the revision:

1. every new analysis (alternative endpoints, trigger exclusion, tipping-point
   relabelling, net benefit) uses exactly the same allocation, threshold and
   bootstrap code;
2. the simulation study with known ground truth runs through the same code path
   as the real-data analysis, so the simulation also validates the engine;
3. the engine needs only numpy/pandas and can be unit-tested without patient
   data or GPUs.

Labels may be fractional.  Each row carries a positive weight ``w1`` and a
negative weight ``w0`` with ``w1 + w0 = 1`` (ordinary labels: ``w1 = y``).
Fractional labels are used by the tipping-point analysis in
:mod:`sharedcap_afa.revision.diagnostics`.

Conventions that mirror the original code (so that the E1 / disposition
numbers can be reproduced exactly from exported probabilities):

* allocation is top-K per consecutive pool of ``B`` rows, ties broken by row
  order (stable sort);
* random allocation draws ``rng.choice(pool, K, replace=False)`` pool by pool
  from a single ``default_rng(seed)`` stream, averaged over repeats;
* threshold selection: lowest FPR among grid thresholds whose FNR meets the
  target, ties broken toward the larger threshold; if none is feasible, lowest
  FNR, then FPR, then larger threshold;
* the global centre is the candidate with the lowest mean tune-split FPR among
  candidates feasible at every nontrivial tuning capacity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ..statistical_evaluation import paired_cluster_bootstrap


# --------------------------------------------------------------------------- config

def grid(lo: float, hi: float, step: float) -> np.ndarray:
    n = int(np.floor((hi - lo) / step + 1e-9))
    vals = [round(lo + i * step, 10) for i in range(n + 1)]
    if vals[-1] < hi - 1e-10:
        vals.append(float(hi))
    return np.asarray(vals, dtype=float)


@dataclass
class SeedScheme:
    """Seeds for random allocation, mirroring the original scripts.

    seed(phase, K, rep) = global_seed + base[phase] + K * k_mult
                          + random_method_index * method_mult + rep
    """

    global_seed: int = 13
    base: dict = field(default_factory=lambda: {
        "tune": 1_000_000, "calibration": 3_000_000, "audit": 5_000_000, "test": 7_000_000,
    })
    k_mult: int = 100_000
    method_mult: int = 10_000
    random_method_index: int = 0

    def seed(self, phase: str, K: int, rep: int) -> int:
        return int(self.global_seed + self.base[phase] + K * self.k_mult
                   + self.random_method_index * self.method_mult + rep)


MIMIC_SEEDS = SeedScheme()
MCMED_SEEDS = SeedScheme(base={"calibration": 850_000, "test": 950_000}, k_mult=1000,
                         method_mult=100, random_method_index=2)


@dataclass
class ProtocolConfig:
    B: int = 16
    fnr_target: float = 0.06
    tune_slots: tuple = (2, 4, 5, 8, 11)
    eval_slots: tuple = (5, 8, 11)
    tune_thresholds: np.ndarray = field(default_factory=lambda: grid(0.15, 0.35, 0.0025))
    calib_thresholds: np.ndarray = field(default_factory=lambda: grid(0.15, 0.35, 0.0025))
    centers: tuple = (0.18, 0.20, 0.22, 0.23, 0.24, 0.245, 0.25, 0.27, 0.30, 0.32)
    miss_weights: tuple = (0.25, 0.5, 1.0, 2.0, 4.0)
    random_repeats: int = 10
    seeds: SeedScheme = field(default_factory=SeedScheme)
    compare_dtype: str = "float32"   # float32 mirrors torch evaluation
    frozen_center: float | None = None          # skip centre tuning if given
    frozen_decision_risk: tuple | None = None   # (centre, weight)
    include_mechanistic: bool = True            # lowest- and highest-risk-first
    extra_score_columns: tuple = ()             # e.g. ("voi",) ranked descending

    @staticmethod
    def for_rare_outcome(prevalence_hint: float = 0.05, **kw) -> "ProtocolConfig":
        """Grids appropriate for a low-prevalence endpoint (thresholds near 0)."""
        hi = min(0.5, max(0.10, 6 * prevalence_hint))
        step = 0.0005 if hi <= 0.2 else 0.0025
        th = grid(0.0005, hi, step)
        centers = tuple(np.round(np.r_[0.0, grid(0.0025, hi, 0.0025 if hi <= 0.2 else 0.01)], 5))
        return ProtocolConfig(tune_thresholds=th, calib_thresholds=th, centers=centers, **kw)


# ------------------------------------------------------------------- primitives

def _as_cmp(x: np.ndarray, dtype: str) -> np.ndarray:
    return np.asarray(x, dtype=np.float32 if dtype == "float32" else np.float64)


def label_weights(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "w1" in df.columns:
        w1 = df["w1"].to_numpy(float)
    else:
        w1 = df["y"].to_numpy(float)
    return w1, 1.0 - w1


def priority_scores(p0: np.ndarray, priority: str, center: float | None = None,
                    miss_weight: float | None = None) -> np.ndarray | None:
    p = np.asarray(p0, dtype=float)
    if priority == "random":
        return None
    if priority == "entropy":
        q = np.clip(p, 1e-9, 1 - 1e-9)
        return -(q * np.log(q) + (1 - q) * np.log(1 - q))
    if priority == "margin":
        return -np.abs(p - float(center))
    if priority == "decision_risk":
        c, w = float(center), float(miss_weight)
        return np.where(p >= c, 1.0 - p, w * p)
    raise ValueError(priority)


def allocate(n: int, B: int, K: int, score: np.ndarray | None, *, seed: int | None = None) -> np.ndarray:
    """Boolean selection mask over the first floor(n/B)*B rows."""
    usable = (n // B) * B
    sel = np.zeros(usable, dtype=bool)
    if K <= 0 or usable == 0:
        return sel
    if K >= B:
        sel[:] = True
        return sel
    if score is None:
        rng = np.random.default_rng(int(seed))
        for lo in range(0, usable, B):
            chosen = rng.choice(np.arange(lo, lo + B), size=K, replace=False)
            sel[chosen] = True
        return sel
    S = np.asarray(score, dtype=float)[:usable].reshape(-1, B)
    order = np.argsort(-S, axis=1, kind="stable")[:, :K]
    rows = np.repeat(np.arange(S.shape[0]), K)
    sel.reshape(-1, B)[rows, order.ravel()] = True
    return sel


def final_prob(p0: np.ndarray, p1: np.ndarray, sel: np.ndarray) -> np.ndarray:
    u = len(sel)
    return np.where(sel, np.asarray(p1)[:u], np.asarray(p0)[:u])


def error_rates(w1: np.ndarray, w0: np.ndarray, prob: np.ndarray, tau: float,
                dtype: str = "float32") -> tuple[float, float]:
    u = len(prob)
    a = _as_cmp(prob, dtype) >= _as_cmp(np.asarray([tau]), dtype)[0]
    pos, neg = w1[:u].sum(), w0[:u].sum()
    fnr = float((w1[:u] * (~a)).sum() / pos) if pos > 0 else float("nan")
    fpr = float((w0[:u] * a).sum() / neg) if neg > 0 else float("nan")
    return fnr, fpr


def matched_fpr(w1: np.ndarray, w0: np.ndarray, prob: np.ndarray, target: float) -> float:
    """Lowest FPR attainable on this sample with FNR <= target (oracle threshold).

    Analysis-only quantity: it removes calibration-to-test threshold noise and
    therefore isolates the quality of the allocation ranking.  It is not a
    deployable protocol and is reported separately from calibrated results.
    """
    u = len(prob)
    a1, a0 = np.asarray(w1, float)[:u], np.asarray(w0, float)[:u]
    order = np.argsort(-np.asarray(prob, float), kind="stable")
    p = np.asarray(prob, float)[order]
    tp, fp = np.cumsum(a1[order]), np.cumsum(a0[order])
    last = np.r_[np.flatnonzero(np.diff(p) != 0), u - 1]
    P, N = a1.sum(), a0.sum()
    fnr = 1.0 - tp[last] / max(P, 1e-12)
    fpr = fp[last] / max(N, 1e-12)
    ok = fnr <= float(target) + 1e-12
    return float(fpr[ok].min()) if ok.any() else 1.0


def threshold_scan(w1: np.ndarray, w0: np.ndarray, probs: Sequence[np.ndarray], thresholds: np.ndarray,
                   target: float, dtype: str = "float32") -> dict:
    """Original selection rule, vectorised; FNR/FPR averaged across repeats."""
    ts = np.asarray(thresholds, dtype=float)
    tcmp = _as_cmp(ts, dtype)
    fnrs, fprs = [], []
    for p in probs:
        u = len(p)
        a = _as_cmp(p, dtype)[None, :] >= tcmp[:, None]
        pos, neg = w1[:u].sum(), w0[:u].sum()
        fnrs.append(((~a) * w1[None, :u]).sum(axis=1) / max(pos, 1e-12))
        fprs.append((a * w0[None, :u]).sum(axis=1) / max(neg, 1e-12))
    fnr = np.mean(np.stack(fnrs), axis=0)
    fpr = np.mean(np.stack(fprs), axis=0)
    feas = np.flatnonzero(fnr <= float(target) + 1e-12)
    if len(feas):
        idx = int(feas[np.lexsort((-ts[feas], fpr[feas]))[0]])
        ok = True
    else:
        idx = int(np.lexsort((-ts, fpr, fnr))[0])
        ok = False
    return {"threshold": float(ts[idx]), "feasible": bool(ok), "fnr": float(fnr[idx]), "fpr": float(fpr[idx])}


# -------------------------------------------------------------------- policies

@dataclass
class PolicySpec:
    name: str
    priority: str                    # random | entropy | margin | decision_risk | score:<column>
    center: float | None = None
    miss_weight: float | None = None


def _selections(role_df: pd.DataFrame, spec: PolicySpec, K: int, B: int, *, phase: str,
                cfg: ProtocolConfig) -> list[np.ndarray]:
    n = len(role_df)
    p0 = role_df["p0"].to_numpy(float)
    if spec.priority == "random":
        reps = int(cfg.random_repeats)
        return [allocate(n, B, K, None, seed=cfg.seeds.seed(phase, K, r)) for r in range(reps)]
    if spec.priority.startswith("score:"):
        score = role_df[spec.priority.split(":", 1)[1]].to_numpy(float)
    else:
        score = priority_scores(p0, spec.priority, spec.center, spec.miss_weight)
    return [allocate(n, B, K, score)]


def _probs_for(role_df: pd.DataFrame, sels: list[np.ndarray]) -> list[np.ndarray]:
    p0 = role_df["p0"].to_numpy(float)
    p1 = role_df["p1"].to_numpy(float)
    return [final_prob(p0, p1, s) for s in sels]


def calibrate_policy(role_df: pd.DataFrame, spec: PolicySpec, K: int, *, phase: str,
                     cfg: ProtocolConfig, thresholds: np.ndarray) -> dict:
    sels = _selections(role_df, spec, K, cfg.B, phase=phase, cfg=cfg)
    w1, w0 = label_weights(role_df)
    return threshold_scan(w1, w0, _probs_for(role_df, sels), thresholds, cfg.fnr_target, cfg.compare_dtype)


def anchor_probs(role_df: pd.DataFrame, which: str, B: int) -> np.ndarray:
    u = (len(role_df) // B) * B
    col = "p1" if which == "cbc_all" else "p0"
    return role_df[col].to_numpy(float)[:u]


# ------------------------------------------------------------------ tuning

def tune_global(tune_df: pd.DataFrame, cfg: ProtocolConfig, family: str = "margin") -> dict:
    """Select one centre (or centre/weight) across all nontrivial tuning capacities."""
    w1, w0 = label_weights(tune_df)
    keys: list[tuple] = []
    if family == "margin":
        keys = [(float(c), None) for c in cfg.centers]
    elif family == "decision_risk":
        keys = [(float(c), float(w)) for c in cfg.centers for w in cfg.miss_weights]
    else:
        raise ValueError(family)
    slots = [K for K in cfg.tune_slots if 0 < K < cfg.B]
    rows = []
    for c, w in keys:
        spec = PolicySpec("cand", family, c, w)
        res = []
        for K in slots:
            sels = _selections(tune_df, spec, K, cfg.B, phase="tune", cfg=cfg)
            res.append(threshold_scan(w1, w0, _probs_for(tune_df, sels), cfg.tune_thresholds,
                                      cfg.fnr_target, cfg.compare_dtype))
        rows.append({"center": c, "miss_weight": w,
                     "feasible_all": all(r["feasible"] for r in res),
                     "mean_fpr": float(np.mean([r["fpr"] for r in res])),
                     "max_fpr": float(np.max([r["fpr"] for r in res]))})
    table = pd.DataFrame(rows)
    elig = table[table["feasible_all"]]
    if elig.empty:
        best = table.sort_values(["mean_fpr", "max_fpr", "center"]).iloc[0]
        status = "no_candidate_feasible_at_all_capacities"
    else:
        sort_cols = ["mean_fpr", "max_fpr", "center"] + (["miss_weight"] if family == "decision_risk" else [])
        best = elig.sort_values(sort_cols).iloc[0]
        status = "ok"
    return {"family": family, "center": float(best["center"]),
            "miss_weight": (None if best["miss_weight"] is None or pd.isna(best["miss_weight"]) else float(best["miss_weight"])),
            "status": status, "table": table}


# ------------------------------------------------------------------ protocol

@dataclass
class RoleFrames:
    tune: pd.DataFrame | None
    calibration: pd.DataFrame
    test: pd.DataFrame
    audit: pd.DataFrame | None = None

    @staticmethod
    def from_frame(df: pd.DataFrame, roles: dict | None = None) -> "RoleFrames":
        roles = roles or {"tune": "tune", "calibration": "calibration", "test": "test", "audit": "audit"}
        def get(key):
            r = roles.get(key)
            if r is None:
                return None
            x = df[df["role"] == r]
            if "order" in x.columns:
                x = x.sort_values("order", kind="mergesort")
            return x.reset_index(drop=True) if len(x) else None
        return RoleFrames(get("tune"), get("calibration"), get("test"), get("audit"))


def default_policies(cfg: ProtocolConfig, margin_center: float, dr: tuple | None) -> list[PolicySpec]:
    pols = [PolicySpec("margin_global", "margin", margin_center),
            PolicySpec("random", "random"),
            PolicySpec("entropy", "entropy")]
    if dr is not None:
        pols.append(PolicySpec("decision_risk_global", "decision_risk", dr[0], dr[1]))
    if cfg.include_mechanistic:
        pols += [PolicySpec("lowest_risk_first", "margin", 0.0), PolicySpec("highest_risk_first", "margin", 1.0)]
    for col in cfg.extra_score_columns:
        pols.append(PolicySpec(f"score_{col}", f"score:{col}"))
    return pols


def _batch_frame(method: str, rep: int, w1: np.ndarray, w0: np.ndarray, prob: np.ndarray,
                 sel: np.ndarray | None, tau: float, B: int, dtype: str) -> pd.DataFrame:
    u = len(prob)
    a = _as_cmp(prob, dtype) >= _as_cmp(np.asarray([tau]), dtype)[0]
    bid = np.repeat(np.arange(u // B), B)
    nb = u // B
    bc = lambda v: np.bincount(bid, weights=np.asarray(v, float), minlength=nb)
    tested = np.zeros(u) if sel is None else sel.astype(float)
    return pd.DataFrame({
        "method": method, "repeat": rep, "batch_id": np.arange(nb), "n": bc(np.ones(u)),
        "positives": bc(w1[:u]), "misses": bc(w1[:u] * (~a)), "negatives": bc(w0[:u]),
        "false_positives": bc(w0[:u] * a), "total_cost": bc(tested), "total_groups": bc(tested),
        "untested_count": bc(1 - tested),
    })


def run_protocol(frames: RoleFrames, cfg: ProtocolConfig, *, bootstrap_reps: int = 2000,
                 bootstrap_seed: int = 20260908, policies: list[PolicySpec] | None = None,
                 comparators: Sequence[str] | None = None) -> dict:
    """Tune (if needed) -> calibrate thresholds -> evaluate on test -> paired bootstrap."""
    B = cfg.B
    out: dict = {"config": {"B": B, "fnr_target": cfg.fnr_target, "eval_slots": list(cfg.eval_slots)}}
    # ---- tuning of global allocation parameters (policy-tune role only)
    if cfg.frozen_center is not None:
        c_star = float(cfg.frozen_center)
        out["margin_tuning"] = {"center": c_star, "status": "frozen"}
    else:
        if frames.tune is None:
            raise ValueError("tune role required when no frozen centre is supplied")
        mt = tune_global(frames.tune, cfg, "margin")
        c_star = mt["center"]
        out["margin_tuning"] = {k: v for k, v in mt.items() if k != "table"}
        out["margin_tuning_table"] = mt["table"]
    dr = None
    if cfg.frozen_decision_risk is not None:
        dr = tuple(cfg.frozen_decision_risk)
    elif frames.tune is not None and cfg.miss_weights:
        dt = tune_global(frames.tune, cfg, "decision_risk")
        dr = (dt["center"], dt["miss_weight"])
        out["decision_risk_tuning"] = {k: v for k, v in dt.items() if k != "table"}
    pols = policies or default_policies(cfg, c_star, dr)

    cal, test = frames.calibration, frames.test
    w1c, w0c = label_weights(cal)
    w1t, w0t = label_weights(test)
    # ---- anchors
    anchors = {}
    for a in ("free_only", "cbc_all"):
        pc = anchor_probs(cal, a, B)
        sel_c = threshold_scan(w1c, w0c, [pc], cfg.calib_thresholds, cfg.fnr_target, cfg.compare_dtype)
        pt = anchor_probs(test, a, B)
        fnr, fpr = error_rates(w1t, w0t, pt, sel_c["threshold"], cfg.compare_dtype)
        anchors[a] = {"threshold": sel_c["threshold"], "fnr": fnr, "fpr": fpr,
                      "fpr_matched": matched_fpr(w1t, w0t, pt, cfg.fnr_target),
                      "tested_fraction": 1.0 if a == "cbc_all" else 0.0}
    out["anchors"] = anchors
    fpr_free, fpr_all = anchors["free_only"]["fpr"], anchors["cbc_all"]["fpr"]
    span = fpr_free - fpr_all
    mfree, mall = anchors["free_only"]["fpr_matched"], anchors["cbc_all"]["fpr_matched"]
    mspan = mfree - mall

    out["capacity"] = {}
    for K in cfg.eval_slots:
        res, batch_parts = {}, []
        for spec in pols:
            calsel = calibrate_policy(cal, spec, K, phase="calibration", cfg=cfg, thresholds=cfg.calib_thresholds)
            tau = calsel["threshold"]
            sels = _selections(test, spec, K, B, phase="test", cfg=cfg)
            probs = _probs_for(test, sels)
            rates = [error_rates(w1t, w0t, p, tau, cfg.compare_dtype) for p in probs]
            fnr = float(np.mean([r[0] for r in rates]))
            fpr = float(np.mean([r[1] for r in rates]))
            fm = float(np.mean([matched_fpr(w1t, w0t, p, cfg.fnr_target) for p in probs]))
            res[spec.name] = {"threshold": tau, "calibration_feasible": calsel["feasible"],
                              "fnr": fnr, "fpr": fpr, "fpr_matched": fm,
                              "share_of_universal_benefit": (fpr_free - fpr) / span if abs(span) > 1e-12 else float("nan"),
                              "share_matched": (mfree - fm) / mspan if abs(mspan) > 1e-12 else float("nan"),
                              "center": spec.center, "miss_weight": spec.miss_weight}
            for rep, (p, s) in enumerate(zip(probs, sels)):
                batch_parts.append(_batch_frame(spec.name, rep, w1t, w0t, p, s, tau, B, cfg.compare_dtype))
        for a in ("free_only", "cbc_all"):
            pt = anchor_probs(test, a, B)
            sel = np.zeros(len(pt), bool) if a == "free_only" else np.ones(len(pt), bool)
            batch_parts.append(_batch_frame(a, 0, w1t, w0t, pt, sel, anchors[a]["threshold"], B, cfg.compare_dtype))
        bdf = pd.concat(batch_parts, ignore_index=True)
        comps = list(comparators) if comparators else [m for m in bdf["method"].unique() if m != "margin_global"]
        boot = None
        if bootstrap_reps and bootstrap_reps >= 10:
            boot = paired_cluster_bootstrap(bdf, reference="margin_global", comparators=comps,
                                            n_boot=int(bootstrap_reps), seed=int(bootstrap_seed) + int(K))
        out["capacity"][int(K)] = {"methods": res, "bootstrap": boot, "batch_frame": bdf}
    return out


# ----------------------------------------------------------------- reporting helpers

def summary_table(result: dict) -> pd.DataFrame:
    rows = []
    for a, z in result["anchors"].items():
        rows.append({"K": None, "method": a, "fnr": z["fnr"], "fpr": z["fpr"], "threshold": z["threshold"],
                     "share": 0.0 if a == "free_only" else 1.0, "fpr_matched": z.get("fpr_matched"),
                     "share_matched": 0.0 if a == "free_only" else 1.0})
    for K, cz in result["capacity"].items():
        for m, z in cz["methods"].items():
            row = {"K": K, "method": m, "fnr": z["fnr"], "fpr": z["fpr"], "threshold": z["threshold"],
                   "share": z["share_of_universal_benefit"], "fpr_matched": z.get("fpr_matched"),
                   "share_matched": z.get("share_matched")}
            boot = cz.get("bootstrap")
            if boot is not None and m != "margin_global":
                c = boot["comparisons"].get(f"margin_global_vs_{m}")
                if c is not None:
                    row.update({"dFPR": c["delta"]["fpr"], "dFPR_lo": c["ci"]["fpr"]["lower"],
                                "dFPR_hi": c["ci"]["fpr"]["upper"], "dFNR": c["delta"]["fnr"],
                                "dFNR_lo": c["ci"]["fnr"]["lower"], "dFNR_hi": c["ci"]["fnr"]["upper"]})
            rows.append(row)
    return pd.DataFrame(rows)


def make_pools(df: pd.DataFrame, B: int) -> pd.DataFrame:
    """Re-number consecutive pools after filtering; drops the incomplete tail."""
    x = df.reset_index(drop=True)
    u = (len(x) // B) * B
    x = x.iloc[:u].copy()
    x["order"] = np.arange(u)
    x["pool"] = np.repeat(np.arange(u // B), B)
    return x
