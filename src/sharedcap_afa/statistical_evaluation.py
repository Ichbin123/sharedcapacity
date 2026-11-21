"""Paired shared-capacity-batch inference and publication tables.

The resampling unit is one shared-capacity competition batch.  Stochastic policy repeats are
first averaged within batch, then the same sampled batch indices are applied to
all methods.  This preserves pairing induced by the shared patient cohort while
keeping Monte Carlo variability separate from patient/batch sampling
uncertainty.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METRICS = ("fnr", "fpr", "mean_cost", "mean_groups")
DEFAULT_COMPARATORS = ("random", "entropy", "decision_risk_global")


def _ensure_same_batches(frames: dict[str, pd.DataFrame]) -> np.ndarray:
    batch_sets = {m: tuple(sorted(df["batch_id"].unique().tolist())) for m, df in frames.items()}
    first_method = next(iter(batch_sets))
    ref = batch_sets[first_method]
    for method, batches in batch_sets.items():
        if batches != ref:
            raise ValueError(f"Batch IDs do not match between {first_method} and {method}")
    return np.asarray(ref, dtype=int)


def collapse_mc_within_batch(batch_df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Return one row per batch after averaging stochastic repeats.

    Label-dependent denominators are required to be identical across repeats;
    policy-dependent numerators/costs are averaged across repeats.  Static
    policies naturally have one repeat and therefore pass through unchanged.
    """
    x = batch_df.loc[batch_df["method"] == method].copy()
    if x.empty:
        raise ValueError(f"No batch rows found for method={method!r}")

    # Verify cohort invariants before averaging stochastic policy outputs.
    for col in ("n", "positives", "negatives"):
        nunique = x.groupby("batch_id")[col].nunique(dropna=False)
        if (nunique > 1).any():
            bad = nunique[nunique > 1].index.tolist()[:5]
            raise ValueError(f"{col} differs across repeats for method={method}, batches={bad}")

    invariant = x.groupby("batch_id", sort=True)[["n", "positives", "negatives"]].first()
    stochastic = x.groupby("batch_id", sort=True)[
        ["misses", "false_positives", "total_cost", "total_groups"]
    ].mean()
    out = invariant.join(stochastic).reset_index()
    out["method"] = method
    return out


def metrics_from_collapsed(df: pd.DataFrame) -> dict[str, float]:
    pos = float(df["positives"].sum())
    neg = float(df["negatives"].sum())
    n = float(df["n"].sum())
    return {
        "fnr": float(df["misses"].sum() / pos) if pos > 0 else float("nan"),
        "fpr": float(df["false_positives"].sum() / neg) if neg > 0 else float("nan"),
        "mean_cost": float(df["total_cost"].sum() / n) if n > 0 else float("nan"),
        "mean_groups": float(df["total_groups"].sum() / n) if n > 0 else float("nan"),
    }


def _metric_on_sample(df: pd.DataFrame, sample: np.ndarray, metric: str) -> np.ndarray:
    n = df["n"].to_numpy(dtype=float)
    pos = df["positives"].to_numpy(dtype=float)
    neg = df["negatives"].to_numpy(dtype=float)
    miss = df["misses"].to_numpy(dtype=float)
    fp = df["false_positives"].to_numpy(dtype=float)
    cost = df["total_cost"].to_numpy(dtype=float)
    groups = df["total_groups"].to_numpy(dtype=float)

    if metric == "fnr":
        num = miss[sample].sum(axis=1)
        den = pos[sample].sum(axis=1)
    elif metric == "fpr":
        num = fp[sample].sum(axis=1)
        den = neg[sample].sum(axis=1)
    elif metric == "mean_cost":
        num = cost[sample].sum(axis=1)
        den = n[sample].sum(axis=1)
    elif metric == "mean_groups":
        num = groups[sample].sum(axis=1)
        den = n[sample].sum(axis=1)
    else:
        raise KeyError(metric)
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)


def _ci(v: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - float(level)) / 2.0
    lo, hi = np.quantile(v, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def paired_cluster_bootstrap(
    batch_df: pd.DataFrame,
    *,
    reference: str = "margin_global",
    comparators: Iterable[str] = DEFAULT_COMPARATORS,
    n_boot: int = 5000,
    seed: int = 20260906,
    level: float = 0.95,
) -> dict:
    """Paired nonparametric bootstrap over concurrent batches.

    Reported deltas are reference - comparator.  Thus negative delta FPR/FNR/
    cost favors the reference method.
    """
    comparators = tuple(comparators)
    methods = (reference,) + tuple(m for m in comparators if m != reference)
    collapsed = {m: collapse_mc_within_batch(batch_df, m) for m in methods}
    batch_ids = _ensure_same_batches(collapsed)
    for m in methods:
        collapsed[m] = collapsed[m].set_index("batch_id").loc[batch_ids].reset_index()

    k = len(batch_ids)
    if k < 2:
        raise ValueError("At least two concurrent batches are required for cluster bootstrap")
    if int(n_boot) < 10:
        raise ValueError("n_boot must be at least 10")

    rng = np.random.default_rng(int(seed))
    # ~5000 x 2000 indices is about 80 MB as int64.  Use int32 where safe.
    dtype = np.int32 if k < np.iinfo(np.int32).max else np.int64
    sample = rng.integers(0, k, size=(int(n_boot), k), dtype=dtype)

    out = {
        "resampling_unit": "shared_capacity_batch",
        "mc_handling": "average policy sufficient statistics within batch before cluster bootstrap",
        "delta_definition": "reference_minus_comparator",
        "reference": reference,
        "n_batches": int(k),
        "bootstrap_reps": int(n_boot),
        "bootstrap_seed": int(seed),
        "ci_level": float(level),
        "methods": {},
        "comparisons": {},
    }

    boot_cache: dict[tuple[str, str], np.ndarray] = {}
    for method, df in collapsed.items():
        point = metrics_from_collapsed(df)
        entry = {"point": point, "ci": {}}
        for metric in METRICS:
            vals = _metric_on_sample(df, sample, metric)
            boot_cache[(method, metric)] = vals
            lo, hi = _ci(vals, level)
            entry["ci"][metric] = {"lower": lo, "upper": hi}
        out["methods"][method] = entry

    for comp in comparators:
        if comp == reference:
            continue
        ckey = f"{reference}_vs_{comp}"
        point_ref = out["methods"][reference]["point"]
        point_cmp = out["methods"][comp]["point"]
        centry = {"comparator": comp, "delta": {}, "ci": {}, "probability_reference_lower": {}}
        for metric in METRICS:
            delta_point = float(point_ref[metric] - point_cmp[metric])
            delta_boot = boot_cache[(reference, metric)] - boot_cache[(comp, metric)]
            lo, hi = _ci(delta_boot, level)
            centry["delta"][metric] = delta_point
            centry["ci"][metric] = {"lower": lo, "upper": hi}
            finite = delta_boot[np.isfinite(delta_boot)]
            centry["probability_reference_lower"][metric] = (
                float((finite < 0).mean()) if len(finite) else float("nan")
            )
        out["comparisons"][ckey] = centry
    return out


def _fmt_ci(point: float, ci: dict, digits: int = 4) -> str:
    return f"{point:.{digits}f} [{ci['lower']:.{digits}f}, {ci['upper']:.{digits}f}]"


def write_publication_tables(
    results: dict,
    bootstrap: dict,
    *,
    capacity: float,
    out_dir: str | Path,
    fnr_target: float | None = None,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap_key = str(float(capacity))
    if cap_key not in results["capacity"]:
        # JSON may preserve "1.0" but callers sometimes pass an int-like key.
        matches = [k for k in results["capacity"] if abs(float(k) - float(capacity)) < 1e-12]
        if not matches:
            raise KeyError(f"capacity={capacity} missing from results")
        cap_key = matches[0]
    cres = results["capacity"][cap_key]

    method_rows = []
    display = {
        "margin_global": "Operating-point margin",
        "decision_risk_global": "Decision risk",
        "random": "Random",
        "entropy": "Entropy",
    }
    for method in ("margin_global", "decision_risk_global", "random", "entropy"):
        if method not in cres or method not in bootstrap["methods"]:
            continue
        b = bootstrap["methods"][method]
        m = cres[method]["metrics"]
        fnr = b["point"]["fnr"]
        row = {
            "method": display.get(method, method),
            "method_key": method,
            "fnr": fnr,
            "fnr_ci_lower": b["ci"]["fnr"]["lower"],
            "fnr_ci_upper": b["ci"]["fnr"]["upper"],
            "fpr": b["point"]["fpr"],
            "fpr_ci_lower": b["ci"]["fpr"]["lower"],
            "fpr_ci_upper": b["ci"]["fpr"]["upper"],
            "mean_cost": b["point"]["mean_cost"],
            "cost_ci_lower": b["ci"]["mean_cost"]["lower"],
            "cost_ci_upper": b["ci"]["mean_cost"]["upper"],
            "auroc": m.get("auroc"),
            "worst_fnr_ucb": cres[method].get("worst_fnr_ucb"),
            "formal_safety_pass": cres[method].get("safety_pass"),
            "point_estimate_fnr_below_target": (None if fnr_target is None else bool(fnr <= fnr_target)),
        }
        method_rows.append(row)
    method_df = pd.DataFrame(method_rows)
    method_csv = out_dir / "method_summary.csv"
    method_df.to_csv(method_csv, index=False)

    pair_rows = []
    for key, c in bootstrap["comparisons"].items():
        pair_rows.append({
            "comparison": key,
            "comparator": c["comparator"],
            "delta_fnr": c["delta"]["fnr"],
            "delta_fnr_ci_lower": c["ci"]["fnr"]["lower"],
            "delta_fnr_ci_upper": c["ci"]["fnr"]["upper"],
            "delta_fpr": c["delta"]["fpr"],
            "delta_fpr_ci_lower": c["ci"]["fpr"]["lower"],
            "delta_fpr_ci_upper": c["ci"]["fpr"]["upper"],
            "delta_cost": c["delta"]["mean_cost"],
            "delta_cost_ci_lower": c["ci"]["mean_cost"]["lower"],
            "delta_cost_ci_upper": c["ci"]["mean_cost"]["upper"],
            "p_reference_lower_fnr": c["probability_reference_lower"]["fnr"],
            "p_reference_lower_fpr": c["probability_reference_lower"]["fpr"],
            "p_reference_lower_cost": c["probability_reference_lower"]["mean_cost"],
        })
    pair_df = pd.DataFrame(pair_rows)
    pair_csv = out_dir / "pairwise_vs_reference.csv"
    pair_df.to_csv(pair_csv, index=False)

    lines = [
        f"# Primary capacity statistical table (capacity={float(capacity):g})",
        "",
        "Cluster bootstrap resamples shared-capacity competition batches. Stochastic policy repeats are averaged within batch before resampling.",
        "",
        "| Method | FNR (95% CI) | FPR (95% CI) | Mean cost (95% CI) | AUROC | Worst FNR UCB | Formal audit |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        b = bootstrap["methods"][row["method_key"]]
        lines.append(
            "| {method} | {fnr} | {fpr} | {cost} | {auroc:.4f} | {ucb:.4f} | {audit} |".format(
                method=row["method"],
                fnr=_fmt_ci(b["point"]["fnr"], b["ci"]["fnr"]),
                fpr=_fmt_ci(b["point"]["fpr"], b["ci"]["fpr"]),
                cost=_fmt_ci(b["point"]["mean_cost"], b["ci"]["mean_cost"], digits=3),
                auroc=float(row["auroc"]),
                ucb=float(row["worst_fnr_ucb"]),
                audit="PASS" if row["formal_safety_pass"] else "FAIL",
            )
        )
    lines += [
        "",
        "## Paired deltas: reference policy minus comparator",
        "",
        "Negative FNR/FPR/cost deltas favor the reference policy.",
        "",
        "| Comparator | ΔFNR (95% CI) | ΔFPR (95% CI) | Δcost (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for key, c in bootstrap["comparisons"].items():
        lines.append(
            "| {comp} | {fnr} | {fpr} | {cost} |".format(
                comp=display.get(c["comparator"], c["comparator"]),
                fnr=_fmt_ci(c["delta"]["fnr"], c["ci"]["fnr"]),
                fpr=_fmt_ci(c["delta"]["fpr"], c["ci"]["fpr"]),
                cost=_fmt_ci(c["delta"]["mean_cost"], c["ci"]["mean_cost"], digits=3),
            )
        )
    table_md = out_dir / "TABLE_PRIMARY.md"
    table_md.write_text("\n".join(lines) + "\n")

    return {
        "method_summary_csv": str(method_csv),
        "pairwise_csv": str(pair_csv),
        "primary_table_md": str(table_md),
    }


def run_from_files(
    *,
    results_path: str | Path,
    batch_csv: str | Path,
    out_dir: str | Path,
    capacity: float,
    n_boot: int,
    seed: int,
    fnr_target: float | None = None,
) -> dict:
    results = json.loads(Path(results_path).read_text())
    batch_df = pd.read_csv(batch_csv)
    if "capacity" in batch_df.columns:
        z = np.isclose(batch_df["capacity"].to_numpy(dtype=float), float(capacity))
        batch_df = batch_df.loc[z].copy()
        if batch_df.empty:
            raise ValueError(f"No batch rows found for capacity={capacity}")
    boot = paired_cluster_bootstrap(batch_df, n_boot=n_boot, seed=seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    boot_path = out_dir / "paired_cluster_bootstrap.json"
    boot_path.write_text(json.dumps(boot, indent=2))
    tables = write_publication_tables(results, boot, capacity=capacity, out_dir=out_dir, fnr_target=fnr_target)
    return {"bootstrap_json": str(boot_path), **tables}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--batch-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--capacity", type=float, required=True)
    ap.add_argument("--bootstrap-reps", type=int, default=5000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260906)
    ap.add_argument("--fnr-target", type=float, default=None)
    args = ap.parse_args()
    artifacts = run_from_files(
        results_path=args.results,
        batch_csv=args.batch_csv,
        out_dir=args.out_dir,
        capacity=args.capacity,
        n_boot=args.bootstrap_reps,
        seed=args.bootstrap_seed,
        fnr_target=args.fnr_target,
    )
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
