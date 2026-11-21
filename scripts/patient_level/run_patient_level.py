#!/usr/bin/env python3
"""Recompute the application from patient-level scores: full empirical frontiers, the frontier decomposition
(Proposition 1) and nested pool-bootstrap intervals (Section 3.2), under the original grid rule and with
order-statistic thresholds.

RUN ONLY INSIDE THE CREDENTIALED ENVIRONMENT. Inputs are patient-level; every output is aggregate.

  python scripts/patient_level/run_patient_level.py --input-dir /secure/exports --out results/11_patient_level --B 2000 --workers 8
  python scripts/patient_level/run_patient_level.py --self-test --out /tmp/pl_selftest      # synthetic end-to-end check
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from patient_level import policy_eval as pe  # noqa: E402
from patient_level.export_schema import NAME_RE, validate_frame, write_comparison  # noqa: E402

TARGETS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12)
F_LO, F_HI, FMAX, MIN_HEADROOM = 0.02, 0.05, 0.25, 0.02
LABEL = {("mimic", "critical"): "MIMIC critical (24 h)", ("mcmed", "critical"): "MC-MED critical", ("mimic", "death30"): "MIMIC death (0–30 d)",
         ("mimic", "disposition"): "MIMIC ED disposition", ("mcmed", "disposition"): "MC-MED ED disposition"}
PANEL = {"cbc": "CBC", "chem7": "chem7"}
ORDER = ["mimic_critical_cbc", "mimic_critical_chem7", "mcmed_critical_cbc", "mcmed_critical_chem7", "mimic_death30_cbc", "mimic_death30_chem7",
         "mimic_disposition_cbc", "mimic_disposition_chem7", "mcmed_disposition_cbc", "mcmed_disposition_chem7"]


def load(path: Path, m: int, K: int):
    df = pd.read_csv(path)
    errs = validate_frame(df, path.name, m, K)
    if errs:
        raise ValueError(f"{path.name}: " + "; ".join(errs))
    site, ep, panel = NAME_RE.match(path.name).groups()
    ar = sorted([c for c in df.columns if c.startswith("A_R_")], key=lambda c: int(c[4:]))
    data = {}
    for s, g in df.groupby("split"):
        g = g.sort_values("pool", kind="stable").reset_index(drop=True)
        data[s] = dict(y=g.y.to_numpy(int), p0=g.p0.to_numpy(float), p1=g.p1.to_numpy(float), AM=g.A_M.to_numpy(bool),
                       pools=np.arange(len(g)).reshape(-1, m), AR=(g[ar].to_numpy(bool).T if ar else None))
    return (site, ep, panel), data


def frontier(y, s):
    """Lower convex hull of the operating characteristic; thresholds at positive scores span every frontier vertex."""
    y = np.asarray(y).ravel(); s = np.asarray(s, float).ravel()
    pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    t = pos[: min(pos.size, int(math.ceil(FMAX * pos.size)) + 1)]
    return pe.lower_hull(np.searchsorted(pos, t, "left") / pos.size, 1.0 - np.searchsorted(neg, t, "left") / neg.size)


def support(*hulls):
    return max(F_LO, *(h[:, 0].min() for h in hulls)), min(F_HI, *(h[:, 0].max() for h in hulls))


def voi(hM, hR, h0, h1):
    lo, hi = support(hM, hR, h0, h1)
    if hi - lo < 0.01:
        return np.nan, np.nan, np.nan, lo, hi
    g = np.linspace(lo, hi, 301); F = lambda h: np.interp(g, h[:, 0], h[:, 1])
    den = float(np.mean(F(h0) - F(h1)))
    if den <= MIN_HEADROOM:
        return np.nan, np.nan, den, lo, hi
    return float(np.mean(F(h0) - F(hR)) / den), float(np.mean(F(h0) - F(hM)) / den), den, lo, hi


def sM(d):
    return np.where(d["AM"], d["p1"], d["p0"])


def sR(d, AR):
    return np.where(AR, d["p1"][None, :], d["p0"][None, :])


def select(rule, pos, q):
    return pe.threshold_orderstat(pos, q) if rule == "orderstat" else pe.threshold_grid(pos, q)


def evaluate(cal, ev, ARc, ARe, q=0.06, rules=("grid", "orderstat"), want_curves=False):
    yc, ye = cal["y"], ev["y"]
    sMc, sRc, sMe, sRe = sM(cal), sR(cal, ARc), sM(ev), sR(ev, ARe)
    yRe = np.broadcast_to(ye, sRe.shape)
    hM, hR, h0, h1 = frontier(ye, sMe), frontier(yRe, sRe), frontier(ye, ev["p0"]), frontier(ye, ev["p1"])
    ilo, ihi = support(hM, hR)
    iso = pe.matched_average(hM, hR, F_LO, F_HI)
    psiR, psiM, head, vlo, vhi = voi(hM, hR, h0, h1)
    out = {}
    for rule in rules:
        tM, tR = select(rule, sMc[yc == 1], q), select(rule, sRc[:, yc == 1], q)
        fM, yM = pe.rates(ye, sMe, tM); fR, yR = pe.rates(yRe, sRe, tR)
        dec = pe.frontier_decomposition(hM, hR, fM, yM, fR, yR)
        out[rule] = dict(tM=tM, tR=tR, fM=fM, fR=fR, yM=yM, yR=yR, iso=iso, ilo=ilo, ihi=ihi, psiR=psiR, psiM=psiM, head=head, vlo=vlo, vhi=vhi,
                         isoM=pe.F(hM, fM) - pe.F(hR, fM), isoR=pe.F(hM, fR) - pe.F(hR, fR), **dec)
    return (out, dict(margin=hM, random=hR, no_panel=h0, full_panel=h1)) if want_curves else out


def operating_points(cal, ev, ARc, ARe):
    yc, ye = cal["y"], ev["y"]
    sc = dict(margin=sM(cal), random=sR(cal, ARc), no_panel=cal["p0"], full_panel=cal["p1"])
    se = dict(margin=sM(ev), random=sR(ev, ARe), no_panel=ev["p0"], full_panel=ev["p1"])
    rows = []
    for pol in sc:
        yev = np.broadcast_to(ye, se[pol].shape)
        for q in TARGETS:
            t = pe.threshold_grid(sc[pol][..., yc == 1], q); f, y = pe.rates(yev, se[pol], t)
            rows.append(dict(policy=pol, target=q, threshold=t, fnr=f, fpr=y))
    return rows


def resample(d, rng):
    J = d["pools"].shape[0]; idx = d["pools"][rng.integers(0, J, J)].ravel()
    out = {k: d[k][idx] for k in ("y", "p0", "p1", "AM")}
    out["pools"] = np.arange(idx.size).reshape(J, -1)
    out["AR"] = d["AR"][:, idx] if d.get("AR") is not None else None
    return out


def _boot(task):
    cal, ev, seeds, q, draws, K, fixed = task
    res = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        c, e = resample(cal, rng), resample(ev, rng)
        ARc = pe.allocate_random(c["pools"], K, rng, c["y"].size, draws)
        ARe = pe.allocate_random(e["pools"], K, rng, e["y"].size, draws)
        try:
            out = evaluate(c, e, ARc, ARe, q)
        except Exception:
            out = None
        sRe = sR(e, e["AR"])          # original-style interval: thresholds and Random draws carried, test pools resampled
        cond = pe.rates(e["y"], sM(e), fixed[0])[1] - pe.rates(np.broadcast_to(e["y"], sRe.shape), sRe, fixed[1])[1]
        res.append((out, cond))
    return res


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v.size >= 10 else (np.nan, np.nan)


def pattern(proc_pp, A_pp):
    if not np.isfinite(A_pp):
        return "n/a"
    return "eliminated" if abs(A_pp) < 0.25 else ("reversed" if np.sign(proc_pp) != np.sign(A_pp) else "preserved")


def validate_against_aggregate(site, ep, panel, split, ops):
    f = ROOT / "results/10_refactor_derived/threshold_selection_audit.csv"
    if not f.is_file() or ep not in ("critical", "death30"):
        return []
    ta = pd.read_csv(f); out = []
    for r in ops:
        q = ta[(ta.split == split) & (ta.site == site) & (ta.endpoint == ep) & (ta.panel == panel) & (ta.policy == r["policy"]) & np.isclose(ta.target, r["target"])]
        if len(q):
            q = q.iloc[0]
            out.append(dict(policy=r["policy"], target=r["target"], threshold_original=float(q.threshold), threshold_recomputed=float(r["threshold"]),
                            threshold_match=bool(abs(q.threshold - r["threshold"]) < 1e-9),
                            fnr_diff_pp=100 * r["fnr"] - float(q.test_fnr_pct), fpr_diff_pp=100 * r["fpr"] - float(q.test_fpr_pct)))
    return out


def run_file(path: Path, a, idx: int) -> dict:
    (site, ep, panel), data = load(path, a.m, a.K)
    rng = np.random.default_rng([a.seed, idx])
    stored = {s: data[s]["AR"] is not None for s in data}
    for s in data:
        if data[s]["AR"] is None:
            data[s]["AR"] = pe.allocate_random(data[s]["pools"], a.K, rng, data[s]["y"].size, a.draws)
    cal = data["cal"]; key = dict(site=site, endpoint=ep, panel=panel)
    out = dict(decomp=[], hulls=[], ops=[], iso=[], validation={})
    for j, split in enumerate(s for s in ("test", "audit") if s in data):
        t0 = time.time(); ev = data[split]
        pt, curves = evaluate(cal, ev, cal["AR"], ev["AR"], a.target, want_curves=True)
        ops = operating_points(cal, ev, cal["AR"], ev["AR"])
        seeds = [int(x.generate_state(1)[0]) for x in np.random.SeedSequence([a.seed, idx, j]).spawn(a.B)]
        chunks = [c for c in np.array_split(np.array(seeds, dtype=np.uint64), max(1, 4 * a.workers)) if len(c)]
        tasks = [(cal, ev, [int(x) for x in c], a.target, a.draws, a.K, (pt["grid"]["tM"], pt["grid"]["tR"])) for c in chunks]
        if a.workers > 1:
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                res = [r for part in ex.map(_boot, tasks) for r in part]
        else:
            res = [r for t in tasks for r in _boot(t)]
        cond = ci([c for _, c in res])
        for rule in ("grid", "orderstat"):
            rep = lambda k: [o[rule][k] if o is not None else np.nan for o, _ in res]
            p = pt[rule]; c = {k: ci(rep(k)) for k in ("proc", "A", "O", "iso", "psiR", "psiM")}
            P = lambda x: 100 * x
            out["decomp"].append(dict(split=split, **key, comparison=f"{LABEL[(site, ep)]}, {PANEL[panel]}", rule=rule,
                source="patient-level full frontier", ci_type="nested pool bootstrap", B=a.B, n_boot_valid=int(np.isfinite(rep("proc")).sum()),
                random_draws_stored=bool(stored[split] and stored["cal"]), threshold_margin=p["tM"], threshold_random=p["tR"],
                same_threshold=bool(abs(p["tM"] - p["tR"]) < 1e-12), fnr_margin_pct=P(p["fM"]), fnr_random_pct=P(p["fR"]),
                fpr_margin_pct=P(p["yM"]), fpr_random_pct=P(p["yR"]), proc_pp=P(p["proc"]), proc_ci_low_pp=P(c["proc"][0]), proc_ci_high_pp=P(c["proc"][1]),
                proc_cond_ci_low_pp=P(cond[0]) if rule == "grid" else np.nan, proc_cond_ci_high_pp=P(cond[1]) if rule == "grid" else np.nan,
                A_matched_pp=P(p["A"]), A_ci_low_pp=P(c["A"][0]), A_ci_high_pp=P(c["A"][1]),
                O_displacement_pp=P(p["O"]), O_ci_low_pp=P(c["O"][0]), O_ci_high_pp=P(c["O"][1]), residual_pp=P(p["residual"]),
                slope_pp_per_pp=p["slope"], iso_at_fM_pp=P(p["isoM"]), iso_at_fR_pp=P(p["isoR"]), iso_range_lo_pct=P(p["ilo"]), iso_range_hi_pct=P(p["ihi"]),
                iso_avg_2to5_pp=P(p["iso"]), iso_ci_low_pp=P(c["iso"][0]), iso_ci_high_pp=P(c["iso"][1]),
                voi_range_lo_pct=P(p["vlo"]), voi_range_hi_pct=P(p["vhi"]), voi_headroom_2to5_pp=P(p["head"]),
                psi_random=p["psiR"], psi_random_ci_low=c["psiR"][0], psi_random_ci_high=c["psiR"][1],
                psi_margin=p["psiM"], psi_margin_ci_low=c["psiM"][0], psi_margin_ci_high=c["psiM"][1], pattern=pattern(P(p["proc"]), P(p["A"]))))
        for pol, h in curves.items():
            out["hulls"] += [dict(split=split, **key, policy=pol, fnr=float(x), fpr=float(y)) for x, y in h]
        out["ops"] += [dict(split=split, **key, **r) for r in ops]
        for f in (0.02, 0.03, 0.04, 0.05):
            v = {pol: 100 * pe.F(h, f) for pol, h in curves.items()}
            out["iso"].append(dict(split=split, **key, fnr_pct=100 * f, **v, margin_minus_random_pp=v["margin"] - v["random"]))
        out["validation"][split] = validate_against_aggregate(site, ep, panel, split, ops)
        print(f"  {path.stem:26s} {split:5s} B={a.B} done in {time.time() - t0:6.1f}s", flush=True)
    return out


def make_synthetic(dir_: Path, seed=1, m=16, K=5):
    rng = np.random.default_rng(seed)
    spec = {"mimic_critical_cbc": (0.10, 0.8), "mimic_critical_chem7": (0.10, 1.0), "mcmed_critical_cbc": (0.05, 0.1),
            "mcmed_critical_chem7": (0.05, 0.5), "mimic_death30_cbc": (0.04, 0.8), "mimic_death30_chem7": (0.04, 0.9)}
    for name, (prev, info) in spec.items():
        splits = {}
        for s, J in (("cal", 120), ("test", 200), ("audit", 100)):
            if s == "audit" and "death30" not in name:
                continue
            n = J * m; x = rng.normal(size=n); z = rng.normal(size=n)
            lin0 = math.log(prev / (1 - prev)) - 0.6 + 1.4 * x; lin = lin0 + info * z
            y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
            p0 = 1 / (1 + np.exp(-(lin0 + 0.2))); p1 = 1 / (1 + np.exp(-lin))
            pools = np.arange(n).reshape(J, m)
            AM = pe.allocate_margin(p0, pools, K, float(np.quantile(p0, 0.5)))
            AR = pe.allocate_random(pools, K, rng, n, 10)
            splits[s] = dict(pool=np.repeat(np.arange(J), m), y=y, p0=p0, p1=p1, A_M=AM.astype(int), A_R=AR.astype(int))
        write_comparison(dir_ / f"{name}.csv", splits, m, K)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir"); ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=None, help="bootstrap replicates (default 2000; 40 in --self-test)")
    ap.add_argument("--seed", type=int, default=20261119); ap.add_argument("--target", type=float, default=0.06)
    ap.add_argument("--draws", type=int, default=10, help="Random draws when A_R columns are absent and in every replicate")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--m", type=int, default=16); ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--only", nargs="*", help="restrict to these comparison names, e.g. mimic_death30_chem7")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.self_test:
        inp = out / "_synthetic_inputs"; inp.mkdir(exist_ok=True); make_synthetic(inp); a.input_dir = str(inp)
        a.B = a.B or 40; mode = "SELF-TEST (synthetic data; not for the manuscript)"
    else:
        if not a.input_dir:
            ap.error("--input-dir is required (or use --self-test)")
        a.B = a.B or 2000; mode = "patient-level"
    files = sorted(Path(a.input_dir).glob("*.csv"), key=lambda p: ORDER.index(p.stem) if p.stem in ORDER else 99)
    files = [f for f in files if not a.only or f.stem in a.only]
    if not files:
        sys.exit("no input CSV files found")
    print(f"{mode}: {len(files)} comparisons, B={a.B}, workers={a.workers}", flush=True)
    t0 = time.time(); acc = dict(decomp=[], hulls=[], ops=[], iso=[]); validation = {}
    for i, f in enumerate(files):
        r = run_file(f, a, ORDER.index(f.stem) if f.stem in ORDER else 100 + i)
        for k in acc:
            acc[k] += r[k]
        validation[f.stem] = r["validation"]
    names = dict(decomp="patient_level_decomposition.csv", hulls="patient_level_frontier_hulls.csv",
                 ops="patient_level_operating_points.csv", iso="patient_level_iso_fpr_2to5.csv")
    for k, fn in names.items():
        pd.DataFrame(acc[k]).to_csv(out / fn, index=False)
    flat = [dict(comparison=c, split=s, **v) for c, d in validation.items() for s, rows in d.items() for v in rows]
    v6 = [x for x in flat if abs(x["target"] - a.target) < 1e-12 and x["policy"] in ("margin", "random")]
    summary = dict(n_checked=len(flat), all_thresholds_match=bool(flat) and all(x["threshold_match"] for x in flat),
                   thresholds_match_at_target=bool(v6) and all(x["threshold_match"] for x in v6),
                   max_abs_fnr_diff_pp=max((abs(x["fnr_diff_pp"]) for x in flat), default=None),
                   max_abs_fpr_diff_pp=max((abs(x["fpr_diff_pp"]) for x in flat), default=None))
    (out / "validation_report.json").write_text(json.dumps(dict(summary=summary, rows=flat,
        note="Recomputed grid-rule thresholds and test operating points versus results/10_refactor_derived/threshold_selection_audit.csv. "
             "Exact agreement is expected when the original A_M and A_R columns were exported from the same cal/test sets."), indent=1))
    info = dict(mode=mode, B=a.B, seed=a.seed, target=a.target, draws=a.draws, m=a.m, K=a.K, workers=a.workers, fmax=FMAX,
                fnr_range=[F_LO, F_HI], runtime_s=round(time.time() - t0, 1), python=platform.python_version(), numpy=np.__version__,
                inputs={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (out / "run_info.json").write_text(json.dumps(info, indent=1))
    if a.self_test:
        shutil.rmtree(Path(a.input_dir), ignore_errors=True)
    print(f"done in {info['runtime_s']} s -> {out}\nvalidation: {summary}")


if __name__ == "__main__":
    main()
