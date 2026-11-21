"""Aggregate-level derived analyses for the Statistics in Medicine refactor.

Reads only files under results/ (no patient-level data). Writes CSV tables to
results/10_refactor_derived/ and a flat numbers.json that is the single source of every
number quoted in the manuscript templates.

Main constructs (see manuscript Section 3):
  * lower convex hull of each policy's realised (FNR, FPR) test operating points; any point
    on it is attainable by randomising between two thresholds;
  * frontier (matched-sensitivity) decomposition of the nominal-target procedure contrast
    (Proposition 1): Delta_proc = A + O + residual;
  * value-of-information share psi_P over FNR 2-5%;
  * threshold-grid audit (Proposition 2) and closed-form label-flip check (Proposition 3);
  * evaluation-gap identity summaries for the synthetic study (Theorem 1, Corollaries 1-2).
"""
from __future__ import annotations
import json, math, os
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from .paths import RESULTS, DERIVED

PI = 5 / 16
GRID_START, GRID_STEP = 0.001, 0.0025
F_LO, F_HI = 0.02, 0.05
F_GRID = np.round(np.linspace(F_LO, F_HI, 301), 8)
POLICY = {"margin_global": "margin", "random": "random", "free_only": "no_panel", "cbc_all": "full_panel",
          "panel_all": "full_panel", "entropy": "entropy", "decision_risk_global": "decision_risk"}
CAL_POS = {("mimic", "critical"): 1151, ("mcmed", "critical"): 202, ("mimic", "death30"): 304}
COMPARISONS = [("mimic", "critical", "cbc"), ("mimic", "critical", "chem7"), ("mcmed", "critical", "cbc"),
               ("mcmed", "critical", "chem7"), ("mimic", "death30", "cbc"), ("mimic", "death30", "chem7")]
ENDPOINT_LABEL = {("mimic", "critical"): "MIMIC critical (24 h)", ("mcmed", "critical"): "MC-MED critical",
                  ("mimic", "death30"): "MIMIC death (0–30 d)"}
PANEL_LABEL = {"cbc": "CBC", "chem7": "chem7"}
KEY = {("mimic", "critical"): "mimic_crit", ("mcmed", "critical"): "mcmed_crit", ("mimic", "death30"): "mimic_death"}
ENDPOINT_LABEL.update({("mimic", "disposition"): "MIMIC ED disposition", ("mcmed", "disposition"): "MC-MED ED disposition"})
KEY.update({("mimic", "disposition"): "mimic_disp", ("mcmed", "disposition"): "mcmed_disp"})
MAIN_EP = ("critical", "death30")
PL_DIR = RESULTS / "11_patient_level"


def label(site, ep, panel):
    return f"{ENDPOINT_LABEL[(site, ep)]}, {PANEL_LABEL[panel]}"


def fmt(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "−")


# ----------------------------------------------------------------------------- frontiers
def load_frontiers() -> pd.DataFrame:
    e = pd.read_csv(RESULTS / "02_operating_points/E2_all_methods_six_targets.csv").rename(
        columns={"nominal_calibration_fnr_target": "target"})
    e["endpoint"] = "critical"
    frames = [e]
    for split in ("test", "audit"):
        m = pd.read_csv(RESULTS / f"02_operating_points/mortality_{split}_all_methods_six_targets.csv").rename(
            columns={"nominal_validation_fnr_target": "target"})
        m["endpoint"] = "death30"
        m["split"] = split
        frames.append(m)
    d = pd.concat(frames, ignore_index=True)
    d["policy"] = d["method"].map(POLICY)
    assert d.policy.notna().all(), d.method.unique()
    cov = d.test_coverage_fraction
    assert np.allclose(cov[d.policy == "no_panel"], 0) and np.allclose(cov[d.policy == "full_panel"], 1)
    assert np.allclose(cov[d.policy.isin(["margin", "random"])], PI)
    d["fnr"], d["fpr"] = d.test_fnr_estimate, d.test_fpr_estimate
    return d


def lower_hull(x, y) -> np.ndarray:
    """Lower convex hull of points (x=FNR, y=FPR), x ascending (Andrew's monotone chain)."""
    df = pd.DataFrame({"x": np.round(np.asarray(x, float), 12), "y": np.round(np.asarray(y, float), 12)})
    df = df.groupby("x", as_index=False).y.min().sort_values("x")
    hull: list[tuple[float, float]] = []
    for p in df.itertuples(index=False, name=None):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return np.array(hull)


def H(curve: np.ndarray, f: float) -> float:
    x, y = curve[:, 0], curve[:, 1]
    if f < x.min() - 1e-12 or f > x.max() + 1e-12:
        return float("nan")
    return float(np.interp(f, x, y))


def frontier_curves(d: pd.DataFrame) -> dict:
    out = {}
    for (split, site, ep, panel, pol), g in d.groupby(["split", "site", "endpoint", "panel", "policy"]):
        out[(split, site, ep, panel, pol)] = lower_hull(g.fnr.values, g.fpr.values)
    return out


def _paired_ci(split, site, ep, panel, target=0.06):
    if split == "test":
        main = pd.read_csv(RESULTS / "01_primary/ALL_main_policy_contrasts.csv")
        epname = "E2_site_specific_critical" if ep == "critical" else "mortality_calendar_day_0_to_30"
        r = main[(main.site == site) & (main.endpoint == epname) & (main.panel == panel) & (main.metric == "fpr")].iloc[0]
        return float(r.delta_pp), float(r.ci_low_pp), float(r.ci_high_pp)
    p = pd.read_csv(RESULTS / "02_operating_points/mortality_audit_paired_contrasts_six_targets.csv")
    r = p[(p.metric == "fpr") & (p.contrast == f"{panel}_margin_minus_random") &
          np.isclose(p.nominal_validation_fnr_target, target)].iloc[0]
    return 100 * r.estimate, 100 * r.ci_lower, 100 * r.ci_upper


def nominal_decomposition(d, curves, target=0.06) -> pd.DataFrame:
    rows = []
    for split in ("test", "audit"):
        for site, ep, panel in COMPARISONS:
            if split == "audit" and ep != "death30":
                continue
            q = d[(d.split == split) & (d.site == site) & (d.endpoint == ep) & (d.panel == panel) & np.isclose(d.target, target)]
            M, R = q[q.policy == "margin"].iloc[0], q[q.policy == "random"].iloc[0]
            c = {pol: curves[(split, site, ep, panel, pol)] for pol in ("margin", "random", "no_panel", "full_panel")}
            fM, yM, fR, yR = M.fnr, M.fpr, R.fnr, R.fpr
            a, b, cc, dd = H(c["margin"], fM), H(c["margin"], fR), H(c["random"], fM), H(c["random"], fR)
            A = 0.5 * ((a - cc) + (b - dd))
            O = 0.5 * ((a - b) + (cc - dd))
            res = (yM - a) - (yR - dd)
            proc = yM - yR
            assert abs(proc - (A + O + res)) < 1e-12
            sbar = -O / (fM - fR) if abs(fM - fR) > 1e-9 else float("nan")
            def support(pols):
                lo_ = max(F_LO, max(c[q][:, 0].min() for q in pols)); hi_ = min(F_HI, min(c[q][:, 0].max() for q in pols))
                return lo_, hi_
            lo_f, hi_f = support(("margin", "random"))
            assert hi_f - lo_f >= 0.02, (split, site, ep, panel, lo_f, hi_f)
            grid = np.linspace(lo_f, hi_f, 301)
            iso_avg = float(np.mean([H(c["margin"], f) - H(c["random"], f) for f in grid]))
            lo4, hi4 = support(("margin", "random", "no_panel", "full_panel"))
            if hi4 - lo4 >= 0.01:
                g4 = np.linspace(lo4, hi4, 301)
                cg = {pol: np.array([H(c[pol], f) for f in g4]) for pol in c}
                den = float(np.mean(cg["no_panel"] - cg["full_panel"]))
                psiR = float(np.mean(cg["no_panel"] - cg["random"]) / den) if den > 0.02 else float("nan")
                psiM = float(np.mean(cg["no_panel"] - cg["margin"]) / den) if den > 0.02 else float("nan")
            else:
                den = psiR = psiM = float("nan")
            pp, lo, hi = _paired_ci(split, site, ep, panel, target)
            assert abs(pp - 100 * proc) < 1e-6, (pp, proc)
            rows.append(dict(split=split, site=site, endpoint=ep, panel=panel, comparison=label(site, ep, panel),
                             threshold_margin=M.validation_threshold, threshold_random=R.validation_threshold,
                             same_threshold=bool(np.isclose(M.validation_threshold, R.validation_threshold)),
                             fnr_margin_pct=100 * fM, fnr_random_pct=100 * fR, fpr_margin_pct=100 * yM, fpr_random_pct=100 * yR,
                             proc_pp=100 * proc, proc_ci_low_pp=lo, proc_ci_high_pp=hi,
                             A_matched_pp=100 * A, O_displacement_pp=100 * O, residual_pp=100 * res,
                             slope_pp_per_pp=sbar, iso_at_fM_pp=100 * (a - cc), iso_at_fR_pp=100 * (b - dd),
                             iso_range_lo_pct=100 * lo_f, iso_range_hi_pct=100 * hi_f, iso_avg_2to5_pp=100 * iso_avg,
                             voi_range_lo_pct=100 * lo4, voi_range_hi_pct=100 * hi4, voi_headroom_2to5_pp=100 * den, psi_random=psiR, psi_margin=psiM,
                             sign_reversal=bool(np.sign(proc) != np.sign(A))))
    out = pd.DataFrame(rows)
    out["pattern"] = np.where(out.A_matched_pp.abs() < 0.25, "eliminated",
                              np.where(np.sign(out.proc_pp) != np.sign(out.A_matched_pp), "reversed", "preserved"))
    return out


def iso_long(d, curves) -> pd.DataFrame:
    rows = []
    for (split, site, ep, panel, pol), cv in curves.items():
        for f in (0.02, 0.03, 0.04, 0.05):
            rows.append(dict(split=split, site=site, endpoint=ep, panel=panel, policy=pol, fnr_pct=100 * f, fpr_hull_pct=100 * H(cv, f)))
    t = pd.DataFrame(rows).pivot_table(index=["split", "site", "endpoint", "panel", "fnr_pct"], columns="policy", values="fpr_hull_pct").reset_index()
    t["margin_minus_random_pp"] = t.margin - t.random
    return t


def hull_table(curves) -> pd.DataFrame:
    rows = []
    for (split, site, ep, panel, pol), cv in curves.items():
        for x, y in cv:
            rows.append(dict(split=split, site=site, endpoint=ep, panel=panel, policy=pol, fnr=x, fpr=y))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ threshold-grid audit
def threshold_audit(d):
    rows = []
    for (split, site, ep, panel), g in d.groupby(["split", "site", "endpoint", "panel"]):
        n1 = CAL_POS[(site, ep)]
        for pol in ("margin", "random", "no_panel", "full_panel"):
            gp = g[g.policy == pol].sort_values("target")
            ndist = gp.validation_threshold.round(6).nunique()
            for r in gp.itertuples():
                k = (r.validation_threshold - GRID_START) / GRID_STEP
                nxt = gp[np.isclose(gp.validation_threshold, r.validation_threshold + GRID_STEP)]
                fn_next = float(nxt.validation_fnr.iloc[0] * n1) if len(nxt) else float("nan")
                allow = math.floor(r.target * n1 + 1e-9)
                rows.append(dict(split=split, site=site, endpoint=ep, panel=panel, policy=pol, target=r.target,
                                 threshold=r.validation_threshold, grid_index=int(round(k)), on_grid=bool(abs(k - round(k)) < 1e-6),
                                 calibration_positives=n1, calibration_fn=r.validation_fnr * n1, allowance_fn=allow,
                                 calibration_fnr_pct=100 * r.validation_fnr, slack_pp=100 * (r.target - r.validation_fnr),
                                 next_grid_calibration_fn=fn_next, next_grid_excess_fn=fn_next - allow if len(nxt) else float("nan"),
                                 test_fnr_pct=100 * r.fnr, test_fpr_pct=100 * r.fpr,
                                 test_fpr_jump_to_next_grid_pp=100 * (r.fpr - nxt.fpr.iloc[0]) if len(nxt) else float("nan"),
                                 n_distinct_thresholds_over_6_targets=ndist))
    ta = pd.DataFrame(rows)
    mr = ta[ta.policy.isin(["margin", "random"])].pivot_table(
        index=["split", "site", "endpoint", "panel", "target"], columns="policy", values="threshold").reset_index()
    mr["same_threshold"] = np.isclose(mr.margin, mr.random)
    return ta, mr


def threshold_component_check(mr) -> pd.DataFrame:
    dec = pd.read_csv(RESULTS / "03_decomposition/ALL_5_experiments_6_targets_summary.csv")
    amap = {"critical_mimic_test": ("test", "mimic", "critical"), "critical_mcmed_test": ("test", "mcmed", "critical"),
            "mortality_test": ("test", "mimic", "death30"), "mortality_audit": ("audit", "mimic", "death30")}
    dec = dec[dec.analysis.isin(amap) & (dec.metric == "fpr")].copy()
    dec = dec.drop(columns=["site", "endpoint", "split"])
    dec[["split", "site", "endpoint"]] = pd.DataFrame(dec.analysis.map(amap).tolist(), index=dec.index)
    dec["target"] = dec.nominal_validation_fnr_target.round(4)
    mr = mr.assign(target=mr.target.round(4))
    m = dec.merge(mr, on=["split", "site", "endpoint", "panel", "target"], how="left", validate="one_to_one")
    m["threshold_component_zero"] = m.threshold_pp.abs() < 1e-9
    m["consistent"] = m.threshold_component_zero == m.same_threshold
    return m[["analysis", "split", "site", "endpoint", "panel", "target", "margin", "random", "same_threshold",
              "observed_pp", "allocation_pp", "threshold_pp", "ci_low_threshold_pp", "ci_high_threshold_pp",
              "threshold_component_zero", "consistent"]]


# ------------------------------------------------------------ closed-form label flips
def label_flip_check() -> pd.DataFrame:
    ex = pd.read_csv(RESULTS / "05_label_sensitivity/exact_mortality_audit_200_rows.csv")
    disc = ex[ex.bound_type == "common_label"].groupby(["panel", "random_repeat"])[
        ["n_discordant_margin_worse", "n_discordant_margin_better"]].first()
    out = []
    for r in ex.itertuples():
        Dw, Db = disc.loc[(r.panel, r.random_repeat)]
        n, delta = r.n, r.observed_margin_minus_random
        if r.bound_type == "common_label":
            k = r.max_flips
            lo, hi = delta - 2 * min(k, Dw) / n, delta + 2 * min(k, Db) / n
        else:
            k = r.max_flips_margin
            lo = delta - min(k / n, r.error_margin) - min(k / n, 1 - r.error_random)
            hi = delta + min(k / n, 1 - r.error_margin) + min(k / n, r.error_random)
        minflip = math.ceil((Db - Dw) / 2) if delta < 0 else math.ceil((Dw - Db) / 2)
        out.append(dict(panel=r.panel, random_repeat=r.random_repeat, rho=r.assumed_label_flip_fraction, bound_type=r.bound_type,
                        n=n, D_margin_worse=Dw, D_margin_better=Db, delta=delta, delta_from_discordance=(Dw - Db) / n,
                        lower_exact=r.lower, upper_exact=r.upper, lower_closed=lo, upper_closed=hi,
                        abs_dev=max(abs(lo - r.lower), abs(hi - r.upper)), certified_exact=bool(r.ranking_certified),
                        certified_closed=bool(lo > 0 or hi < 0), minflips_exact=r.min_flips_to_non_strict_reversal,
                        minflips_closed=minflip))
    return pd.DataFrame(out)


# --------------------------------------------------------------------- simulation
def tci(a):
    a = np.asarray(a, float); n = len(a); m = float(a.mean()); sd = float(a.std(ddof=1))
    h = float(student_t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    return m, sd, sd / math.sqrt(n), m - h, m + h


def simulation_tables():
    s = pd.read_csv(RESULTS / "06_simulation/confirmatory_3x3_100_replicate_metrics_900.csv")
    assert np.allclose(s.group_margin_only_fraction, s.group_random_only_fraction)
    g = lambda k, v: s[f"group_{k}_{v}"]
    s["g0_pp"] = np.where(g("random_only", "fraction") > 0, 100 * g("random_only", "net_gap") / g("random_only", "fraction"), np.nan)
    s["g1_pp"] = np.where(g("margin_only", "fraction") > 0, -100 * g("margin_only", "net_gap") / g("margin_only", "fraction"), np.nan)
    s["share"] = g("margin_only", "fraction")
    s["identity_resid"] = 100 * s.retrospective_to_deployment_gap - s.share * (s.g0_pp - s.g1_pp)
    p0 = (g("neither", "fraction") * g("neither", "label_change_rate_within") + g("random_only", "fraction") * g("random_only", "label_change_rate_within")) / (g("neither", "fraction") + g("random_only", "fraction"))
    p1 = (g("both", "fraction") * g("both", "label_change_rate_within") + g("margin_only", "fraction") * g("margin_only", "label_change_rate_within")) / (g("both", "fraction") + g("margin_only", "fraction"))
    s["bound_corollary2_pp"] = 100 * PI * (1 - PI) * (p0 + p1)
    s["bound_strata_pp"] = 100 * sum(g(k, "absolute_gap") for k in ("both", "neither", "margin_only", "random_only"))
    s["bound_naive_pp"] = 100 * s.bound_rho_sum
    rows = []
    for (a, gm), x in s.groupby(["alpha", "gamma"]):
        B = 100 * x.retrospective_to_deployment_gap
        m, sd, mc, lo, hi = tci(B) if gm > 0 else (0.0, 0.0, 0.0, 0.0, 0.0)
        rev = float(x.policy_ranking_strict_reversal.mean())
        rows.append(dict(alpha=a, gamma=gm, n_reps=len(x), label_change_pct=100 * x.observed_D1_vs_D0_change.mean(),
                         B_mean_pp=m, B_sd_pp=sd, B_mcse_pp=mc, B_ci_low_pp=lo, B_ci_high_pp=hi,
                         mean_abs_B_pp=float(B.abs().mean()), g0_pp=float(x.g0_pp.mean()) if gm > 0 else 0.0,
                         g1_pp=float(x.g1_pp.mean()) if gm > 0 else 0.0, share_singly_each_pct=100 * x.share.mean(),
                         max_abs_identity_resid=float(np.nanmax(np.abs(x.identity_resid))) if gm > 0 else 0.0,
                         bound_naive_pp=x.bound_naive_pp.mean(), bound_corollary2_pp=x.bound_corollary2_pp.mean(),
                         bound_strata_pp=x.bound_strata_pp.mean(), bound_always_holds=bool((B.abs() <= x.bound_strata_pp + 1e-9).all()),
                         retro_delta_pp=100 * x.delta_retrospective_error.mean(), deploy_delta_pp=100 * x.delta_deployment_error.mean(),
                         strict_reversal_pct=100 * rev, strict_reversal_mcse_pct=100 * math.sqrt(rev * (1 - rev) / len(x)),
                         tie_changes=int(x.policy_ranking_tie_change.sum())))
    cells = pd.DataFrame(rows)
    neither_draws = float((g("neither", "fraction") * s.n * 3).sum())
    neither_disagree = float((g("neither", "fraction") * g("neither", "prediction_disagreement_rate_within") * s.n * 3).sum())
    g1 = s[s.gamma == 1]
    per_rep = g1.groupby("rep").agg(B=("retrospective_to_deployment_gap", "mean"), g0=("g0_pp", "mean"), g1=("g1_pp", "mean"),
                                    share=("share", "mean"), chg=("observed_D1_vs_D0_change", "mean"),
                                    naive=("bound_naive_pp", "mean"), c2=("bound_corollary2_pp", "mean"), strata=("bound_strata_pp", "mean"))
    pooled = dict(B=tci(100 * per_rep.B), g0=tci(per_rep.g0), g1=tci(per_rep.g1), share=100 * per_rep.share.mean(),
                  chg=100 * per_rep.chg.mean(), naive=per_rep.naive.mean(), c2=per_rep.c2.mean(), strata=per_rep.strata.mean())
    return s, cells, dict(neither_draws=neither_draws, neither_disagree=neither_disagree), pooled


# ------------------------------------------------------- patient-level results (optional)
ORDER_ALL = COMPARISONS + [("mimic", "disposition", "cbc"), ("mimic", "disposition", "chem7"), ("mcmed", "disposition", "cbc"), ("mcmed", "disposition", "chem7")]


def patient_level_outputs():
    """Outputs of scripts/patient_level/run_patient_level.py, if present (self-test output is ignored unless allowed)."""
    f = PL_DIR / "patient_level_decomposition.csv"
    if not f.is_file():
        return None
    info = json.loads((PL_DIR / "run_info.json").read_text()) if (PL_DIR / "run_info.json").is_file() else {}
    if info.get("mode") != "patient-level" and os.environ.get("SHAREDCAP_ALLOW_SELFTEST") != "1":
        print("NOTE: results/11_patient_level holds self-test output and is ignored (SHAREDCAP_ALLOW_SELFTEST=1 to test the pipeline)")
        return None
    return dict(decomp=pd.read_csv(f), hulls=pd.read_csv(PL_DIR / "patient_level_frontier_hulls.csv"),
                ops=pd.read_csv(PL_DIR / "patient_level_operating_points.csv"), info=info)


def _order(df):
    rank = {c: i for i, c in enumerate(ORDER_ALL)}
    o = [(0 if s == "test" else 1, rank.get((a, b, c), 99)) for s, a, b, c in df[["split", "site", "endpoint", "panel"]].itertuples(index=False)]
    return df.assign(_o=o).sort_values("_o").drop(columns="_o").reset_index(drop=True)


def merge_patient_level(nom_agg, pl):
    if pl is None:
        return nom_agg
    key = ["split", "site", "endpoint", "panel"]
    p = pl["decomp"][pl["decomp"].rule == "grid"]
    rest = nom_agg[~pd.MultiIndex.from_frame(nom_agg[key]).isin(pd.MultiIndex.from_frame(p[key]))]
    return _order(pd.concat([p, rest], ignore_index=True))


def frontier_inputs(pl="auto"):
    """Operating points and frontiers for tables and figures: patient-level full frontiers where available."""
    d = load_frontiers(); curves = frontier_curves(d)
    pl = patient_level_outputs() if isinstance(pl, str) else pl
    if pl is None:
        return d, curves
    key = ["split", "site", "endpoint", "panel"]
    for k, g in pl["hulls"].groupby(key + ["policy"]):
        curves[tuple(k)] = g.sort_values("fnr")[["fnr", "fpr"]].to_numpy()
    ops = pl["ops"]
    keep = ~pd.MultiIndex.from_frame(d[key]).isin(pd.MultiIndex.from_frame(ops[key].drop_duplicates()))
    return pd.concat([d[keep], ops], ignore_index=True), curves


W_ = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def patient_level_sentences(test, pl):
    """Data-driven sentences that exist only when patient-level results with nested intervals are present."""
    S = {"S.abs_tail": "", "S.nested_block": "", "S.orderstat_block": "", "S.disc_nested": "", "S.s12_block": "",
         "S.tau_sentence": ("the sensitivity analysis of Remark 1 requires the rates $u_0$ and $u_1$, which can be computed only from patient-level data "
                            "[AUTHOR: rerun scripts/patient_level/run_patient_level.py, which reports $u_0$, $u_1$ and $B(\\tau_0)$].")}
    r = test[(test.site == "mimic") & (test.endpoint == "death30") & (test.panel == "cbc")]
    S["S.death_cbc_frac"] = f"{100 * float(r.A_matched_pp.iloc[0] / r.proc_pp.iloc[0]):.0f}%" if len(r) else "n/a"
    if pl is None or not test.ci_type.astype(str).str.startswith("nested").all():
        return S
    inc = lambda lo, hi: (lo <= 0) & (hi >= 0)
    wn = test.proc_ci_high_pp - test.proc_ci_low_pp; wf = test.proc_cond_ci_high_pp - test.proc_cond_ci_low_pp
    n_nest = int((~inc(test.proc_ci_low_pp, test.proc_ci_high_pp)).sum()); n_fix = int((~inc(test.proc_cond_ci_low_pp, test.proc_cond_ci_high_pp)).sum())
    ratio = f"{float(np.median(wn / wf)):.0f}"
    nest_phrase = ("no procedure contrast was distinguishable from zero" if n_nest == 0 else
                   f"only {W_[n_nest]} procedure contrast{'s' if n_nest > 1 else ''} remained distinguishable from zero")
    S["S.abs_tail"] = f"; with calibration variability propagated, {nest_phrase}"
    aex = test[~inc(test.A_ci_low_pp, test.A_ci_high_pp)]
    alist = " and ".join(f"{c} ($A$ = {fmt(a)} pp [95% CI {fmt(l)} to {fmt(h)}])" for c, a, l, h in
                         zip(aex.comparison, aex.A_matched_pp, aex.A_ci_low_pp, aex.A_ci_high_pp)) or "none of the comparisons"
    S["S.nested_block"] = (f"Propagating calibration-set variability changed the assessment of uncertainty (Table 3; Supporting Information, Table S3). "
        f"Fixed-threshold intervals for the six procedure contrasts were {fmt(wf.min(), 1, False)}–{fmt(wf.max(), 1, False)} pp wide and "
        f"{'all six' if n_fix == 6 else W_[n_fix]} excluded zero, whereas the nested intervals were {fmt(wn.min(), 1, False)}–{fmt(wn.max(), 1, False)} pp wide, "
        f"a median of {ratio} times wider, and {W_[n_nest]} excluded zero. The matched-sensitivity components were estimated more stably: "
        f"their nested intervals excluded zero for {alist}.")
    S["S.disc_nested"] = (f" In our application, propagating calibration-set variability widened the intervals for the procedure contrasts by a median factor of {ratio}, "
        f"so that {'none of them' if n_nest == 0 else 'only ' + W_[n_nest] + ' of them'} excluded zero, whereas matched-sensitivity contrasts remained estimable.")
    os_ = pl["decomp"][(pl["decomp"].rule == "orderstat") & (pl["decomp"].split == "test") & pl["decomp"].endpoint.isin(MAIN_EP)]
    g_ = test.set_index(["site", "endpoint", "panel"]); o_ = os_.set_index(["site", "endpoint", "panel"])
    common = g_.index.intersection(o_.index)
    if len(common):
        dA = (o_.loc[common, "A_matched_pp"] - g_.loc[common, "A_matched_pp"]).abs(); dP = (o_.loc[common, "proc_pp"] - g_.loc[common, "proc_pp"]).abs()
        flips = [k for k in common if np.sign(o_.loc[k, "proc_pp"]) != np.sign(g_.loc[k, "proc_pp"])]
        flip_txt = "; ".join(f"{g_.loc[k, 'comparison']}, from {fmt(g_.loc[k, 'proc_pp'])} to {fmt(o_.loc[k, 'proc_pp'])} pp" for k in flips)
        S["S.orderstat_block"] = (f" On the test sets, replacing the grid by order-statistic thresholds changed the matched-sensitivity components by at most {fmt(dA.max(), 2, False)} pp "
            f"but the procedure contrasts by up to {fmt(dP.max(), 2, False)} pp" + (f", reversing the sign for {flip_txt}" if flips else "") +
            " (Supporting Information, Table S3).")
    tf = PL_DIR / "patient_level_tau_sensitivity.csv"
    if tf.is_file():
        t = pd.read_csv(tf); tt = t[(t.split == "test") & t.endpoint.isin(MAIN_EP)]
        band = "band_violation_pct" in t.columns
        band_txt = ""
        if band:
            vmax = float(tt.band_violation_pct.max())
            band_txt = ("; the band condition of Corollary 1 held in every comparison" if vmax == 0 else
                        f"; the band condition of Corollary 1 failed for up to {vmax:.2g}% of the encounters that Margin left unallocated, "
                        "so these values were computed from Theorem 1 rather than equation (9)")
        S["S.tau_sentence"] = (f"for these comparisons the rates of equation (9) gave $u_0-u_1$ between {fmt(tt.u0_minus_u1.min(), 2)} and {fmt(tt.u0_minus_u1.max(), 2)}, "
            f"so that a constant effect of panel availability on the label of $\\tau_0=0.05$ would change the Margin–Random contrast in cost-weighted risk by "
            f"{fmt(tt['B_tau_0.05_pp'].min(), 2)} to {fmt(tt['B_tau_0.05_pp'].max(), 2)} pp{band_txt} (Supporting Information, Section S13).")
        rows = "\n".join(f"| {r.split} | {r.comparison} | {r.u0:.3f} | {r.u1:.3f} | {fmt(r.u0_minus_u1, 3)} |"
                         + (f" {r.band_violation_pct:.2f} |" if band else "")
                         + f" {fmt(r['B_tau_0.01_pp'], 3)} | {fmt(r['B_tau_0.02_pp'], 3)} | {fmt(r['B_tau_0.05_pp'], 3)} |" for _, r in t.iterrows())
        hdr = ("| Set | Comparison | $u_0$ | $u_1$ | $u_0-u_1$ |" + (" Band violations (%) |" if band else "")
               + " $\\tau_0=0.01$ | $\\tau_0=0.02$ | $\\tau_0=0.05$ |\n"
               + "|" + "|".join(["-----", "--------------------------", "------", "------", "-------"] + (["--------"] if band else []) + ["-------"] * 3) + "|\n")
        intro = ("# S13 SENSITIVITY ANALYSIS FOR DECISION-DEPENDENT LABELS\n\nTable S11 applies Remark 1 of the main article: with a constant effect $\\tau_0$ of panel "
            "availability on the label, $B=\\tau_0\\pi(1-\\pi)(u_0-u_1)$ for cost-weighted loss, where $u_a$ is the rate of positive no-panel predictions at the Random "
            "threshold among encounters that Margin leaves unallocated ($a=0$) or allocates ($a=1$). For 0–1 loss the values double.")
        if band:
            intro += (" This expression requires the band condition of Corollary 1. The column “Band violations” gives the percentage of encounters left unallocated "
                      "by Margin whose no-panel predictions differ between the Margin and Random thresholds. The values of $B$ are computed from Theorem 1 directly, "
                      "$B=\\tau_0(1-\\pi)\\{\\bar u_0^{M}-(1-\\pi)u_0-\\pi u_1\\}$, where $\\bar u_0^{M}$ is the rate of positive no-panel predictions at the Margin "
                      "threshold among encounters that Margin leaves unallocated; they equal the expression above when there are no violations.")
        S["S.s12_block"] = (intro + "\n\nTable: **Table S11.** Sensitivity of the Margin–Random contrast to decision-dependent labels (pp of cost-weighted risk).\n\n"
                            + hdr + rows + "\n")
    return S


# --------------------------------------------------------------------- run all
def run() -> dict:
    d = load_frontiers()
    curves = frontier_curves(d)
    nom_agg = nominal_decomposition(d, curves).assign(rule="grid", source="aggregate six-point hull",
                                                      ci_type="fixed-threshold paired bootstrap")
    nom_agg["proc_cond_ci_low_pp"], nom_agg["proc_cond_ci_high_pp"] = nom_agg.proc_ci_low_pp, nom_agg.proc_ci_high_pp
    nom_agg.to_csv(DERIVED / "frontier_decomposition_nominal6_aggregate.csv", index=False)
    pl = patient_level_outputs()
    nom = merge_patient_level(nom_agg, pl)
    nom.to_csv(DERIVED / "frontier_decomposition_nominal6.csv", index=False)
    d_fig, curves_fig = frontier_inputs(pl)
    iso_long(d_fig, curves_fig).to_csv(DERIVED / "iso_sensitivity_fpr_at_fnr_2_to_5.csv", index=False)
    hull_table(curves_fig).to_csv(DERIVED / "frontier_lower_hulls.csv", index=False)
    ta, mr = threshold_audit(d)
    ta.to_csv(DERIVED / "threshold_selection_audit.csv", index=False)
    mr.to_csv(DERIVED / "threshold_coincidence_margin_random.csv", index=False)
    tc = threshold_component_check(mr)
    tc.to_csv(DERIVED / "threshold_space_component_zero_check.csv", index=False)
    lf = label_flip_check()
    lf.to_csv(DERIVED / "label_flip_closed_form_check.csv", index=False)
    s, cells, band, pooled = simulation_tables()
    cells.to_csv(DERIVED / "simulation_identity_bounds_by_cell.csv", index=False)

    N: dict[str, str] = {}
    # ---- application, per comparison
    for r in nom.itertuples():
        k = ("audit_" if r.split == "audit" else "") + f"{KEY[(r.site, r.endpoint)]}_{r.panel}"
        N.update({f"{k}.label": r.comparison, f"{k}.tM": f"{r.threshold_margin:.4f}", f"{k}.tR": f"{r.threshold_random:.4f}",
                  f"{k}.fM": fmt(r.fnr_margin_pct, 2, False), f"{k}.fR": fmt(r.fnr_random_pct, 2, False),
                  f"{k}.yM": fmt(r.fpr_margin_pct, 1, False), f"{k}.yR": fmt(r.fpr_random_pct, 1, False),
                  f"{k}.proc": fmt(r.proc_pp), f"{k}.proc_ci": f"{fmt(r.proc_ci_low_pp)} to {fmt(r.proc_ci_high_pp)}",
                  f"{k}.proc_cci": f"{fmt(r.proc_cond_ci_low_pp)} to {fmt(r.proc_cond_ci_high_pp)}",
                  f"{k}.A": fmt(r.A_matched_pp), f"{k}.O": fmt(r.O_displacement_pp), f"{k}.res": fmt(r.residual_pp),
                  f"{k}.slope": fmt(r.slope_pp_per_pp, 1, False), f"{k}.iso": fmt(r.iso_avg_2to5_pp),
                  f"{k}.isoM": fmt(r.iso_at_fM_pp), f"{k}.isoR": fmt(r.iso_at_fR_pp),
                  f"{k}.head": fmt(r.voi_headroom_2to5_pp, 1, False), f"{k}.range": f"{r.iso_range_lo_pct:.1f}–{r.iso_range_hi_pct:.1f}%", f"{k}.vrange": f"{r.voi_range_lo_pct:.1f}–{r.voi_range_hi_pct:.1f}%",
                  f"{k}.psiR": fmt(r.psi_random, 2, False), f"{k}.psiM": fmt(r.psi_margin, 2, False),
                  f"{k}.dfnr": fmt(r.fnr_margin_pct - r.fnr_random_pct)})
        for nm, lo, hi in (("A", "A_ci_low_pp", "A_ci_high_pp"), ("O", "O_ci_low_pp", "O_ci_high_pp"), ("iso", "iso_ci_low_pp", "iso_ci_high_pp"),
                           ("psiR", "psi_random_ci_low", "psi_random_ci_high"), ("psiM", "psi_margin_ci_low", "psi_margin_ci_high")):
            l, h = getattr(r, lo, float("nan")), getattr(r, hi, float("nan"))
            ok = isinstance(l, float) and np.isfinite(l)
            sg = nm not in ("psiR", "psiM")
            N[f"{k}.{nm}_ci"] = f"{fmt(l, 2, sg)} to {fmt(h, 2, sg)}" if ok else ""
            N[f"{k}.{nm}ci"] = f" [95% CI {fmt(l, 2, sg)} to {fmt(h, 2, sg)}]" if ok else ""
    test = nom[(nom.split == "test") & nom.endpoint.isin(MAIN_EP)]
    N["pl.B"] = f"{int(nom.B.max()):,}" if "B" in nom.columns and nom.B.notna().any() else ""
    N.update(patient_level_sentences(test, pl))
    N["app.n_reversed"] = str(int((test.pattern == "reversed").sum()))
    N["app.n_eliminated"] = str(int((test.pattern == "eliminated").sum()))
    N["app.max_absO"] = fmt(test.O_displacement_pp.abs().max(), 1, False)
    N["app.max_abs_dfnr"] = fmt((test.fnr_margin_pct - test.fnr_random_pct).abs().max(), 2, False)
    mimic_psi = test[(test.site == "mimic")]
    N["app.psiM_min"], N["app.psiM_max"] = fmt(mimic_psi.psi_margin.min(), 2, False), fmt(mimic_psi.psi_margin.max(), 2, False)
    N["app.psiR_min"], N["app.psiR_max"] = fmt(mimic_psi.psi_random.min(), 2, False), fmt(mimic_psi.psi_random.max(), 2, False)
    N["app.slope_min"], N["app.slope_max"] = fmt(test.slope_pp_per_pp.min(), 1, False), fmt(test.slope_pp_per_pp.max(), 1, False)
    # ---- threshold audit
    tmr = mr[mr.split == "test"]
    N["thr.same_test"] = str(int(tmr.same_threshold.sum())); N["thr.rows_test"] = str(len(tmr))
    N["thr.zero_consistent"] = str(bool(tc.consistent.all()))
    N["thr.zero_test"] = str(int(tc[tc.split == "test"].threshold_component_zero.sum()))
    mm = ta[(ta.split == "test") & (ta.policy == "margin")]
    N["thr.margin_val_fnr_6_death"] = ", ".join(fmt(v, 1, False) for v in mm[(mm.endpoint == "death30") & np.isclose(mm.target, 0.06)].calibration_fnr_pct)
    flip = mm[(mm.endpoint == "death30") & (mm.panel == "chem7") & np.isclose(mm.target, 0.06)].iloc[0]
    N["thr.flip_fn_next"] = str(int(round(flip.next_grid_calibration_fn))); N["thr.flip_allow"] = str(int(flip.allowance_fn))
    N["thr.flip_fn_sel"] = str(int(round(flip.calibration_fn))); N["thr.flip_jump"] = fmt(flip.test_fpr_jump_to_next_grid_pp, 1, False)
    dc = tc[(tc.split == "test") & (tc.endpoint == "death30") & (tc.panel == "chem7") & np.isclose(tc.target, 0.06)].iloc[0]
    N["thr.flip_alloc"], N["thr.flip_thr"] = fmt(dc.allocation_pp), fmt(dc.threshold_pp)
    N["thr.flip_thr_ci"] = f"{fmt(dc.ci_low_threshold_pp)} to {fmt(dc.ci_high_threshold_pp)}"
    alt = tc[(tc.split == "test") & (tc.endpoint == "death30") & (tc.panel == "chem7") & np.isclose(tc.target, 0.08)].iloc[0]
    assert bool(alt.same_threshold) and abs(alt.margin - 0.006) < 1e-9
    N["thr.flip_alt"] = fmt(alt.observed_pp)
    N["thr.same_audit"] = str(int(mr[mr.split == "audit"].same_threshold.sum())); N["thr.rows_audit"] = str(int((mr.split == "audit").sum()))
    distinct = ta[(ta.split == "test") & ta.policy.isin(["margin", "random"])].groupby(["site", "endpoint", "panel", "policy"]).n_distinct_thresholds_over_6_targets.first()
    N["thr.distinct_death_min"] = str(int(distinct.xs("death30", level="endpoint").min()))
    N["thr.distinct_death_max"] = str(int(distinct.xs("death30", level="endpoint").max()))
    # ---- label flips
    N["lf.max_dev"] = f"{lf.abs_dev.max():.1e}"; N["lf.cert_agree"] = str(int((lf.certified_exact == lf.certified_closed).sum()))
    N["lf.minflip_agree"] = str(int((lf.dropna(subset=['minflips_exact']).minflips_exact == lf.dropna(subset=['minflips_exact']).minflips_closed).sum()))
    N["lf.rows"] = str(len(lf))
    ex0 = lf[(lf.panel == "cbc") & (lf.random_repeat == 0) & (lf.rho == 0) & (lf.bound_type == "common_label")].iloc[0]
    N["lf.Dw"], N["lf.Db"], N["lf.minflip"] = str(int(ex0.D_margin_worse)), str(int(ex0.D_margin_better)), str(int(ex0.minflips_closed))
    # ---- simulation
    c = cells.set_index(["alpha", "gamma"])
    for (a, gm), r in c.iterrows():
        k = "sim." + f"a{a:g}g{gm:g}".replace(".", "p")
        N.update({f"{k}.chg": fmt(r.label_change_pct, 2, False), f"{k}.B": fmt(r.B_mean_pp, 3), f"{k}.mcse": fmt(r.B_mcse_pp, 3, False),
                  f"{k}.ci": f"{fmt(r.B_ci_low_pp, 3)} to {fmt(r.B_ci_high_pp, 3)}", f"{k}.g0": fmt(r.g0_pp, 2), f"{k}.g1": fmt(r.g1_pp, 2),
                  f"{k}.naive": fmt(r.bound_naive_pp, 2, False), f"{k}.c2": fmt(r.bound_corollary2_pp, 2, False),
                  f"{k}.strata": fmt(r.bound_strata_pp, 2, False), f"{k}.rev": fmt(r.strict_reversal_pct, 0, False),
                  f"{k}.rev_mcse": fmt(r.strict_reversal_mcse_pct, 1, False), f"{k}.retro": fmt(r.retro_delta_pp, 2),
                  f"{k}.deploy": fmt(r.deploy_delta_pp, 2)})
    for nm, v in [("B", pooled["B"]), ("g0", pooled["g0"]), ("g1", pooled["g1"])]:
        N[f"sim.g1.{nm}"] = fmt(v[0], 3 if nm == "B" else 2)
        N[f"sim.g1.{nm}_ci"] = f"{fmt(v[3], 3 if nm == 'B' else 2)} to {fmt(v[4], 3 if nm == 'B' else 2)}"
        N[f"sim.g1.{nm}_mcse"] = fmt(v[2], 3, False)
    N["sim.g1.share"] = fmt(pooled["share"], 2, False); N["sim.g1.chg"] = fmt(pooled["chg"], 2, False)
    N["sim.g1.naive"], N["sim.g1.c2"], N["sim.g1.strata"] = (fmt(pooled[x], 2, False) for x in ("naive", "c2", "strata"))
    N["sim.pi1mpi"] = fmt(100 * PI * (1 - PI), 2, False)
    N["sim.prev"] = fmt(100 * s.d1_prevalence.mean(), 1, False)
    N["sim.check"] = fmt(PI * (1 - PI) * (pooled["g0"][0] - pooled["g1"][0]), 3)
    N["sim.max_identity_resid"] = f"{cells.max_abs_identity_resid.max():.1e}"
    N["sim.bounds_hold"] = str(bool(cells.bound_always_holds.all()))
    N["sim.neither_draws"] = f"{band['neither_draws']/1e6:.2f}"; N["sim.neither_disagree"] = str(int(round(band["neither_disagree"])))
    pg = pd.read_csv(RESULTS / "06_simulation/paired_gamma_difference.csv").iloc[0]
    N["sim.dgamma"] = fmt(pg.paired_mean_pp, 3); N["sim.dgamma_ci"] = f"{fmt(pg.ci95_low_pp, 3)} to {fmt(pg.ci95_high_pp, 3)}"
    # ---- cohorts, endpoints and discrimination
    cal = pd.read_csv(RESULTS / "04_calibration/ALL_prediction_calibration_and_discrimination.csv")
    raw = pd.concat([cal[(cal.stage == "v4.4_raw") & (cal.split == "test")],
                     cal[(cal.stage == "v4.6_validation_fit") & (cal.split == "test") & (cal.calibration == "raw")]])
    disc = []
    for r in raw.itertuples():
        ep = "death30" if "mortality" in r.endpoint else "critical"
        disc.append(dict(site=r.site, endpoint=ep, panel=r.panel, feature_set=r.feature_set, n=r.n, n_positive=r.n_positive,
                         auroc=r.auroc, average_precision=r.average_precision, brier=r.brier, mean_predicted=r.mean_predicted, prevalence=r.prevalence))
    disc = pd.DataFrame(disc)
    disc.to_csv(DERIVED / "discrimination_no_panel_vs_full_panel_test.csv", index=False)
    for (site, ep, panel), g in disc.groupby(["site", "endpoint", "panel"]):
        f0, f1 = g[g.feature_set == "free_only"].iloc[0], g[g.feature_set == "panel_all"].iloc[0]
        k = f"disc.{KEY[(site, ep)]}_{panel}"
        N.update({f"{k}.auc0": f"{f0.auroc:.3f}", f"{k}.auc1": f"{f1.auroc:.3f}", f"{k}.ap0": f"{f0.average_precision:.3f}",
                  f"{k}.ap1": f"{f1.average_precision:.3f}", f"{k}.mp1": fmt(100 * f1.mean_predicted, 1, False), f"{k}.prev": fmt(100 * f1.prevalence, 2, False)})
    mc = pd.read_csv(RESULTS / "01_primary/ALL_main_policy_contrasts.csv")
    e1 = mc[mc.endpoint == "E1_ED_disposition"]
    for r in e1.itertuples():
        N[f"e1.{r.site}_{r.panel}_{r.metric}"] = fmt(r.delta_pp)
        N[f"e1.{r.site}_{r.panel}_{r.metric}_M"] = fmt(100 * r.margin_rate, 1, False)
    N["e1.max_abs_dfnr"] = fmt(e1[e1.metric == "fnr"].delta_pp.abs().max(), 2, False)
    (DERIVED / "numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False), encoding="utf-8")
    return dict(nom=nom, ta=ta, mr=mr, tc=tc, lf=lf, cells=cells, N=N, pooled=pooled, band=band)


if __name__ == "__main__":
    run()
