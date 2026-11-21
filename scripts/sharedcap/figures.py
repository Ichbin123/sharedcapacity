"""Main and supplementary figures (aggregate data only). Colour-blind-safe Okabe-Ito palette."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .paths import RESULTS, DERIVED, FIG
from . import derived as D

COL = {"margin": "#0072B2", "random": "#D55E00", "no_panel": "#7F7F7F", "full_panel": "#009E73"}
MK = {"margin": "o", "random": "s", "no_panel": "v", "full_panel": "^"}
NAME = {"margin": "Margin", "random": "Random", "no_panel": "No panel (coverage 0)", "full_panel": "Full panel (coverage 1)"}
G05, G1 = "#56B4E9", "#0072B2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.1,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.titlelocation": "left", "axes.titleweight": "bold"})


def save(fig, name, main=True):
    sub = FIG / ("main" if main else "supplementary")
    fig.savefig(sub / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(sub / f"{name}.png", dpi=300, bbox_inches="tight")
    if main:
        fig.savefig(sub / f"{name}.tif", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        fig.savefig(sub / f"{name}.eps", bbox_inches="tight")
    plt.close(fig)


def tag(ax, s, x=-0.16, y=1.06):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def grid(ax, axis="both"):
    ax.grid(axis=axis, color="#E6E6E6", lw=0.5)
    ax.set_axisbelow(True)


def pts_of(d, split, site, ep, panel, pol):
    return d[(d.split == split) & (d.site == site) & (d.endpoint == ep) & (d.panel == panel) & (d.policy == pol)]


def minus(s):
    return s.replace("-", "−")


def fig1(d, curves, nom):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05), constrained_layout=True)
    spec = [(("mcmed", "critical", "cbc"), (3.0, 8.0)), (("mimic", "death30", "chem7"), (1.0, 4.0))]
    for ax, ((site, ep, panel), xl), t in zip(axes, spec, "ab"):
        r = nom[(nom.split == "test") & (nom.site == site) & (nom.endpoint == ep) & (nom.panel == panel)].iloc[0]
        ys = []
        for pol in ("random", "margin"):
            cv = curves[("test", site, ep, panel, pol)]
            p = pts_of(d, "test", site, ep, panel, pol)
            ax.plot(100 * cv[:, 0], 100 * cv[:, 1], color=COL[pol], lw=1.3)
            ax.plot(100 * p.fnr, 100 * p.fpr, MK[pol], color=COL[pol], ms=3.4, mfc="white", mew=0.9)
            xx = np.linspace(xl[0], xl[1], 60) / 100
            ys += [100 * D.H(cv, x) for x in xx if np.isfinite(D.H(cv, x))]
        fM, fR, yM, yR = r.fnr_margin_pct, r.fnr_random_pct, r.fpr_margin_pct, r.fpr_random_pct
        cM, cR = curves[("test", site, ep, panel, "margin")], curves[("test", site, ep, panel, "random")]
        xs = np.linspace(min(fM, fR), max(fM, fR), 60)
        ax.plot(xs, [100 * D.H(cR, x / 100) for x in xs], color="black", lw=2.6, ls=(0, (1.2, 1.0)), zorder=4)
        ax.annotate("", xy=(fM, 100 * D.H(cM, fM / 100)), xytext=(fM, 100 * D.H(cR, fM / 100)),
                    arrowprops=dict(arrowstyle="<->", color="0.25", lw=0.9, shrinkA=0, shrinkB=0), zorder=6)
        ax.plot(fM, yM, "o", ms=7, color=COL["margin"], mec="black", mew=0.8, zorder=7)
        ax.plot(fR, yR, "s", ms=6.5, color=COL["random"], mec="black", mew=0.8, zorder=7)
        txt = minus(f"Procedure contrast  {r.proc_pp:+.2f}\nA (matched sensitivity)  {r.A_matched_pp:+.2f}\n"
                    f"O (displacement)  {r.O_displacement_pp:+.2f}\nHull residual  {r.residual_pp:+.2f}")
        ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=6.8, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", lw=0.5))
        ax.set_xlim(*xl)
        lo, hi = min(ys + [yM, yR]), max(ys + [yM, yR])
        ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.10 * (hi - lo))
        ax.set_xlabel("Realised FNR on test (%)")
        ax.set_ylabel("Realised FPR on test (%)")
        ax.set_title(f"{D.label(site, ep, panel)}  (values in pp)")
        grid(ax)
        tag(ax, t)
    h = [Line2D([], [], color=COL["margin"], marker="o", mfc="white", label="Margin: operating points and lower hull"),
         Line2D([], [], color=COL["random"], marker="s", mfc="white", label="Random: operating points and lower hull"),
         Line2D([], [], color="black", marker="o", ls="", mfc="0.6", mec="black", ms=6, label="Nominal 6% operating points"),
         Line2D([], [], color="black", lw=2.4, ls=(0, (1.2, 1.0)), label="Displacement along Random hull"),
         Line2D([], [], color="0.25", lw=0.9, marker=r"$\updownarrow$", ms=8, ls="", label="Matched-sensitivity gap at Margin FNR")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13), frameon=False)
    save(fig, "Fig1_frontier_decomposition")


def frontier_panel(ax, d, curves, split, site, ep, panel, xl):
    for pol in ("no_panel", "random", "margin", "full_panel"):
        cv = curves[(split, site, ep, panel, pol)]
        p = pts_of(d, split, site, ep, panel, pol)
        ax.plot(100 * cv[:, 0], 100 * cv[:, 1], color=COL[pol], lw=1.05)
        ax.plot(100 * p.fnr, 100 * p.fpr, MK[pol], color=COL[pol], ms=2.8, mfc="white", mew=0.8)
    for pol in ("margin", "random"):
        p = pts_of(d, split, site, ep, panel, pol)
        p = p[np.isclose(p.target, 0.06)]
        ax.plot(100 * p.fnr, 100 * p.fpr, MK[pol], ms=5.6, color=COL[pol], mec="black", mew=0.7, zorder=5)
    ax.set_xlim(*xl)
    grid(ax)


def legend4(fig, y=-0.07):
    h = [Line2D([], [], color=COL[p], marker=MK[p], mfc="white", label=NAME[p]) for p in ("no_panel", "random", "margin", "full_panel")]
    h.append(Line2D([], [], color="black", marker="o", ls="", mfc="0.6", mec="black", ms=5.5, label="Nominal 6% target (Margin, Random)"))
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, y), frameon=False, columnspacing=1.2, handletextpad=0.4)


def fig2(d, curves):
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.7), constrained_layout=True)
    cols = [("mimic", "critical", (0, 11)), ("mcmed", "critical", (0, 15.5)), ("mimic", "death30", (0, 7))]
    for j, (site, ep, xl) in enumerate(cols):
        for i, panel in enumerate(("cbc", "chem7")):
            ax = axes[i, j]
            frontier_panel(ax, d, curves, "test", site, ep, panel, xl)
            ax.set_title(D.label(site, ep, panel), fontsize=7.6)
            if i == 1:
                ax.set_xlabel("Realised FNR on test (%)")
            if j == 0:
                ax.set_ylabel("Realised FPR on test (%)")
            tag(ax, "abcdef"[3 * i + j], x=-0.2, y=1.04)
    legend4(fig, -0.075)
    save(fig, "Fig3_frontiers_test")


def fig3(nom):
    t = nom[(nom.split == "test") & nom.endpoint.isin(D.MAIN_EP)].reset_index(drop=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw=dict(width_ratios=[1.3, 1]), constrained_layout=True)
    y = np.arange(len(t))[::-1].astype(float)
    a.axvline(0, color="0.45", lw=0.7)
    has = "A_ci_low_pp" in t.columns and t["A_ci_low_pp"].notna().any()
    nested = "ci_type" in t.columns and t["ci_type"].astype(str).str.startswith("nested").any()
    a.errorbar(t.proc_pp, y + 0.24, xerr=[t.proc_pp - t.proc_ci_low_pp, t.proc_ci_high_pp - t.proc_pp], fmt="D", color="black", ms=3.8, capsize=2, lw=0.8,
               label="Procedure contrast (joint-resampling 95% CI)" if nested else "Procedure contrast (95% CI, thresholds held fixed)")
    if has:
        a.errorbar(t.A_matched_pp, y, xerr=[t.A_matched_pp - t.A_ci_low_pp, t.A_ci_high_pp - t.A_matched_pp], fmt="o", color=COL["margin"], ms=4.6,
                   capsize=2, lw=0.8, label="A: matched-sensitivity component (95% CI)")
        a.errorbar(t.O_displacement_pp, y - 0.24, xerr=[t.O_displacement_pp - t.O_ci_low_pp, t.O_ci_high_pp - t.O_displacement_pp], fmt="^",
                   color=COL["random"], ms=4.6, capsize=2, lw=0.8, label="O: displacement component (95% CI)")
    else:
        a.plot(t.A_matched_pp, y, "o", color=COL["margin"], ms=4.6, label="A: matched-sensitivity component")
        a.plot(t.O_displacement_pp, y - 0.24, "^", color=COL["random"], ms=4.6, label="O: displacement component")
    for yi, pat in zip(y, t.pattern):
        a.text(1.01, yi, pat, transform=a.get_yaxis_transform(), fontsize=6.4, va="center", ha="left", color="0.3", style="italic")
    a.set_yticks(y, t.comparison)
    cols = ["proc_ci_low_pp", "proc_ci_high_pp", "A_matched_pp", "O_displacement_pp"] + (["A_ci_low_pp", "A_ci_high_pp", "O_ci_low_pp", "O_ci_high_pp"] if has else [])
    lo_, hi_ = float(np.nanmin(t[cols].to_numpy(float))), float(np.nanmax(t[cols].to_numpy(float))); pad = 0.06 * (hi_ - lo_)
    a.set_xlim(lo_ - pad, hi_ + pad)
    a.set_xlabel("FPR difference, Margin − Random (pp)")
    grid(a, "x")
    a.legend(loc="upper center", bbox_to_anchor=(0.42, -0.17), ncol=1, frameon=False)
    tag(a, "a", x=-0.62)
    v = nom[(nom.site == "mimic") & nom.endpoint.isin(D.MAIN_EP)].reset_index(drop=True)
    labels = [c.replace("MIMIC ", "") + (" [audit]" if s == "audit" else "") for c, s in zip(v.comparison, v.split)]
    yy = np.arange(len(v))[::-1].astype(float)
    hasp = "psi_margin_ci_low" in v.columns and v["psi_margin_ci_low"].notna().any()
    kw = lambda est, lo, hi: dict(xerr=[v[est] - v[lo], v[hi] - v[est]], capsize=2, error_kw=dict(lw=0.7)) if hasp else {}
    b.barh(yy + 0.19, v.psi_random, height=0.36, color=COL["random"], label="Random", **kw("psi_random", "psi_random_ci_low", "psi_random_ci_high"))
    b.barh(yy - 0.19, v.psi_margin, height=0.36, color=COL["margin"], label="Margin", **kw("psi_margin", "psi_margin_ci_low", "psi_margin_ci_high"))
    b.axvline(D.PI, color="0.2", ls="--", lw=0.8)
    b.text(D.PI + 0.02, yy.max() + 0.62, "coverage π = 5/16", fontsize=6.5, color="0.2")
    b.axvline(0, color="0.45", lw=0.7)
    b.set_yticks(yy, labels)
    pv = v[["psi_random", "psi_margin"] + (["psi_random_ci_low", "psi_random_ci_high", "psi_margin_ci_low", "psi_margin_ci_high"] if hasp else [])].to_numpy(float)
    b.set_xlim(min(-0.2, float(np.nanmin(pv)) - 0.05), max(1.0, float(np.nanmax(pv)) + 0.05))
    b.set_xlabel("ψ: share of no-panel → full-panel FPR\nreduction captured (FNR 2–5%)")
    grid(b, "x")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, frameon=False)
    tag(b, "b", x=-0.5)
    save(fig, "Fig4_procedure_vs_matched")


def fig4(s):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.0, 2.55), constrained_layout=True)
    alphas = np.array([0.0, 0.5, 1.0])
    for gm, off, col, mk in ((0.5, -0.035, G05, "o"), (1.0, 0.035, G1, "s")):
        m, lo, hi = [], [], []
        for al in alphas:
            x = 100 * s[(s.alpha == al) & (s.gamma == gm)].retrospective_to_deployment_gap
            mm, _, _, l, h = D.tci(x)
            m.append(mm); lo.append(mm - l); hi.append(h - mm)
        a.errorbar(alphas + off, m, yerr=[lo, hi], fmt=mk + "-", color=col, ms=3.8, capsize=2, lw=1, label=f"γ = {gm:g}")
    a.axhline(0, color="0.45", lw=0.7)
    a.set_xticks(alphas); a.set_xlabel("Prognostic information α"); a.set_ylabel("Evaluation gap B (pp)")
    a.legend(frameon=False, loc="upper right"); grid(a, "y"); tag(a, "a", x=-0.3)
    g1 = s[s.gamma == 1.0]
    for k, (var, col, lab, off) in enumerate((("g0_pp", "#009E73", r"$\bar g_0$: Margin unallocated", -0.09),
                                             ("g1_pp", "#CC79A7", r"$\bar g_1$: Margin allocated", 0.09))):
        m, e = [], []
        for al in alphas:
            mm, _, _, l, h = D.tci(g1[g1.alpha == al][var])
            m.append(mm); e.append(h - mm)
        b.bar(alphas + off, m, width=0.17, color=col, yerr=e, capsize=2, error_kw=dict(lw=0.7), label=lab)
    b.set_xticks(alphas); b.set_xlabel("Prognostic information α (γ = 1)"); b.set_ylabel("Mean signed label change (pp)")
    b.legend(frameon=False, loc="upper right"); grid(b, "y"); tag(b, "b", x=-0.3)
    for gm, off, col, mk in ((0.5, -0.035, G05, "o"), (1.0, 0.035, G1, "s")):
        p = np.array([s[(s.alpha == al) & (s.gamma == gm)].policy_ranking_strict_reversal.mean() for al in alphas])
        se = np.sqrt(p * (1 - p) / 100)
        lo_e = np.minimum(196 * se, 100 * p)
        c.errorbar(alphas + off, 100 * p, yerr=[lo_e, 196 * se], fmt=mk + "-", color=col, ms=3.8, capsize=2, lw=1, label=f"γ = {gm:g}")
    c.set_xticks(alphas); c.set_ylim(0, 40); c.set_xlabel("Prognostic information α"); c.set_ylabel("Strict ranking reversals (%)")
    c.legend(frameon=False, loc="upper right"); grid(c, "y"); tag(c, "c", x=-0.3)
    save(fig, "Fig2_simulation")


# ------------------------------------------------------------------ supplementary
def figS1():
    m = pd.read_csv(RESULTS / "01_primary/ALL_main_policy_contrasts.csv")
    order = [("E1_ED_disposition", "mimic"), ("E1_ED_disposition", "mcmed"), ("E2_site_specific_critical", "mimic"),
             ("E2_site_specific_critical", "mcmed"), ("mortality_calendar_day_0_to_30", "mimic")]
    epn = {"E1_ED_disposition": "ED disposition", "E2_site_specific_critical": "critical", "mortality_calendar_day_0_to_30": "death (0–30 d)"}
    rows = [(ep, site, panel) for ep, site in order for panel in ("cbc", "chem7")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True, constrained_layout=True)
    y = np.arange(len(rows))[::-1].astype(float)
    for ax, metric, ttl in ((axes[0], "fpr", "False-positive rate"), (axes[1], "fnr", "False-negative rate")):
        for yi, (ep, site, panel) in zip(y, rows):
            r = m[(m.endpoint == ep) & (m.site == site) & (m.panel == panel) & (m.metric == metric)].iloc[0]
            col = COL["margin"] if panel == "cbc" else COL["random"]
            ax.plot([r.ci_low_pp, r.ci_high_pp], [yi, yi], color=col, lw=1.6)
            ax.plot(r.delta_pp, yi, "o", color=col, ms=4)
        for yb in (y[1] - 0.5, y[3] - 0.5, y[5] - 0.5, y[7] - 0.5):
            ax.axhline(yb, color="0.85", lw=0.6)
        ax.axvline(0, color="0.45", lw=0.7, ls="--")
        ax.set_title(ttl); ax.set_xlabel("Margin − Random (pp)"); grid(ax, "x")
    axes[0].set_yticks(y, [f"{'MIMIC' if s == 'mimic' else 'MC-MED'} {epn[e]}, {PANEL(p)}" for e, s, p in rows])
    h = [Line2D([], [], color=COL["margin"], marker="o", label="CBC"), Line2D([], [], color=COL["random"], marker="o", label="chem7")]
    fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07), frameon=False)
    save(fig, "FigS1_procedure_contrasts_all", main=False)


def PANEL(p):
    return D.PANEL_LABEL[p]


def figS2(tc):
    t = tc[tc.split == "test"]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), constrained_layout=True, sharex=True)
    cols = [("mimic", "critical"), ("mcmed", "critical"), ("mimic", "death30")]
    dec = pd.read_csv(RESULTS / "03_decomposition/ALL_5_experiments_6_targets_summary.csv")
    amap = {("mimic", "critical"): "critical_mimic_test", ("mcmed", "critical"): "critical_mcmed_test", ("mimic", "death30"): "mortality_test"}
    for j, (site, ep) in enumerate(cols):
        for i, panel in enumerate(("cbc", "chem7")):
            ax = axes[i, j]
            q = dec[(dec.analysis == amap[(site, ep)]) & (dec.panel == panel) & (dec.metric == "fpr")].sort_values("nominal_validation_fnr_target")
            x = 100 * q.nominal_validation_fnr_target.to_numpy()
            ax.bar(x - 0.42, q.allocation_pp, width=0.8, color=COL["margin"],
                   yerr=[q.allocation_pp - q.ci_low_allocation_pp, q.ci_high_allocation_pp - q.allocation_pp], capsize=1.5, error_kw=dict(lw=0.6))
            ax.bar(x + 0.42, q.threshold_pp, width=0.8, color=COL["random"],
                   yerr=[q.threshold_pp - q.ci_low_threshold_pp, q.ci_high_threshold_pp - q.threshold_pp], capsize=1.5, error_kw=dict(lw=0.6))
            ax.plot(x, q.observed_pp, "D", color="black", ms=3.2)
            same = t[(t.site == site) & (t.endpoint == ep) & (t.panel == panel)].set_index("target").same_threshold
            for xv in x:
                if bool(same.get(round(xv / 100, 4), False)):
                    ax.text(xv + 0.42, 0, "=", ha="center", va="bottom", fontsize=8, color=COL["random"], fontweight="bold")
            ax.axhline(0, color="0.45", lw=0.6)
            ax.set_title(D.label(site, ep, panel), fontsize=7.6); grid(ax, "y")
            ax.set_xticks([2, 4, 6, 8, 10, 12])
            if i == 1: ax.set_xlabel("Nominal calibration FNR target (%)")
            if j == 0: ax.set_ylabel("FPR attribution (pp)")
    h = [Patch(color=COL["margin"], label="Allocation (threshold-space Shapley)"), Patch(color=COL["random"], label="Threshold (threshold-space Shapley)"),
         Line2D([], [], color="black", marker="D", ls="", label="Procedure contrast"),
         Line2D([], [], color=COL["random"], marker="$=$", ls="", ms=7, label="t̂_M = t̂_R (component exactly 0)")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), frameon=False)
    save(fig, "FigS2_threshold_space_decomposition", main=False)


def figS3(ta):
    t = ta[(ta.split == "test") & ta.policy.isin(["margin", "random"])]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0), constrained_layout=True, sharex=True)
    for j, (site, ep) in enumerate([("mimic", "critical"), ("mcmed", "critical"), ("mimic", "death30")]):
        for i, panel in enumerate(("cbc", "chem7")):
            ax = axes[i, j]
            q = t[(t.site == site) & (t.endpoint == ep) & (t.panel == panel)]
            for pol, off in (("margin", -0.18), ("random", 0.18)):
                z = q[q.policy == pol].sort_values("target")
                ax.plot(100 * z.target + off, z.grid_index, MK[pol] + "-", color=COL[pol], ms=3.6, lw=0.8, label=NAME[pol])
            zm, zr = q[q.policy == "margin"].sort_values("target"), q[q.policy == "random"].sort_values("target")
            same = np.isclose(zm.threshold.values, zr.threshold.values)
            ax.plot(100 * zm.target.values[same], zm.grid_index.values[same], "o", ms=9, mfc="none", mec="black", mew=0.8)
            ax.set_title(D.label(site, ep, panel), fontsize=7.6); grid(ax)
            ax.set_xticks([2, 4, 6, 8, 10, 12])
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            if i == 1: ax.set_xlabel("Nominal calibration FNR target (%)")
            if j == 0: ax.set_ylabel("Selected grid index k\n(threshold = 0.001 + 0.0025k)")
    h = [Line2D([], [], color=COL[p], marker=MK[p], label=NAME[p]) for p in ("margin", "random")]
    h.append(Line2D([], [], marker="o", ms=9, mfc="none", mec="black", ls="", label="identical thresholds"))
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07), frameon=False)
    save(fig, "FigS3_threshold_grid_selection", main=False)


def figS4(lf):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    for ax, panel in zip(axes, ("cbc", "chem7")):
        q = lf[lf.panel == panel]
        for bt, ls, col in (("common_label", "-", COL["margin"]), ("policy_specific", "--", COL["random"])):
            g = q[q.bound_type == bt].groupby("rho")[["lower_exact", "upper_exact", "lower_closed", "upper_closed", "delta"]].mean().reset_index()
            x = 100 * g.rho
            ax.plot(x, 100 * g.lower_closed, ls, color=col, lw=1.1)
            ax.plot(x, 100 * g.upper_closed, ls, color=col, lw=1.1)
            ax.plot(x, 100 * g.lower_exact, "o", color=col, ms=3.4, mfc="white")
            ax.plot(x, 100 * g.upper_exact, "o", color=col, ms=3.4, mfc="white")
        dl = 100 * q.delta.mean()
        ax.axhline(0, color="0.45", lw=0.7, ls=":")
        ax.axvline(abs(dl) / 2, color="0.3", lw=0.7, ls="-.")
        ax.text(abs(dl) / 2 + 0.15, ax.get_ylim()[0] * 0.85 if dl < 0 else ax.get_ylim()[1] * 0.85, "|Δ|/2", fontsize=6.5)
        ax.set_title(f"MIMIC death audit, {PANEL(panel)}: observed Δ = {dl:+.2f} pp".replace("-", "−"))
        ax.set_xlabel("Label-flip budget ρ (% of encounters)"); ax.set_ylabel("0–1 error difference (pp)"); grid(ax)
    h = [Line2D([], [], color=COL["margin"], ls="-", label="Common label: closed form"), Line2D([], [], color=COL["random"], ls="--", label="Policy-specific: closed form"),
         Line2D([], [], color="0.3", marker="o", mfc="white", ls="", label="Previously reported exact bounds (mean of 10 Random draws)")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), frameon=False)
    save(fig, "FigS5_label_flip_closed_form", main=False)


def figS5():
    g = pd.read_csv(RESULTS / "06_simulation/MAIN_gamma1_assignment_strata.csv")
    names = {"both": "Both allocated (≡ 0)", "neither": "Neither allocated", "margin_only": "Margin only", "random_only": "Random only"}
    fig, ax = plt.subplots(figsize=(5.2, 2.3), constrained_layout=True)
    y = np.arange(len(g))[::-1]
    ax.barh(y, g.positive_pp, color="#009E73", height=0.55, label="positive contributions")
    ax.barh(y, g.negative_pp, color="#CC79A7", height=0.55, label="negative contributions")
    ax.plot(g.net_pp, y, "D", color="black", ms=4, label="net")
    ax.axvline(0, color="0.45", lw=0.7)
    ax.set_yticks(y, [names[k] for k in g.group]); ax.set_xlabel("Contribution to B at γ = 1, averaged over α (pp)"); grid(ax, "x")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.5)
    save(fig, "FigS6_signed_contributions", main=False)


def figS6(d, curves):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    for ax, panel, t in zip(axes, ("cbc", "chem7"), "ab"):
        frontier_panel(ax, d, curves, "audit", "mimic", "death30", panel, (0, 12))
        ax.set_title(f"MIMIC death (0–30 d), {PANEL(panel)} — audit split")
        ax.set_xlabel("Realised FNR (%)"); ax.set_ylabel("Realised FPR (%)"); tag(ax, t, x=-0.14)
    legend4(fig, -0.13)
    save(fig, "FigS4_frontiers_audit", main=False)


def run():
    d, curves = D.frontier_inputs()
    nom = pd.read_csv(DERIVED / "frontier_decomposition_nominal6.csv")
    ta = pd.read_csv(DERIVED / "threshold_selection_audit.csv")
    tc = pd.read_csv(DERIVED / "threshold_space_component_zero_check.csv")
    lf = pd.read_csv(DERIVED / "label_flip_closed_form_check.csv")
    s, _, _, _ = D.simulation_tables()
    fig1(d, curves, nom); fig2(d, curves); fig3(nom); fig4(s)
    figS1(); figS2(tc); figS3(ta); figS4(lf); figS5(); figS6(d, curves)


if __name__ == "__main__":
    run()
