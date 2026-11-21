"""Synthetic-only fixed-predictor evaluation-gap attribution.

Decomposes deployment-minus-retrospective policy contrast by the pair of
binary *test allocation* decisions. Output is a finite-sample accounting
identity, not a causal estimator or FNR/FPR safety certificate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

GROUPS = ("both", "neither", "margin_only", "random_only")


def _binary(x, name, n=None):
    x = np.asarray(x)
    if x.ndim != 1 or (n is not None and x.size != n):
        raise ValueError(f"{name}: expected 1-D vector of length {n}")
    if not np.all(np.isin(x, [0, 1, False, True])):
        raise ValueError(f"{name}: must be binary (0 or 1)")
    return x.astype(np.int8, copy=False)


def decompose_by_assignment(d0, d1, pred_margin, pred_randoms,
                            alloc_margin, alloc_randoms):
    """Exact pooled average over fixed Random seeds, normalised by all patients.

    c_i = (D1-D0) * [(1-A_M)(2h_M-1) - (1-A_R)(2h_R-1)]

    The four assignment groups partition every patient × Random seed pair.
    The net contribution of each group uses the *whole-cohort denominator*,
    so their sum equals the overall deployment-minus-retrospective gap.
    """
    d0 = _binary(d0, "D0")
    n = d0.size
    if n == 0:
        raise ValueError("Cannot decompose an empty test set")
    d1 = _binary(d1, "D1", n)
    hm = _binary(pred_margin, "pred_margin", n)
    am = _binary(alloc_margin, "alloc_margin", n).astype(bool)
    if len(pred_randoms) != len(alloc_randoms) or len(pred_randoms) < 1:
        raise ValueError("Need matching nonempty Random prediction/allocation repetitions")
    delta = (d1.astype(np.int16) - d0.astype(np.int16))
    change = (delta != 0)
    m_error_change = (~am) * delta * (2*hm.astype(np.int16)-1)
    acc = {g: {"n": 0,"n_changed": 0,"n_pred_disagree": 0,
               "n_changed_disagree": 0, "signed": 0.0,
               "positive": 0.0,"negative": 0.0,"absolute": 0.0}
           for g in GROUPS}
    total = 0.0
    for j, (h_r_raw, a_r_raw) in enumerate(zip(pred_randoms, alloc_randoms)):
        hr = _binary(h_r_raw, f"pred_random[{j}]", n)
        ar = _binary(a_r_raw, f"alloc_random[{j}]", n).astype(bool)
        c = m_error_change - (~ar) * delta * (2*hr.astype(np.int16)-1)
        disagree = hm != hr
        masks = {
            "both": am & ar,
            "neither": (~am) & (~ar),
            "margin_only": am & (~ar),
            "random_only": (~am) & ar,
        }
        if not np.all(sum(g.astype(np.int8) for g in masks.values()) == 1):
            raise AssertionError("Assignment group partition is incomplete")
        for g, mask in masks.items():
            sub = c[mask].astype(float)
            a = acc[g]
            a["n"] += int(mask.sum())
            a["n_changed"] += int(np.count_nonzero(mask & change))
            a["n_pred_disagree"] += int(np.count_nonzero(mask & disagree))
            a["n_changed_disagree"] += int(np.count_nonzero(mask & disagree & change))
            a["signed"] += float(sub.sum())
            a["positive"] += float(np.maximum(sub, 0).sum())
            a["negative"] += float(np.minimum(sub, 0).sum())
            a["absolute"] += float(np.abs(sub).sum())
        total += float(c.sum())
    denom = n * len(pred_randoms)
    results = {}
    for g in GROUPS:
        a = acc[g]
        k = a["n"]
        results[g] = {
            "fraction": k/denom,
            "label_change_rate_within": a["n_changed"]/k if k else 0.0,
            "prediction_disagreement_rate_within": a["n_pred_disagree"]/k if k else 0.0,
            "changed_and_prediction_disagree_fraction": a["n_changed_disagree"]/denom,
            "net_gap": a["signed"]/denom,
            "positive_gap": a["positive"]/denom,
            "negative_gap": a["negative"]/denom,
            "absolute_gap": a["absolute"]/denom,
        }
        if not np.isclose(results[g]["positive_gap"]+results[g]["negative_gap"],
                          results[g]["net_gap"], atol=1e-15):
            raise AssertionError(f"Signed components do not add for group {g}")
    gap = total / denom
    if not np.isclose(sum(results[g]["net_gap"] for g in GROUPS), gap, atol=1e-15):
        raise AssertionError("Group attribution does not sum to overall policy gap")
    if abs(results["both"]["net_gap"]) > 1e-15:
        raise AssertionError("Both-tested group must have zero label correction")
    return {"gap": gap, "groups": results}


def group_summary(input_csv: Path, output_csv: Path):
    """Aggregate synthetic replicate-level group results; no patient rows."""
    df = pd.read_csv(input_csv)
    fields = ("fraction", "label_change_rate_within",
              "prediction_disagreement_rate_within",
              "changed_and_prediction_disagree_fraction",
              "net_gap", "positive_gap", "negative_gap", "absolute_gap")
    required = ["alpha", "gamma", "rep", "retrospective_to_deployment_gap"] + [
        f"group_{g}_{f}" for g in GROUPS for f in fields]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing new per-replicate assignment attribution columns: {missing}")
    if df.duplicated(["alpha", "gamma", "rep"]).any():
        raise ValueError("Duplicate alpha/gamma/rep records")
    if not np.allclose(df[[f"group_{g}_net_gap" for g in GROUPS]].sum(axis=1),
                       df["retrospective_to_deployment_gap"], rtol=0, atol=1e-12):
        raise ValueError("Assignment group contribution identity FAILED")
    rows=[]
    for (alpha,gamma), subset in df.groupby(["alpha", "gamma"], sort=True):
        for g in GROUPS:
            n=len(subset)
            row={"alpha":alpha,"gamma":gamma,"group":g,"n_independent_reps":n}
            for f in fields:
                x=subset[f"group_{g}_{f}"].to_numpy(float)
                mu=float(x.mean())
                se=float(np.std(x, ddof=1)/np.sqrt(n)) if n>1 else np.nan
                crit=float(student_t.ppf(.975,n-1)) if n>1 else np.nan
                row[f+"_mean"]=mu
                row[f+"_ci95_low"]=mu-crit*se if n>1 else np.nan
                row[f+"_ci95_high"]=mu+crit*se if n>1 else np.nan
            rows.append(row)
    out=pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        raise FileExistsError(f"Refusing to overwrite {output_csv}; choose a new filename")
    out.to_csv(output_csv,index=False)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("summarize", nargs="?")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    a=p.parse_args()
    if a.summarize not in (None, "summarize"):
        p.error("Supported command: summarize")
    res=group_summary(Path(a.input),Path(a.output))
    print(f"Saved {len(res)} synthetic-only group-level summary rows to {a.output}")

if __name__=="__main__":
    main()
