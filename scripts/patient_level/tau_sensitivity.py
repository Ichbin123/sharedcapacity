#!/usr/bin/env python3
"""Remark 1 sensitivity analysis and band-condition check from the patient-level exports.

RUN ONLY INSIDE THE CREDENTIALED ENVIRONMENT. Reads the same exports as run_patient_level.py
(<site>_<endpoint>_<panel>.csv with columns split, pool, p0, A_M, ...) and the grid-rule thresholds in
results/11_patient_level/patient_level_decomposition.csv. Writes only aggregate rates to
results/11_patient_level/patient_level_tau_sensitivity.csv, which derived.py turns into the Section 5.2
sentence and Supporting Information Section S11.

  python scripts/patient_level/tau_sensitivity.py --input-dir ~/secure_sharedcap/exports

With a constant effect tau0 of panel availability on the label (Remark 1), Theorem 1 gives for cost-weighted
loss, complete pools and uniform random allocation of K of m (pi = K/m)
    B(tau0) = tau0 * (1 - pi) * {u0M - (1 - pi) * u0 - pi * u1},
with u_a = Pr(h0_R = 1 | A_M = a) and u0M = Pr(h0_M = 1 | A_M = 0). When the band condition of Corollary 1 holds
(h0_M = h0_R for every encounter that Margin leaves unallocated), u0M = u0 and this is equation (9),
tau0 * pi * (1 - pi) * (u0 - u1). For 0-1 loss the values double.
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

NAME = re.compile(r"^(mimic|mcmed)_(critical|death30|disposition)_(cbc|chem7)\.csv$")
ORDER = ["mimic_critical_cbc", "mimic_critical_chem7", "mcmed_critical_cbc", "mcmed_critical_chem7", "mimic_death30_cbc",
         "mimic_death30_chem7", "mimic_disposition_cbc", "mimic_disposition_chem7", "mcmed_disposition_cbc", "mcmed_disposition_chem7"]
TAUS = (0.01, 0.02, 0.05)


def rates(p0, AM, tM, tR, pi):
    hR, hM = p0 >= tR, p0 >= tM
    u0, u1, u0M = float(hR[~AM].mean()), float(hR[AM].mean()), float(hM[~AM].mean())
    row = dict(u0=u0, u1=u1, u0_minus_u1=u0 - u1, u0_margin=u0M, band_violation_pct=100 * float((hM != hR)[~AM].mean()))
    for t in TAUS:
        row[f"B_tau_{t:g}_pp"] = 100 * t * (1 - pi) * (u0M - (1 - pi) * u0 - pi * u1)   # Theorem 1, constant tau
        row[f"B_tau_{t:g}_eq9_pp"] = 100 * t * pi * (1 - pi) * (u0 - u1)                 # equation (9)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--pl-dir", default="results/11_patient_level")
    ap.add_argument("--m", type=int, default=16); ap.add_argument("--K", type=int, default=5)
    a = ap.parse_args()
    pi = a.K / a.m
    pl = Path(a.pl_dir)
    dec = pd.read_csv(pl / "patient_level_decomposition.csv")
    dec = dec[dec.rule == "grid"]
    files = sorted((f for f in Path(a.input_dir).expanduser().glob("*.csv") if NAME.match(f.name)),
                   key=lambda f: ORDER.index(f.stem) if f.stem in ORDER else 99)
    if not files:
        sys.exit(f"no export files matching <site>_<endpoint>_<panel>.csv in {a.input_dir}")
    rows = []
    for f in files:
        site, ep, panel = NAME.match(f.name).groups()
        df = pd.read_csv(f, usecols=["split", "pool", "p0", "A_M"])
        for split in ("test", "audit"):
            g = df[df.split == split]
            if g.empty:
                continue
            d = dec[(dec.split == split) & (dec.site == site) & (dec.endpoint == ep) & (dec.panel == panel)]
            if len(d) != 1:
                print(f"  skip {f.stem} {split}: no grid-rule row in patient_level_decomposition.csv (rerun run_patient_level.py)")
                continue
            d = d.iloc[0]
            AM = g.A_M.to_numpy().astype(bool); p0 = g.p0.to_numpy(float)
            per_pool = g.groupby("pool").A_M.agg(["size", "sum"])
            if not ((per_pool["size"] == a.m) & (per_pool["sum"] == a.K)).all():
                sys.exit(f"{f.name} [{split}]: pools are not complete with K={a.K} of m={a.m} allocated by Margin")
            rows.append(dict(split=split, site=site, endpoint=ep, panel=panel, comparison=d.comparison, n=int(len(g)), pi=pi,
                             threshold_margin=float(d.threshold_margin), threshold_random=float(d.threshold_random),
                             **rates(p0, AM, float(d.threshold_margin), float(d.threshold_random), pi)))
    out = pd.DataFrame(rows)
    out = pd.concat([out[out.split == "test"], out[out.split == "audit"]], ignore_index=True)
    out.to_csv(pl / "patient_level_tau_sensitivity.csv", index=False)
    show = out[["split", "comparison", "u0", "u1", "u0_minus_u1", "band_violation_pct", "B_tau_0.05_pp", "B_tau_0.05_eq9_pp"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"written {pl / 'patient_level_tau_sensitivity.csv'} ({len(out)} rows, aggregate rates only)")


if __name__ == "__main__":
    main()
