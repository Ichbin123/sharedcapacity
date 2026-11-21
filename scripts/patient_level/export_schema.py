#!/usr/bin/env python3
"""Schema, writer and validator for patient-level exports. RUN ONLY INSIDE THE CREDENTIALED ENVIRONMENT.

One CSV per comparison, named <site>_<endpoint>_<panel>.csv with site in {mimic, mcmed}, endpoint in
{critical, death30, disposition} and panel in {cbc, chem7}. Columns:
  split   cal | test | audit   (cal = threshold-calibration set; audit only for MIMIC death30)
  pool    integer pool index within split; every pool has exactly m (16) rows
  y       0/1 outcome label
  p0, p1  no-panel and panel-augmented predicted risks
  A_M     0/1 Margin allocation used in the original analysis (K = 5 per pool)
  A_R_0 … A_R_9   optional 0/1 original Random allocations (needed to reproduce the original numbers exactly)
No identifiers or timestamps may be exported.

  python scripts/patient_level/export_schema.py --validate /secure/exports
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ["split", "pool", "y", "p0", "p1", "A_M"]
SPLITS = ("cal", "test", "audit")
NAME_RE = re.compile(r"^(mimic|mcmed)_(critical|death30|disposition)_(cbc|chem7)\.csv$")
FORBIDDEN = {"subject_id", "hadm_id", "stay_id", "charttime", "intime", "outtime", "dod", "mrn", "csn", "patient_id", "encounter_id",
             "visit_id", "admittime", "dischtime", "deathtime", "anchor_age", "anchor_year", "dob"}


def write_comparison(path, splits: dict, m: int = 16, K: int = 5):
    """splits = {'cal': dict(pool, y, p0, p1, A_M[, A_R]), 'test': {...}[, 'audit': {...}]}; A_R has shape (draws, n)."""
    frames = []
    for name, s in splits.items():
        df = pd.DataFrame(dict(split=name, pool=np.asarray(s["pool"], int), y=np.asarray(s["y"], int), p0=np.asarray(s["p0"], float),
                               p1=np.asarray(s["p1"], float), A_M=np.asarray(s["A_M"], int)))
        if s.get("A_R") is not None:
            AR = np.asarray(s["A_R"], int)
            for r in range(AR.shape[0]):
                df[f"A_R_{r}"] = AR[r]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    errs = validate_frame(out, Path(path).name, m, K)
    if errs:
        raise ValueError("; ".join(errs))
    out.to_csv(path, index=False)


def validate_frame(df: pd.DataFrame, name: str, m: int = 16, K: int = 5) -> list[str]:
    errs = []
    if not NAME_RE.match(name):
        errs.append(f"file name must be <site>_<endpoint>_<panel>.csv, got {name}")
    if [c for c in df.columns if c.lower() in FORBIDDEN]:
        errs.append(f"identifier/time columns must not be exported: {[c for c in df.columns if c.lower() in FORBIDDEN]}")
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        return errs + [f"missing columns {miss}"]
    if not set(df.split.unique()) <= set(SPLITS):
        errs.append("split must be one of cal, test, audit")
    errs += [f"split '{s}' missing" for s in ("cal", "test") if s not in set(df.split)]
    if not set(pd.unique(df.y)) <= {0, 1}:
        errs.append("y must be 0/1")
    for c in ("p0", "p1"):
        if df[c].isna().any() or (df[c] < 0).any() or (df[c] > 1).any():
            errs.append(f"{c} must be probabilities in [0, 1] without missing values")
    ar = [c for c in df.columns if c.startswith("A_R_")]
    for s, g in df.groupby("split"):
        size = g.groupby("pool").size()
        if (size != m).any():
            errs.append(f"{s}: every pool must have exactly {m} rows ({int((size != m).sum())} do not)")
            continue
        if (g.groupby("pool").A_M.sum() != K).any():
            errs.append(f"{s}: A_M must allocate exactly {K} encounters per pool")
        for c in ar:
            if (g.groupby("pool")[c].sum() != K).any():
                errs.append(f"{s}: {c} must allocate exactly {K} encounters per pool"); break
        if g.y.sum() == 0:
            errs.append(f"{s}: no positive labels")
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", required=True, help="directory with exported CSV files")
    ap.add_argument("--m", type=int, default=16); ap.add_argument("--K", type=int, default=5)
    a = ap.parse_args()
    files = sorted(Path(a.validate).glob("*.csv")); bad = 0
    if not files:
        print("no CSV files found"); sys.exit(1)
    for f in files:
        df = pd.read_csv(f); errs = validate_frame(df, f.name, a.m, a.K)
        info = ", ".join(f"{s}: {int((df.split == s).sum()):,} rows, {int(df[df.split == s].y.sum()):,} positives" for s in SPLITS if (df.split == s).any())
        print(("OK    " if not errs else "ERROR ") + f.name + "  " + info + ("" if any(c.startswith("A_R_") for c in df.columns) else "  [no A_R columns: Random will be re-drawn]"))
        for e in errs:
            print("      -", e)
        bad += bool(errs)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
