from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DERIVED = RESULTS / "10_refactor_derived"
FIG = ROOT / "figures"
MS = ROOT / "manuscript"
for _p in (DERIVED, FIG / "main", FIG / "supplementary"):
    _p.mkdir(parents=True, exist_ok=True)
