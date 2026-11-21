#!/usr/bin/env python3
"""Recompute aggregate-level tables and illustrative figures from public summary inputs.

This does not rerun restricted EHR patient-level processing and does not rebuild
manuscript Word/PDF files. Figure styling may differ from the final paper.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sharedcap import derived, figures

if __name__ == "__main__":
    derived.run()
    figures.run()
    print("Wrote derived tables to results/10_refactor_derived and figures to figures/.")
