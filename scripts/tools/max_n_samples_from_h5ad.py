#!/usr/bin/env python3
"""Print max paired evaluation n_samples for an h5ad (for shell scripts)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.max_eval_samples import resolve_eval_n_samples


def main() -> None:
    p = argparse.ArgumentParser(description="Compute max eval n_samples from h5ad")
    p.add_argument("h5ad", type=str, help="Path to test/valid h5ad")
    p.add_argument(
        "--mode",
        choices=["paired_ifn", "multi_pert", "timepoint"],
        default="paired_ifn",
    )
    p.add_argument("--status-col", default="perturbation_status")
    p.add_argument("--control", default="Control")
    p.add_argument("--stimulated", default="IFN")
    p.add_argument("--time-col", default="treatment_time")
    p.add_argument(
        "--requested",
        type=int,
        default=0,
        help="If >0, return min(requested, max_available); else max_available",
    )
    args = p.parse_args()
    n = resolve_eval_n_samples(
        args.h5ad,
        args.requested,
        mode=args.mode,
        status_col=args.status_col,
        control_value=args.control,
        stimulated_value=args.stimulated,
        time_col=args.time_col,
    )
    print(n)


if __name__ == "__main__":
    main()
