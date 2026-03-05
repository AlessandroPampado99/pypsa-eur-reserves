#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Sobol states from reserve_scheme.csv.

Input:
- reserve_scheme.csv with at least columns: sobol_idx, var, kind, min, max

Output:
- sobol_states.csv with one column per 'var' in Sobol order

Notes:
- Uses scipy.stats.qmc.Sobol if available.
- For int_mod variables, maps to integers in [min, max] inclusive.
- For float variables, maps to floats in [min, max].
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import qmc
except Exception as e:
    raise ImportError(
        "scipy is required for Sobol generation. Install with: pip install scipy"
    ) from e


def _map_to_int(u: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Map u in [0,1) to integers in [lo, hi] inclusive."""
    if hi < lo:
        raise ValueError(f"Invalid int range: [{lo}, {hi}]")
    span = hi - lo + 1
    x = lo + np.floor(u * span).astype(int)
    # Just in case numerical issues produce hi+1
    x = np.clip(x, lo, hi)
    return x


def _map_to_float(u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map u in [0,1) to floats in [lo, hi]."""
    if hi < lo:
        raise ValueError(f"Invalid float range: [{lo}, {hi}]")
    return lo + u * (hi - lo)


def load_scheme(scheme_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(scheme_csv)
    required = {"sobol_idx", "var", "kind", "min", "max"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"reserve_scheme.csv missing required columns: {sorted(missing)}")

    df = df.sort_values("sobol_idx").reset_index(drop=True)
    df["var"] = df["var"].astype(str)

    # Normalize kind
    df["kind"] = df["kind"].astype(str).str.strip()

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Sobol states from reserve_scheme.csv")
    parser.add_argument("--scheme", type=str, required=True, help="Path to reserve_scheme.csv")
    parser.add_argument("--out", type=str, required=True, help="Output sobol_states.csv path")
    parser.add_argument("--n", type=int, default=1024, help="Number of Sobol points to generate")
    parser.add_argument("--scramble", action="store_true", help="Use scrambled Sobol (recommended)")
    parser.add_argument("--seed", type=int, default=12345, help="Seed for scrambling")
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of initial Sobol points to skip (optional)",
    )
    parser.add_argument(
        "--use-base2",
        action="store_true",
        help="Generate using random_base2(m). Requires n to be a power of 2.",
    )
    args = parser.parse_args()

    scheme_path = Path(args.scheme)
    out_path = Path(args.out)

    df = load_scheme(scheme_path)
    vars_order = df["var"].tolist()
    kinds = df["kind"].tolist()

    # Parse ranges
    mins = df["min"].to_numpy()
    maxs = df["max"].to_numpy()

    # Determine Sobol dimension
    d = len(df)
    if d <= 0:
        raise ValueError("Scheme has no variables (empty).")

    n = int(args.n)
    if n <= 0:
        raise ValueError("--n must be > 0")

    # Initialize Sobol engine
    engine = qmc.Sobol(d=d, scramble=bool(args.scramble), seed=int(args.seed))

    # Skip initial points if requested
    if int(args.skip) > 0:
        _ = engine.random(int(args.skip))

    # Generate points in [0, 1)
    if args.use_base2:
        # n must be power of two
        m = int(np.log2(n))
        if 2**m != n:
            raise ValueError("--use-base2 requires --n to be a power of 2 (e.g., 512, 1024, 2048).")
        u = engine.random_base2(m=m)
    else:
        u = engine.random(n=n)

    # Map to physical space
    out = np.zeros((n, d), dtype=float)

    for j, (k, lo, hi) in enumerate(zip(kinds, mins, maxs)):
        if k == "int_mod":
            lo_i = int(round(float(lo)))
            hi_i = int(round(float(hi)))
            out[:, j] = _map_to_int(u[:, j], lo_i, hi_i)
        elif k == "float":
            out[:, j] = _map_to_float(u[:, j].astype(float), float(lo), float(hi))
        else:
            raise ValueError(f"Unknown kind='{k}' for var='{vars_order[j]}'")

    states_df = pd.DataFrame(out, columns=vars_order)

    # Ensure int_mod columns are int dtype
    for j, k in enumerate(kinds):
        if k == "int_mod":
            states_df[vars_order[j]] = states_df[vars_order[j]].round().astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    states_df.to_csv(out_path, index=False)

    print(f"[OK] Wrote Sobol states: {out_path}")
    print(f"     n={n}, d={d}, scramble={bool(args.scramble)}, skip={int(args.skip)}, base2={bool(args.use_base2)}")
    print("     Columns:", ", ".join(vars_order))


if __name__ == "__main__":
    main()

# python scripts/reserves/generate_sobol_states.py \
#   --scheme /home/pampado/reserves/pypsa-eur/results/basecase/europe/reserves_preprocess/reserve_scheme.csv \
#   --out    /home/pampado/reserves/pypsa-eur/resources/basecase/europe/reserves/sobol_states.csv \
#   --n 1024 \
#   --scramble \
#   --seed 12345 \
#   --use-base2