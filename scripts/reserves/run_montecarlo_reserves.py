#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monte Carlo reserve sizing runner.

Inputs:
- reserve_scheme.csv: defines Sobol dimensions (int_mod + float) and parameters:
    - int_mod: var, p_module_mw, n_max, prob_trip_per_min
    - float:   var, std_dev_rel (placeholder; refine later)
- sobol_states.csv: one row = one state, columns matching scheme vars.

Outputs:
- mc_reserves.csv: reserve results per state (FCR/aFRR/mFRR/RR up/down) + diagnostics

Notes:
- Tripping events are generated for ALL maximum modules (n_max) and then
  each state selects the first n_online modules.
- Sigma model is a placeholder: sigma = sqrt(sum((std_rel * value)^2)) over selected vars.
  We'll refine load uncertainty later without touching the MC plumbing.
"""

from __future__ import annotations

import argparse
import gzip
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # PyYAML
except ImportError as e:
    raise ImportError("PyYAML is required (pip install pyyaml).") from e


# -----------------------------
# Config defaults (override via --config)
# -----------------------------

DEFAULT_CFG: Dict[str, Any] = {
    # Monte Carlo horizon
    "years": 2,  # set 192 when you are ready, but start small for sanity checks
    "seed": 12345,

    # Trip model
    "t_trip_min": 30,  # minutes down after a trip
    # If prob_trip_per_min is 0, that cluster never trips.

    # Ramp sawtooth (15-min periodic)
    "t_frr_min": 15,
    "ramp_divisor": 8.0,  # matches Enrico's ramp_value = DDM/8 with DDM=1

    # Quantiles
    "q_fcr": 0.997,
    "q_afrr": 0.95,
    "q_mfrr": 0.99,
    "q_rr": 0.997,

    # Sigma model placeholder
    # Which float vars contribute to the Gaussian forecast-error part.
    # Default: all float vars except "Ramp" (ramp handled separately).
    "sigma_from_vars": None,

    # Safety/performance
    "dtype": "float32",  # float32 saves RAM a lot
    "states_start": 0,   # run a slice [start, stop)
    "states_stop": None,

    # Caching tripping events (recommended once years is large)
    "cache_tripping": True,
}


# -----------------------------
# Helpers
# -----------------------------

def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries."""
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_cfg(path: Path | None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CFG)
    if path is None:
        return cfg
    with path.open("r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    return deep_update(cfg, user)


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sawtooth_sequence(tMC: int, period: int = 15) -> np.ndarray:
    """
    Sawtooth from -1 to +1 over 'period' samples, repeated to length tMC.
    """
    reps = tMC // period
    rem = tMC % period
    single = np.linspace(-1.0, 1.0, period, dtype=np.float32)
    seq = np.tile(single, reps)
    if rem > 0:
        seq = np.concatenate([seq, np.linspace(-1.0, 1.0, rem, dtype=np.float32)])
    return seq.astype(np.float32)


def geometric_trip_times(p: float, tMC: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate Bernoulli-per-minute trip times using geometric waiting times.
    Exact for independent Bernoulli trials with probability p each minute.

    Returns sorted integer times in [0, tMC).
    """
    if p <= 0.0:
        return np.empty(0, dtype=np.int64)

    # Geometric distribution in numpy returns number of trials until success (1..inf)
    # If we consider each minute as a trial, waiting time in minutes is geom(p).
    times: List[int] = []
    t = 0
    while True:
        dt = int(rng.geometric(p))  # >= 1
        t += dt
        if t >= tMC:
            break
        times.append(t)
    return np.array(times, dtype=np.int64)


@dataclass
class ClusterSpec:
    var: str
    group: str
    p_module_mw: float
    n_max: int
    prob_trip_per_min: float


def load_scheme(scheme_csv: Path) -> Tuple[List[ClusterSpec], pd.DataFrame]:
    """
    Load reserve_scheme.csv and return:
    - list of int_mod clusters in Sobol order
    - scheme dataframe (for float vars etc.)
    """
    df = pd.read_csv(scheme_csv)
    if "sobol_idx" in df.columns:
        df = df.sort_values("sobol_idx").reset_index(drop=True)

    needed = {"var", "kind"}
    if not needed.issubset(df.columns):
        raise ValueError(f"reserve_scheme.csv must include columns {sorted(needed)}")

    int_df = df[df["kind"].astype(str) == "int_mod"].copy()
    float_df = df[df["kind"].astype(str) == "float"].copy()

    for col in ["p_module_mw", "n_max", "prob_trip_per_min", "group"]:
        if col not in int_df.columns:
            raise ValueError(f"int_mod rows require column '{col}' in reserve_scheme.csv")

    clusters: List[ClusterSpec] = []
    for _, r in int_df.iterrows():
        clusters.append(
            ClusterSpec(
                var=str(r["var"]),
                group=str(r.get("group", "")),
                p_module_mw=float(r["p_module_mw"]),
                n_max=int(float(r["n_max"])),
                prob_trip_per_min=float(r.get("prob_trip_per_min", 0.0)) if str(r.get("prob_trip_per_min", "")).strip() != "" else 0.0,
            )
        )

    return clusters, df


def build_module_index(clusters: List[ClusterSpec]) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Build global module indexing:
    - total_modules
    - slices per cluster: (start, end) in [0, total_modules)
    """
    slices: List[Tuple[int, int]] = []
    start = 0
    for c in clusters:
        end = start + int(c.n_max)
        slices.append((start, end))
        start = end
    return start, slices


def cache_save(path: Path, obj: Any) -> None:
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def cache_load(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def generate_tripping_events(
    clusters: List[ClusterSpec],
    tMC: int,
    seed: int,
    cache_path: Path | None = None,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Generate trip time arrays for each module (global indexing).
    Returns:
    - events: list length total_modules; events[i] = np.ndarray of trip times
    - slices per cluster
    """
    total_modules, slices = build_module_index(clusters)
    rng = np.random.default_rng(seed)

    # Build events list
    events: List[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(total_modules)]

    for c, (s0, s1) in zip(clusters, slices):
        p = float(c.prob_trip_per_min)
        if p <= 0.0:
            continue
        for mi in range(s0, s1):
            events[mi] = geometric_trip_times(p, tMC, rng)

    if cache_path is not None:
        cache_save(cache_path, {"tMC": tMC, "seed": seed, "events": events, "slices": slices})

    return events, slices


def trip_imbalance_timeseries(
    clusters: List[ClusterSpec],
    slices: List[Tuple[int, int]],
    tripping_events: List[np.ndarray],
    n_online_by_cluster: List[int],
    tMC: int,
    t_trip: int,
    dtype: np.dtype,
) -> np.ndarray:
    """
    Build trip imbalance time series (MW) for a given state.
    """
    trip = np.zeros(tMC, dtype=dtype)
    shifts = np.arange(t_trip, dtype=np.int64)

    for c, (s0, s1), n_on in zip(clusters, slices, n_online_by_cluster):
        if n_on <= 0:
            continue
        n_on = min(n_on, c.n_max)
        if c.prob_trip_per_min <= 0.0:
            continue

        # Take first n_on modules as online (modules are identical)
        mod_indices = range(s0, s0 + n_on)

        # Concatenate all trip times from online modules
        times_list = [tripping_events[i] for i in mod_indices if tripping_events[i].size > 0]
        if not times_list:
            continue
        times = np.concatenate(times_list).astype(np.int64, copy=False)

        # Expand each trip time to t_trip minutes outage window
        # all_idxs shape (n_trips, t_trip) -> flattened
        all_idxs = (times[:, None] + shifts[None, :]).ravel()
        all_idxs = all_idxs[all_idxs < tMC]
        if all_idxs.size == 0:
            continue

        counts = np.bincount(all_idxs, minlength=tMC).astype(dtype, copy=False)
        trip += counts * dtype.type(c.p_module_mw)

    return trip


def sigma_from_state(
    state_row: pd.Series,
    float_vars: List[str],
    std_dev_rel_by_var: Dict[str, float],
    sigma_from_vars: List[str] | None,
) -> float:
    """
    Placeholder sigma model:
      sigma = sqrt( sum( (std_rel[var] * value[var])^2 ) )
    """
    if sigma_from_vars is None:
        sigma_vars = [v for v in float_vars if v.lower() != "ramp"]
    else:
        sigma_vars = sigma_from_vars

    var_terms = []
    for v in sigma_vars:
        if v not in state_row.index:
            continue
        val = float(state_row[v])
        std = float(std_dev_rel_by_var.get(v, 0.0))
        var_terms.append((std * val) ** 2)

    sigma = math.sqrt(sum(var_terms)) if var_terms else 0.0
    return float(sigma)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Monte Carlo reserve sizing on Sobol states")
    parser.add_argument("--scheme", type=str, required=True, help="Path to reserve_scheme.csv")
    parser.add_argument("--states", type=str, required=True, help="Path to Sobol states CSV (columns must match scheme vars)")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config for MC settings")
    parser.add_argument("--tripping-cache", type=str, default=None, help="Optional path to tripping cache .pkl.gz")
    args = parser.parse_args()

    scheme_path = Path(args.scheme)
    states_path = Path(args.states)
    outdir = Path(args.outdir)
    cfg_path = Path(args.config) if args.config else None
    cache_path = Path(args.tripping_cache) if args.tripping_cache else (outdir / "tripping_events.pkl.gz")

    safe_mkdir(outdir)

    cfg = load_cfg(cfg_path)

    years = int(cfg["years"])
    seed = int(cfg["seed"])
    t_trip = int(cfg["t_trip_min"])
    t_frr = int(cfg["t_frr_min"])
    ramp_div = float(cfg["ramp_divisor"])
    q_fcr = float(cfg["q_fcr"])
    q_afrr = float(cfg["q_afrr"])
    q_mfrr = float(cfg["q_mfrr"])
    q_rr = float(cfg["q_rr"])
    cache_tripping = bool(cfg.get("cache_tripping", True))

    dtype = np.float32 if str(cfg.get("dtype", "float32")).lower() == "float32" else np.float64

    tMC = years * 365 * 24 * 60
    if tMC % t_frr != 0:
        raise ValueError(f"tMC={tMC} not divisible by t_frr={t_frr}. Use integer years or adjust.")

    clusters, scheme_df = load_scheme(scheme_path)
    all_vars = scheme_df["var"].astype(str).tolist()

    # Load states
    states_df = pd.read_csv(states_path)
    missing_cols = [v for v in all_vars if v not in states_df.columns]
    if missing_cols:
        raise ValueError(f"Sobol states missing columns from scheme: {missing_cols[:20]} (and {max(0, len(missing_cols)-20)} more)")

    # Slice states
    start = int(cfg.get("states_start", 0))
    stop = cfg.get("states_stop", None)
    stop = int(stop) if stop is not None else None
    states_df = states_df.iloc[start:stop].reset_index(drop=True)

    # Determine float vars from scheme
    float_vars = scheme_df[scheme_df["kind"].astype(str) == "float"]["var"].astype(str).tolist()

    # std_dev_rel map for float vars
    std_dev_rel_by_var: Dict[str, float] = {}
    if "std_dev_rel" in scheme_df.columns:
        for _, r in scheme_df[scheme_df["kind"].astype(str) == "float"].iterrows():
            v = str(r["var"])
            val = r.get("std_dev_rel", 0.0)
            try:
                std_dev_rel_by_var[v] = float(val)
            except Exception:
                std_dev_rel_by_var[v] = 0.0

    sigma_from_vars = cfg.get("sigma_from_vars", None)

    # Precompute ramp saw sequence
    ramp_seq = sawtooth_sequence(tMC, period=t_frr) / dtype.type(ramp_div)

    # Tripping events cache
    events = None
    slices = None
    if cache_tripping and cache_path.exists():
        payload = cache_load(cache_path)
        if int(payload.get("tMC", -1)) == tMC and int(payload.get("seed", -2)) == seed:
            events = payload["events"]
            slices = payload["slices"]

    if events is None or slices is None:
        print(">>> Generating tripping events (cache miss or disabled)...")
        events, slices = generate_tripping_events(
            clusters=clusters,
            tMC=tMC,
            seed=seed,
            cache_path=cache_path if cache_tripping else None,
        )
        if cache_tripping:
            print(f"[OK] Cached tripping events: {cache_path}")

    # Run MC per state
    rng = np.random.default_rng(seed + 1)  # separate stream from trip generation

    results_rows: List[Dict[str, Any]] = []

    n_quart = tMC // t_frr
    for i, row in states_df.iterrows():
        # n_online per cluster from state
        n_online = []
        for c in clusters:
            val = int(float(row[c.var]))
            n_online.append(max(0, val))

        # Trip imbalance
        trip_imb = trip_imbalance_timeseries(
            clusters=clusters,
            slices=slices,
            tripping_events=events,
            n_online_by_cluster=n_online,
            tMC=tMC,
            t_trip=t_trip,
            dtype=dtype,
        )

        # Ramp matrix: ramp_val * saw/8 (as in Enrico)
        ramp_val = float(row["Ramp"]) if "Ramp" in row.index else 0.0
        ramp_matrix = dtype.type(ramp_val) * ramp_seq

        # Sigma placeholder (we refine later)
        sigma = sigma_from_state(
            state_row=row,
            float_vars=float_vars,
            std_dev_rel_by_var=std_dev_rel_by_var,
            sigma_from_vars=sigma_from_vars,
        )

        # Gaussian part
        if sigma > 0.0:
            df_t = rng.normal(loc=0.0, scale=sigma, size=tMC).astype(dtype, copy=False)
        else:
            df_t = np.zeros(tMC, dtype=dtype)

        tot_imb = df_t + trip_imb + ramp_matrix

        # FCR quantiles
        fcr_up = float(np.quantile(tot_imb, q_fcr))
        fcr_dw = float(np.quantile(tot_imb, 1.0 - q_fcr))

        # 15-min filtering for FRR logic
        mean_imb = tot_imb.reshape(n_quart, t_frr).mean(axis=1)  # shape (n_quart,)
        tot_imb_frr = tot_imb - np.repeat(mean_imb, t_frr).astype(dtype, copy=False)

        afrr_up = float(np.quantile(tot_imb_frr, q_afrr))
        afrr_dw = float(np.quantile(tot_imb_frr, 1.0 - q_afrr))

        mfrr_up = -afrr_up + float(np.quantile(tot_imb_frr, q_mfrr))
        mfrr_dw = -afrr_dw + float(np.quantile(tot_imb_frr, 1.0 - q_mfrr))

        rr_up = -afrr_up - mfrr_up + float(np.quantile(tot_imb_frr, q_rr))
        rr_dw = -afrr_dw - mfrr_dw + float(np.quantile(tot_imb_frr, 1.0 - q_rr))

        # Diagnostics similar to Enrico
        trip_p997 = float(np.quantile(trip_imb, 0.997))
        ea_trip = float(trip_imb.sum() / years / 60.0)  # MW-min -> MWh/yr

        out = {
            "state_idx": i + start,
            "sigma": sigma,
            "FCR_up": fcr_up,
            "FCR_dw": fcr_dw,
            "aFRR_up": afrr_up,
            "aFRR_dw": afrr_dw,
            "mFRR_up": mfrr_up,
            "mFRR_dw": mfrr_dw,
            "RR_up": rr_up,
            "RR_dw": rr_dw,
            "trip_p99_7": trip_p997,
            "EA_trip_MWh_per_yr": ea_trip,
        }
        results_rows.append(out)

        if (i + 1) % 10 == 0:
            print(f">>> done {i+1}/{len(states_df)} states")

    res_df = pd.DataFrame(results_rows)
    out_csv = outdir / "mc_reserves.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote: {out_csv}")


if __name__ == "__main__":
    main()