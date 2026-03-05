#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reserve preprocessing builder for PyPSA-eur solved networks.

What it does:
- Load a solved PyPSA Network (.nc)
- Build deterministic time series: Load, RES availability (aggregated by availability groups), RD, Ramp
- Build per-group capacity envelopes from selected components (generators, storage_units, stores; links optional)
- Discretize each group into module clusters (1 or multiple per group) using module_sizes CSV or config defaults
- Output:
  - reserve_timeseries.csv (optional)
  - reserve_caps.csv
  - reserve_scheme.csv (Sobol dimensions: int_mod + float vars)
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import pypsa

try:
    import yaml  # PyYAML
except ImportError as e:
    raise ImportError("PyYAML is required (pip install pyyaml).") from e


# -----------------------------
# Defaults (override via --config)
# -----------------------------

DEFAULT_CFG: Dict[str, Any] = {
    # Mapping carrier -> group (regex, first match wins).
    "group_patterns": [
        {"pattern": r"\bCCGT\b", "group": "gas_ccgt"},
        {"pattern": r"\bOCGT\b", "group": "gas_ocgt"},
        {"pattern": r"oil|diesel", "group": "diesel"},
        {"pattern": r"coal", "group": "coal"},
        {"pattern": r"lignite", "group": "lignite"},
        {"pattern": r"nuclear", "group": "nuclear"},
        {"pattern": r"biomass", "group": "biomass"},
        {"pattern": r"hydro|PHS|reservoir", "group": "hydro"},
        {"pattern": r"battery|li.?ion", "group": "liion"},
    ],

    # Availability groups for renewables (float vars). Each becomes a column like "Solar_avail" [MW].
    # IMPORTANT: These are selected ONLY from n.generators (not storage_units).
    "availability_groups": [
        {"name": "Solar_avail", "carriers": ["solar", "solar-hsat"]},
        {"name": "Onwind_avail", "carriers": ["onwind"]},
        # By default include all offwind types; you can narrow to only offwind-ac in config
        {"name": "Offwind_avail", "carriers": ["offwind-ac", "offwind-dc", "offwind-float"]},
    ],

    # Float vars to include in Sobol state; if empty/null, it is auto-built as:
    # availability group names + ["Load", "Ramp"]
    "float_vars": None,

    # Which availability group names contribute to RD = Load - sum(availability_vars)
    # If empty/null => all availability groups are included.
    "rd_availability_vars": None,

    # Components to aggregate for capacity envelopes and modularization.
    # NOTE: in PyPSA-eur batteries are often Store+Link; excluding links might miss battery power.
    "include_components": ["generators", "storage_units", "stores"],
    "include_links": False,  # optional
    "exclude_link_carriers": ["battery charger"],

    # Margins for capacity envelope (P_total_max = P_total_opt * (1 + margin))
    "margin_non_res": 0.10,
    "margin_res": 0.30,  # not used unless you classify groups as RES; kept for future
    "margin_by_group": {},

    # Module clusters:
    # - If you provide --module-sizes CSV, it is used (supports multiple clusters per group with 'share').
    # - Otherwise, use a single module size per group from p_module_by_group.
    "p_module_by_group": {
        "diesel": 155.0,
        "gas_ccgt": 55.0,
        "gas_ocgt": 55.0,
        "coal": 195.0,
        "lignite": 195.0,
        "nuclear": 930.0,
        "hydro": 175.0,
        "liion": 100.0,
        "biomass": 50.0,
    },
    "default_p_module_mw": 100.0,

    # Cap n_max for MC feasibility (set null to disable)
    "n_max_cap": 500,

    # Stochastic defaults (used later by Monte Carlo; safe to keep 0 for now)
    "default_prob_trip_per_min": 0.0,
    "prob_trip_by_group": {},

    # Relative std dev for float vars (Load / RES availability / Ramp). Default from config.
    "default_std_dev_rel": 0.0,
    "std_dev_rel_by_var": {
        "Load": 0.0,
        "Ramp": 0.0,
        # Availability vars can be added here, e.g. Solar_avail: 0.15
    },

    # Stores: to convert Store energy capacity (MWh) into a proxy power capacity (MW),
    # you can set a duration (hours): P = E / hours.
    # If None, stores are NOT used for envelopes/modularization.
    "default_store_power_hours": None,
    "store_power_hours_by_group": {},

    # If you want to force which groups appear as int_mod, list them here.
    # If None => infer all groups present with capacity > 0 (excluding load/ramp and availability vars).
    "int_mod_groups": None,

    # Whether to write reserve_timeseries.csv
    "write_timeseries": True,
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


def load_cfg(cfg_path: Path | None) -> Dict[str, Any]:
    """Load YAML config and merge into defaults."""
    cfg = dict(DEFAULT_CFG)
    if cfg_path is None:
        return cfg
    with cfg_path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return deep_update(cfg, user_cfg)


def compile_patterns(pattern_rows: List[Dict[str, str]]) -> List[Tuple[re.Pattern, str]]:
    compiled: List[Tuple[re.Pattern, str]] = []
    for row in pattern_rows:
        pat = re.compile(str(row["pattern"]), flags=re.IGNORECASE)
        grp = str(row["group"]).strip()
        compiled.append((pat, grp))
    return compiled


def carrier_to_group(carrier: str, compiled: List[Tuple[re.Pattern, str]]) -> str:
    c = str(carrier or "").strip()
    for pat, grp in compiled:
        if pat.search(c):
            return grp
    return "other"


def get_cap_p_nom(df: pd.DataFrame) -> pd.Series:
    """Return MW capacity: p_nom_opt if available, else p_nom; NaNs -> 0."""
    if "p_nom_opt" in df.columns:
        cap = df["p_nom_opt"].copy()
        if "p_nom" in df.columns:
            cap = cap.where(cap.notna(), df["p_nom"])
    elif "p_nom" in df.columns:
        cap = df["p_nom"].copy()
    else:
        cap = pd.Series(0.0, index=df.index)
    return pd.to_numeric(cap, errors="coerce").fillna(0.0)


def get_cap_e_nom(df: pd.DataFrame) -> pd.Series:
    """Return MWh energy capacity: e_nom_opt if available, else e_nom; NaNs -> 0."""
    if "e_nom_opt" in df.columns:
        cap = df["e_nom_opt"].copy()
        if "e_nom" in df.columns:
            cap = cap.where(cap.notna(), df["e_nom"])
    elif "e_nom" in df.columns:
        cap = df["e_nom"].copy()
    else:
        cap = pd.Series(0.0, index=df.index)
    return pd.to_numeric(cap, errors="coerce").fillna(0.0)


def sum_load_ts(n: pypsa.Network) -> pd.Series:
    """Total load time series (MW)."""
    if hasattr(n, "loads_t") and hasattr(n.loads_t, "p_set") and n.loads_t.p_set is not None and not n.loads_t.p_set.empty:
        return n.loads_t.p_set.sum(axis=1)
    if "p_set" in n.loads.columns:
        static = pd.to_numeric(n.loads["p_set"], errors="coerce").fillna(0.0).sum()
        return pd.Series(static, index=n.snapshots, name="Load")
    return pd.Series(0.0, index=n.snapshots, name="Load")


def availability_ts_from_generators(
    n: pypsa.Network,
    carriers: List[str],
) -> pd.Series:
    """Compute sum(p_nom * p_max_pu) for generators with carrier in carriers."""
    if not carriers:
        return pd.Series(0.0, index=n.snapshots)

    gens = n.generators
    if gens.empty:
        return pd.Series(0.0, index=n.snapshots)

    mask = gens["carrier"].astype(str).isin([str(c) for c in carriers])
    names = gens.index[mask].tolist()
    if not names:
        return pd.Series(0.0, index=n.snapshots)

    cap = get_cap_p_nom(gens).loc[names]

    pmaxpu = None
    if hasattr(n, "generators_t") and hasattr(n.generators_t, "p_max_pu") and n.generators_t.p_max_pu is not None:
        pmaxpu = n.generators_t.p_max_pu
    if pmaxpu is None or pmaxpu.empty:
        return pd.Series(float(cap.sum()), index=n.snapshots)

    cols = [g for g in names if g in pmaxpu.columns]
    if not cols:
        return pd.Series(float(cap.sum()), index=n.snapshots)

    ts = pmaxpu[cols].mul(cap[cols], axis=1).sum(axis=1)
    ts = ts.reindex(n.snapshots).fillna(0.0)
    return ts


def compute_ramp_from_rd(rd: pd.Series) -> pd.Series:
    """
    Ramp indicator:
      0.5*|RD(t+1)-RD(t)| + 0.5*|RD(t)-RD(t-1)|
    with boundary handling.
    """
    if rd is None or rd.size == 0:
        return pd.Series([], dtype=float)

    rd = rd.astype(float)
    ramp = 0.5 * (rd.shift(-1) - rd).abs() + 0.5 * (rd - rd.shift(1)).abs()

    # boundary handling (avoid NaNs at edges)
    if rd.size >= 2:
        ramp.iloc[0] = abs(rd.iloc[1] - rd.iloc[0])
        ramp.iloc[-1] = abs(rd.iloc[-1] - rd.iloc[-2])
    else:
        ramp.iloc[0] = 0.0

    return ramp.fillna(0.0)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_module_clusters_csv(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read module clusters from CSV.

    Expected columns:
      - group (str)
      - p_module_mw (float)
    Optional:
      - share (float)  [0..1] (capacity share of the group assigned to this cluster)
      - label (str)    (used for naming columns; if missing, derived from p_module_mw)

    Returns:
      dict[group] = list of clusters:
        {"p_module_mw": float, "share": float|None, "label": str}
    """
    df = pd.read_csv(path)
    req = {"group", "p_module_mw"}
    if not req.issubset(df.columns):
        raise ValueError(f"module sizes CSV must contain {sorted(req)}; got {list(df.columns)}")

    clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for _, r in df.iterrows():
        g = str(r["group"]).strip()
        p = float(r["p_module_mw"])
        if not g or not math.isfinite(p) or p <= 0:
            continue
        share = None
        if "share" in df.columns and pd.notna(r.get("share")):
            try:
                share = float(r["share"])
            except Exception:
                share = None
        label = None
        if "label" in df.columns and pd.notna(r.get("label")):
            label = str(r["label"]).strip()
        if not label:
            label = str(int(round(p)))
        clusters[g].append({"p_module_mw": p, "share": share, "label": label})

    return dict(clusters)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build reserve preprocessing scheme from a solved PyPSA network")
    parser.add_argument("--network", type=str, required=True, help="Path to solved PyPSA network .nc")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for reserve preprocessing artifacts")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config to override defaults")
    parser.add_argument("--module-sizes", type=str, default=None, help="Optional CSV with module clusters (group,p_module_mw[,share,label])")
    args = parser.parse_args()

    network_path = Path(args.network)
    outdir = Path(args.outdir)
    cfg_path = Path(args.config) if args.config else None
    module_sizes_path = Path(args.module_sizes) if args.module_sizes else None

    safe_mkdir(outdir)

    cfg = load_cfg(cfg_path)
    compiled_patterns = compile_patterns(cfg["group_patterns"])

    print(f">>> Loading network: {network_path}")
    n = pypsa.Network(network_path)

    # -------------------------
    # A) Deterministic time series
    # -------------------------
    load_ts = sum_load_ts(n)

    availability_groups = cfg.get("availability_groups", []) or []
    avail_ts_map: Dict[str, pd.Series] = {}
    for ag in availability_groups:
        name = str(ag["name"]).strip()
        carriers = [str(c) for c in (ag.get("carriers", []) or [])]
        avail_ts_map[name] = availability_ts_from_generators(n, carriers)

    # RD contributors
    rd_vars = cfg.get("rd_availability_vars", None)
    if rd_vars is None:
        rd_vars = list(avail_ts_map.keys())
    else:
        rd_vars = [v for v in rd_vars if v in avail_ts_map]

    total_avail = pd.Series(0.0, index=n.snapshots)
    for v in rd_vars:
        total_avail = total_avail.add(avail_ts_map[v], fill_value=0.0)

    rd_ts = load_ts.sub(total_avail, fill_value=0.0)
    ramp_ts = compute_ramp_from_rd(rd_ts)

    ts_df = pd.DataFrame({"Load": load_ts, "RD": rd_ts, "Ramp": ramp_ts})
    for v, s in avail_ts_map.items():
        ts_df[v] = s
    ts_df.index.name = "snapshot"

    if bool(cfg.get("write_timeseries", True)):
        ts_out = outdir / "reserve_timeseries.csv"
        ts_df.to_csv(ts_out)
        print(f"[OK] Wrote timeseries: {ts_out}")

    # Float maxima
    float_max: Dict[str, float] = {
        "Load": float(ts_df["Load"].max()) if "Load" in ts_df else 0.0,
        "Ramp": float(ts_df["Ramp"].max()) if "Ramp" in ts_df else 0.0,
    }
    for v in avail_ts_map:
        float_max[v] = float(ts_df[v].max()) if v in ts_df else 0.0

    # Auto-build float vars if not provided
    float_vars = cfg.get("float_vars", None)
    if not float_vars:
        float_vars = list(avail_ts_map.keys()) + ["Load", "Ramp"]

    # -------------------------
    # B) Capacity envelopes by group (MW)
    # -------------------------
    include_components = set(cfg.get("include_components", ["generators", "storage_units", "stores"]))
    include_links = bool(cfg.get("include_links", False))

    group_caps_opt_mw = defaultdict(float)
    # generators
    if "generators" in include_components and hasattr(n, "generators") and n.generators is not None and not n.generators.empty:
        cap = get_cap_p_nom(n.generators)
        for name, row in n.generators.iterrows():
            grp = carrier_to_group(row.get("carrier", ""), compiled_patterns)
            group_caps_opt_mw[grp] += float(cap.loc[name])

    # storage_units
    if "storage_units" in include_components and hasattr(n, "storage_units") and n.storage_units is not None and not n.storage_units.empty:
        cap = get_cap_p_nom(n.storage_units)
        for name, row in n.storage_units.iterrows():
            grp = carrier_to_group(row.get("carrier", ""), compiled_patterns)
            group_caps_opt_mw[grp] += float(cap.loc[name])

    # stores (energy -> proxy power via hours)
    store_hours_default = cfg.get("default_store_power_hours", None)
    store_hours_by_group = dict(cfg.get("store_power_hours_by_group", {}))
    if "stores" in include_components and hasattr(n, "stores") and n.stores is not None and not n.stores.empty:
        if store_hours_default is None and not store_hours_by_group:
            print("[WARN] stores included but no store power conversion hours provided. Stores will be ignored for MW envelopes.")
        else:
            ecap = get_cap_e_nom(n.stores)  # MWh
            for name, row in n.stores.iterrows():
                grp = carrier_to_group(row.get("carrier", ""), compiled_patterns)
                hours = store_hours_by_group.get(grp, store_hours_default)
                if hours is None:
                    continue
                hours = float(hours)
                if hours <= 0:
                    continue
                group_caps_opt_mw[grp] += float(ecap.loc[name]) / hours

    exclude_link_carriers = {str(c).strip().lower() for c in (cfg.get("exclude_link_carriers", []) or [])}

    if include_links and hasattr(n, "links") and n.links is not None and not n.links.empty:
        cap = get_cap_p_nom(n.links)
        for name, row in n.links.iterrows():
            carrier = str(row.get("carrier", "")).strip().lower()
            if carrier in exclude_link_carriers:
                continue
            grp = carrier_to_group(row.get("carrier", ""), compiled_patterns)
            group_caps_opt_mw[grp] += float(cap.loc[name])

    if not include_links:
        # Practical warning for batteries in PyPSA-eur
        if any(
            "battery" in str(c).lower()
            for c in n.links.get("carrier", pd.Series([], dtype=str)).astype(str).unique()
        ):
            print("[WARN] include_links=False but links contain battery-like carriers. Battery power may be missing from MW envelopes.")
    
    # -------------------------
    # C) Module clusters per group
    # -------------------------
    # Load clusters from CSV if provided (supports multiple clusters per group).
    clusters_by_group: Dict[str, List[Dict[str, Any]]] = {}
    if module_sizes_path is not None:
        clusters_by_group = read_module_clusters_csv(module_sizes_path)

    p_module_by_group = dict(cfg.get("p_module_by_group", {}))
    default_p_module = float(cfg.get("default_p_module_mw", 100.0))

    # Groups to modularize (int_mod)
    forced_int_groups = cfg.get("int_mod_groups", None)
    if forced_int_groups:
        int_groups = [g for g in forced_int_groups if float(group_caps_opt_mw.get(g, 0.0)) > 0.0]
    else:
        # infer from present capacities, exclude availability pseudo-groups and load/ramp
        int_groups = [g for g, p in group_caps_opt_mw.items() if p > 0.0]
    int_groups = sorted(set(int_groups))

    # Margins
    margin_non_res = float(cfg.get("margin_non_res", 0.10))
    margin_by_group = dict(cfg.get("margin_by_group", {}))

    # n_max cap
    n_max_cap = cfg.get("n_max_cap", 500)
    if n_max_cap is not None:
        n_max_cap = int(n_max_cap)

    # stochastic defaults
    default_prob_trip = float(cfg.get("default_prob_trip_per_min", 0.0))
    prob_trip_by_group = dict(cfg.get("prob_trip_by_group", {}))

    default_std = float(cfg.get("default_std_dev_rel", 0.0))
    std_by_var = dict(cfg.get("std_dev_rel_by_var", {}))

    caps_rows: List[Dict[str, Any]] = []
    scheme_rows: List[Dict[str, Any]] = []
    sobol_idx = 0

    # int_mod rows (possibly multiple clusters per group)
    for g in int_groups:
        p_opt = float(group_caps_opt_mw.get(g, 0.0))
        if p_opt <= 0:
            continue

        margin = float(margin_by_group.get(g, margin_non_res))
        p_max_group = p_opt * (1.0 + margin)

        prob_trip = float(prob_trip_by_group.get(g, default_prob_trip))

        # clusters list
        clusters = clusters_by_group.get(g, None)
        if not clusters:
            # single cluster from config
            p_mod = float(p_module_by_group.get(g, default_p_module))
            clusters = [{"p_module_mw": p_mod, "share": 1.0, "label": str(int(round(p_mod)))}]
        else:
            # normalize shares if multiple clusters
            shares = []
            for c in clusters:
                s = c.get("share", None)
                shares.append(float(s) if (s is not None and math.isfinite(float(s)) and float(s) > 0) else None)
            if all(s is None for s in shares):
                # equal split
                k = len(clusters)
                for c in clusters:
                    c["share"] = 1.0 / k
            else:
                # fill missing with 0, then renormalize
                svals = [float(s) if s is not None else 0.0 for s in shares]
                tot = sum(svals)
                if tot <= 0:
                    k = len(clusters)
                    for c in clusters:
                        c["share"] = 1.0 / k
                else:
                    for c, sv in zip(clusters, svals):
                        c["share"] = sv / tot

        # emit per-cluster scheme rows
        for c in clusters:
            p_mod = float(c["p_module_mw"])
            share = float(c.get("share", 1.0))
            label = str(c.get("label", str(int(round(p_mod))))).strip() or str(int(round(p_mod)))

            p_max_cluster = p_max_group * share
            n_max = int(math.ceil(p_max_cluster / p_mod)) if p_mod > 0 else 0
            if n_max_cap is not None and n_max > n_max_cap:
                print(f"[WARN] n_max for group '{g}' cluster '{label}' = {n_max} exceeds cap {n_max_cap}. Capping.")
                n_max = n_max_cap

            var_name = f"{g}_{label}-mod"

            scheme_rows.append(
                {
                    "sobol_idx": sobol_idx,
                    "var": var_name,
                    "kind": "int_mod",
                    "group": g,
                    "unit": "count",
                    "min": 0,
                    "max": n_max,
                    "p_module_mw": p_mod,
                    "p_total_opt_mw": p_opt,
                    "p_total_max_mw": p_max_group,
                    "p_total_max_cluster_mw": p_max_cluster,
                    "share": share,
                    "n_max": n_max,
                    "prob_trip_per_min": prob_trip,
                    "std_dev_rel": "",
                }
            )
            sobol_idx += 1

            caps_rows.append(
                {
                    "group": g,
                    "cluster_label": label,
                    "share": share,
                    "p_total_opt_mw": p_opt,
                    "margin": margin,
                    "p_total_max_group_mw": p_max_group,
                    "p_total_max_cluster_mw": p_max_cluster,
                    "p_module_mw": p_mod,
                    "n_max": n_max,
                    "prob_trip_per_min": prob_trip,
                }
            )

    # float rows
    for v in float_vars:
        vmax = float(float_max.get(v, 0.0))
        vstd = float(std_by_var.get(v, default_std))

        scheme_rows.append(
            {
                "sobol_idx": sobol_idx,
                "var": v,
                "kind": "float",
                "group": "float",
                "unit": "MW",
                "min": 0.0,
                "max": vmax,
                "p_module_mw": "",
                "p_total_opt_mw": "",
                "p_total_max_mw": "",
                "p_total_max_cluster_mw": "",
                "share": "",
                "n_max": "",
                "prob_trip_per_min": "",
                "std_dev_rel": vstd,
            }
        )
        sobol_idx += 1

    caps_df = pd.DataFrame(caps_rows).sort_values(["group", "cluster_label"]).reset_index(drop=True)
    scheme_df = pd.DataFrame(scheme_rows).sort_values(["sobol_idx"]).reset_index(drop=True)

    caps_out = outdir / "reserve_caps.csv"
    scheme_out = outdir / "reserve_scheme.csv"
    caps_df.to_csv(caps_out, index=False)
    scheme_df.to_csv(scheme_out, index=False)

    print(f"[OK] Wrote caps:   {caps_out}")
    print(f"[OK] Wrote scheme: {scheme_out}")

    # Small helper: float maxima
    maxima_rows = [{"var": k, "max": v} for k, v in float_max.items()]
    pd.DataFrame(maxima_rows).sort_values("var").to_csv(outdir / "reserve_float_maxima.csv", index=False)
    print(f"[OK] Wrote float maxima: {outdir / 'reserve_float_maxima.csv'}")

    print("\n>>> DONE.")
    print("Next: read reserve_scheme.csv to build Sobol ranges and generate Sobol states.")


if __name__ == "__main__":
    main()

# python scripts/reserves/build_reserve_preprocessing.py \
#   --network /home/pampado/reserves/pypsa-eur/results/basecase/europe/networks/base_s_adm_elec_.nc \
#   --outdir  /home/pampado/reserves/pypsa-eur/results/basecase/europe/reserves_preprocess \
#   --config  /home/pampado/reserves/pypsa-eur/config/reserves_preprocess.yaml \
#  --module-sizes /home/pampado/reserves/pypsa-eur/data/reserves/module_sizes.csv