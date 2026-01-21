# SPDX-FileCopyrightText: Contributors to PyPSA-Eur
# SPDX-License-Identifier: MIT
"""
Reserve modelling add-on for PyPSA-Eur.

This module adds reserve variables and constraints to the PyPSA (linopy) model
via the `extra_functionality(n, snapshots)` hook used in solve_network.py.

Equation numbering follows the user's equation list PDF.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

import numpy as np
import pandas as pd

from pypsa.descriptors import get_switchable_as_dense as get_as_dense

logger = logging.getLogger(__name__)

Direction = Literal["up", "down"]


# -----------------------------
# IO helpers
# -----------------------------

def load_contribution_table(path: str | Path) -> pd.DataFrame:
    """
    Load per-asset contribution coefficients (ctrb:*).

    Expected columns:
      - carrier OR asset identification columns (project-specific)
      - ctrb:FCR, ctrb:aFRR, ctrb:mFRR, ctrb:RR
      - optional: Type_of_component, c-rate_min/max, etc.
    """
    df = pd.read_csv(path, index_col=0)
    return df


def load_coeff_matrices(path: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Load coefficient matrices for reserve requirement model.

    Expected structure:
      - if pickle: dict[str -> DataFrame] with keys in {"FCR","aFRR","mFRR","RR"}
      - if csv: single csv with multiindex columns or one-file-per-ToR (not implemented here)

    NOTE: This keeps the interface compatible with your original code:
      data_coeff_up["FCR"] -> DataFrame indexed by snapshots
    """
    path = Path(path)
    if path.suffix == ".pkl" or path.suffix == ".pickle":
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported coeff format: {path}")


# -----------------------------
# Asset selection helpers
# -----------------------------

def _filter_by_bus_tokens(index: pd.Index, bus_tokens: List[str]) -> pd.Index:
    """Filter component names by zone tokens (string contains)."""
    if not bus_tokens:
        return index
    mask = [any(tok in name for tok in bus_tokens) for name in index]
    return index[mask]


def select_vres_generators(n, cfg: dict) -> pd.Index:
    """Select vRES generators by carrier list (recommended) instead of name parsing."""
    carriers = cfg.get("vres_carriers", [])
    if not carriers:
        return pd.Index([])
    gens = n.generators.index[n.generators.carrier.isin(carriers)]
    return gens


def select_dispatchable_generators(n, cfg: dict, vres_i: pd.Index) -> pd.Index:
    """
    Select dispatchable generators that are eligible to provide reserves.
    Suggested robust approach: use generator.carrier in cfg list.
    """
    carriers = cfg.get("dispatchable_carriers", [])
    if not carriers:
        return pd.Index([])
    gens = n.generators.index[n.generators.carrier.isin(carriers)]
    gens = gens.difference(vres_i)
    return gens


def select_stores(n, cfg: dict) -> pd.Index:
    """Select stores eligible to provide reserves by carrier."""
    carriers = cfg.get("store_carriers", [])
    if not carriers:
        return pd.Index([])
    stores = n.stores.index[n.stores.carrier.isin(carriers)]
    return stores


def select_store_links(n, stores_i: pd.Index, cfg: dict) -> Tuple[pd.Index, pd.Index]:
    """
    Return (charge_links, discharge_links) for the selected stores.

    This version uses link.carrier categories from config and bus matching, not string splitting.
    """
    link_carriers = cfg.get("store_link_carriers", {})
    ch_carriers = set(link_carriers.get("charge", []))
    dsh_carriers = set(link_carriers.get("discharge", []))

    if not ch_carriers and not dsh_carriers:
        return pd.Index([]), pd.Index([])

    store_buses = set(n.stores.loc[stores_i, "bus"].values)

    # discharge links: bus0 is store bus (power out of store bus)
    dsh = n.links.index[n.links.bus0.isin(store_buses) & n.links.carrier.isin(dsh_carriers)]
    # charge links: bus1 is store bus (power into store bus)
    ch = n.links.index[n.links.bus1.isin(store_buses) & n.links.carrier.isin(ch_carriers)]

    return ch, dsh


# -----------------------------
# Variable builders
# -----------------------------

def add_vrd_variable(n, snapshots):
    """Add VRD variable (Eq. 38 proxy) indexed by snapshots."""
    # Keep VRD as 1D over snapshots to avoid the weird Load-dimension replication.
    n.model.add_variables(lower=0, upper=np.inf, coords=(snapshots,), name="VRD")


def add_reserve_variables(
    n,
    snapshots,
    direction: Direction,
    tor: List[str],
    dgen_i: pd.Index,
    stores_i: pd.Index,
    vres_i: pd.Index,
    cfg: dict,
):
    """
    Create reserve variables for generators and stores.

    Variables are created with names consistent with your original convention:
      - r_gen, r_gen_fcr, ...
      - q_store, q_store_fcr, ...
      - optional q_vres_* for downward vRES
    """
    prefix = "r_" if direction == "up" else "q_"

    # Conservative numeric upper bounds to help MIP numerics
    tol = float(cfg.get("bounds_tolerance", 1.1))

    # Generators
    if len(dgen_i) > 0:
        max_p = n.generators.p_nom_max.reindex(dgen_i).max()
        ub_gen = float(np.round(tol * max_p))

        n.model.add_variables(lower=0, upper=ub_gen, coords=(snapshots, dgen_i), name=prefix + "gen")
        for k in tor:
            n.model.add_variables(lower=0, upper=ub_gen, coords=(snapshots, dgen_i), name=prefix + f"gen_{k.lower()}")

    # Stores
    if len(stores_i) > 0:
        max_e = n.stores.e_nom_max.reindex(stores_i).max()
        max_link = n.links.p_nom_max.max() if hasattr(n.links, "p_nom_max") else max_e
        ub_store = float(np.round(tol * max(max_e, max_link)))

        n.model.add_variables(lower=0, upper=ub_store, coords=(snapshots, stores_i), name=prefix + "store")
        for k in tor:
            n.model.add_variables(lower=0, upper=ub_store, coords=(snapshots, stores_i), name=prefix + f"store_{k.lower()}")

    # vRES (downward only)
    if direction == "down" and cfg.get("vres_downward_enabled", False) and len(vres_i) > 0:
        max_p = n.generators.p_nom_max.reindex(vres_i).max()
        ub_vres = float(np.round(tol * max_p))

        n.model.add_variables(lower=0, upper=ub_vres, coords=(snapshots, vres_i), name=prefix + "vres")
        for k in tor:
            n.model.add_variables(lower=0, upper=ub_vres, coords=(snapshots, vres_i), name=prefix + f"vres_{k.lower()}")


# -----------------------------
# Eq. (38)-(39): VRD definition
# -----------------------------

def add_vrd_constraints(n, snapshots, vres_i: pd.Index):
    """
    Eq. (38)-(39) proxy implementation: 2*VRD_t >= |RD_{t+1}-RD_t| + |RD_t-RD_{t-1}|

    RD_t is approximated as:
      D_t - sum_vres(p_max_pu * p_nom_or_fixed)

    NOTE: Here we keep a simplified consistent version (cleaner than the original),
    still linear and compatible with expansion decisions.
    """
    VRD = n.model["VRD"]

    # total demand (system)
    demand = n.loads_t.p_set.sum(axis=1).reindex(snapshots)

    # shifts with cyclic boundaries (same spirit as your original)
    D_p1 = demand.shift(-1).fillna(demand.iloc[0])
    D_m1 = demand.shift(1).fillna(demand.iloc[-1])

    # vRES available (decision-dependent if extendable)
    RES_t = 0
    if len(vres_i) > 0:
        p_max_pu = get_as_dense(n, "Generator", "p_max_pu").reindex(index=snapshots, columns=vres_i)

        ext_i = n.generators.query("p_nom_extendable").index.intersection(vres_i)
        fix_i = vres_i.difference(ext_i)

        if len(ext_i) > 0:
            p_nom = n.model["Generator-p_nom"].loc[ext_i].rename({"Generator-ext": "Generator"})
            RES_t = RES_t + (p_max_pu[ext_i] * p_nom).sum(axis=1)

        if len(fix_i) > 0:
            RES_t = RES_t + (p_max_pu[fix_i] * n.generators.p_nom.loc[fix_i]).sum(axis=1)

    RD_t = demand - RES_t
    RD_p1 = RD_t.shift(-1).fillna(RD_t.iloc[0])
    RD_m1 = RD_t.shift(1).fillna(RD_t.iloc[-1])

    a = RD_p1 - RD_t
    b = RD_t - RD_m1

    # Linearization: 2*VRD >= ±a ± b (4 constraints)
    lhs = 2 * VRD

    n.model.add_constraints(lhs, ">=",  a + b, "VRD_def_1")
    n.model.add_constraints(lhs, ">=",  a - b, "VRD_def_2")
    n.model.add_constraints(lhs, ">=", -a + b, "VRD_def_3")
    n.model.add_constraints(lhs, ">=", -a - b, "VRD_def_4")


# -----------------------------
# Eq. (35)-(37): reserve composition (+ RR rolling) and store energy composition
# -----------------------------

def add_reserve_composition(n, snapshots, direction: Direction, tor: List[str], t_rr: int, vres_i: pd.Index, cfg: dict):
    """
    Eq. (35)-(37): link total reserve to ToR components, with RR rolling effect.
    """
    prefix = "r_" if direction == "up" else "q_"
    name = "up" if direction == "up" else "dw"

    # Generator: r_gen >= sum(ToR) + RR rolling
    if (prefix + "gen") in n.model.variables:
        r_tot = n.model[prefix + "gen"]
        lhs = n.model.linexpr((1, r_tot))

        for k in tor:
            lhs += n.model.linexpr((-1, n.model[prefix + f"gen_{k.lower()}"]))

        # RR rolling: include past RR contributions
        if "RR" in tor and t_rr > 1:
            rr = n.model[prefix + "gen_rr"]
            for shift in range(1, t_rr):
                lhs += n.model.linexpr((-1, rr.roll(snapshot=shift)))

        n.model.add_constraints(lhs, ">=", 0, f"Reserve_composition_{name}-gen")

    # Store: r_store >= sum(ToR) + RR rolling
    if (prefix + "store") in n.model.variables:
        r_tot = n.model[prefix + "store"]
        lhs = n.model.linexpr((1, r_tot))
        for k in tor:
            lhs += n.model.linexpr((-1, n.model[prefix + f"store_{k.lower()}"]))

        if "RR" in tor and t_rr > 1:
            rr = n.model[prefix + "store_rr"]
            for shift in range(1, t_rr):
                lhs += n.model.linexpr((-1, rr.roll(snapshot=shift)))

        n.model.add_constraints(lhs, ">=", 0, f"Reserve_composition_{name}-store")

    # vRES (downward only)
    if direction == "down" and cfg.get("vres_downward_enabled", False) and len(vres_i) > 0 and (prefix + "vres") in n.model.variables:
        r_tot = n.model[prefix + "vres"]
        lhs = n.model.linexpr((1, r_tot))
        for k in tor:
            lhs += n.model.linexpr((-1, n.model[prefix + f"vres_{k.lower()}"]))

        if "RR" in tor and t_rr > 1:
            rr = n.model[prefix + "vres_rr"]
            for shift in range(1, t_rr):
                lhs += n.model.linexpr((-1, rr.roll(snapshot=shift)))

        n.model.add_constraints(lhs, ">=", 0, f"Reserve_composition_{name}-vres")

# -----------------------------
# Eq. (29)-(30): store power feasibility constraints
# -----------------------------

def add_store_power_feasibility(
    n,
    snapshots,
    direction: Direction,
    stores_i: pd.Index,
    ch_links: pd.Index,
    dsh_links: pd.Index,
):
    """
    Eq. (29)-(30) style: instantaneous power feasibility for store reserves.

    This constraint limits the TOTAL store reserve variable (r_store or q_store)
    by converter headroom given the current dispatch (charge/discharge).

    It complements the energy feasibility constraint (SoC-based).

    Assumptions:
      - stores are modelled as Store + (charge Link) + (discharge Link)
      - charge link has bus1 = store bus
      - discharge link has bus0 = store bus
      - power variables Link-p are positive in the direction bus0 -> bus1 (PyPSA convention)

    Upward (r_):
      r_store(t,s) <= eff_dsh * (p_max_pu_dsh(t)*Pnom_dsh(s) - p_dsh(t,s)) + p_ch(t,s)

    Downward (q_):
      q_store(t,s) <= p_max_pu_ch(t)*Pnom_ch(s) - p_ch(t,s) + eff_dsh * p_dsh(t,s)

    NOTE:
      - This matches your original limit_sup_reserve_store_p logic and signs.
      - p_ch enters with + for upward (you can stop charging and discharge more).
      - p_ch enters with - for downward (if already charging, less headroom to charge more).
    """
    prefix = "r_" if direction == "up" else "q_"
    name = "up" if direction == "up" else "dw"

    if len(stores_i) == 0 or (prefix + "store") not in n.model.variables:
        return

    reserve_store = n.model[prefix + "store"].loc[snapshots, stores_i]

    # Link dispatch variables
    p_link = n.model["Link-p"].loc[snapshots, :]

    # Extendable link nominal power (variable) exists only for p_nom_extendable links
    extendable_links = n.links.query("p_nom_extendable").index
    has_link_pnom_var = "Link-p_nom" in n.model.variables

    # We build constraints store-by-store (safer mapping than global groupby by bus)
    for s in stores_i:
        bus_s = n.stores.at[s, "bus"]

        # Discharge links connected to this store bus (bus0 = store bus)
        dsh_s = dsh_links[n.links.loc[dsh_links, "bus0"] == bus_s]
        # Charge links connected to this store bus (bus1 = store bus)
        ch_s = ch_links[n.links.loc[ch_links, "bus1"] == bus_s]

        # If the storage has no links, skip (misconfigured network)
        if len(dsh_s) == 0 and len(ch_s) == 0:
            continue

        # Aggregate discharge and charge dispatch for this store
        p_dsh = 0
        if len(dsh_s) > 0:
            p_dsh = p_link.loc[:, dsh_s].sum(dim="Link")

        p_ch = 0
        if len(ch_s) > 0:
            p_ch = p_link.loc[:, ch_s].sum(dim="Link")

        # Efficiencies (fallback to 1.0 if missing)
        # In the original code, eff_dsh was taken from the discharge link also in downward constraint.
        eff_dsh = 1.0
        if len(dsh_s) > 0 and "efficiency" in n.links.columns:
            eff_dsh = float(n.links.loc[dsh_s, "efficiency"].iloc[0])

        # Time-varying p_max_pu for links (fallback to 1.0)
        # Use discharge p_max_pu for upward, charge p_max_pu for downward
        p_max_pu = pd.Series(1.0, index=snapshots)

        if direction == "up" and len(dsh_s) > 0:
            p_max_pu_df = get_as_dense(n, "Link", "p_max_pu").reindex(index=snapshots, columns=dsh_s)
            # conservative choice if multiple links: use min across them
            p_max_pu = p_max_pu_df.min(axis=1).fillna(1.0)

        if direction == "down" and len(ch_s) > 0:
            p_max_pu_df = get_as_dense(n, "Link", "p_max_pu").reindex(index=snapshots, columns=ch_s)
            p_max_pu = p_max_pu_df.min(axis=1).fillna(1.0)

        # Nominal power Pnom for discharge/charge side:
        # - variable part for extendable links
        # - constant part for fixed links
        def _sum_link_pnom(links: pd.Index):
            if len(links) == 0:
                return 0

            ext = links.intersection(extendable_links)
            fix = links.difference(ext)

            expr = 0
            if len(fix) > 0:
                expr = expr + float(n.links.loc[fix, "p_nom"].sum())

            if len(ext) > 0:
                if not has_link_pnom_var:
                    raise KeyError("Link-p_nom variable not found but extendable links exist.")
                expr = expr + n.model["Link-p_nom"].loc[ext].sum(dim="Link-ext")

            return expr

        Pnom_dsh = _sum_link_pnom(dsh_s)
        Pnom_ch = _sum_link_pnom(ch_s)

        # Build RHS according to direction
        if direction == "up":
            # r_store <= eff_dsh*(p_max_pu*Pnom_dsh - p_dsh) + p_ch
            rhs = eff_dsh * (p_max_pu * Pnom_dsh - p_dsh) + p_ch
            n.model.add_constraints(
                reserve_store.loc[:, s],
                "<=",
                rhs,
                f"Store_power_feasibility_{name}_{s}",
            )

        else:
            # q_store <= p_max_pu*Pnom_ch - p_ch + eff_dsh*p_dsh
            rhs = p_max_pu * Pnom_ch - p_ch + eff_dsh * p_dsh
            n.model.add_constraints(
                reserve_store.loc[:, s],
                "<=",
                rhs,
                f"Store_power_feasibility_{name}_{s}",
            )

# -----------------------------
# Eq. (31)-(32)/(36)-(37): store energy feasibility constraints
# -----------------------------

def add_store_energy_feasibility(n, snapshots, direction: Direction, tor: List[str], t_rr: int, stores_i: pd.Index, ch_links: pd.Index, dsh_links: pd.Index, cfg: dict):
    """
    Eq. (31)-(32) / Eq. (36)-(37) style: store reserve limited by SoC margins.
    Uses ToR sum + RR rolling on the LHS.

    This is the "energy" constraint for storage (multi-hour feasibility).
    """
    prefix = "r_" if direction == "up" else "q_"
    name = "up" if direction == "up" else "dw"

    if len(stores_i) == 0:
        return

    # ToR sum (store)
    lhs = n.model.linexpr((1, n.model[prefix + "store_fcr"]))
    for k in tor:
        if k.lower() == "fcr":
            continue
        lhs += n.model.linexpr((1, n.model[prefix + f"store_{k.lower()}"]))

    # RR rolling weighted (k+1)
    if "RR" in tor and t_rr > 1:
        rr = n.model[prefix + "store_rr"]
        for shift in range(1, t_rr):
            lhs += n.model.linexpr((shift + 1, rr.roll(snapshot=shift)))

    # State of charge
    e = n.model["Store-e"].loc[snapshots, stores_i]
    e_nom = n.model["Store-e_nom"].loc[stores_i].rename({"Store-ext": "Store"})
    e_min_pu = get_as_dense(n, "Store", "e_min_pu").reindex(index=snapshots, columns=stores_i)
    e_max_pu = get_as_dense(n, "Store", "e_max_pu").reindex(index=snapshots, columns=stores_i)

    # Link powers (charge/discharge) aggregated per store bus
    # NOTE: This assumes one store per bus/carrier; can be generalized later.
    p_link = n.model["Link-p"].loc[snapshots, :]

    # Map store -> its links by bus match
    store_bus = n.stores.loc[stores_i, "bus"]

    # Build charge/discharge power per store (fallback 0 if missing)
    p_ch = 0
    p_dsh = 0
    if len(ch_links) > 0:
        tmp = p_link.loc[:, ch_links]
        tmp = tmp.assign_coords({"Link": n.links.loc[ch_links, "bus1"].values})
        p_ch = tmp.groupby("Link").sum().rename({"Link": "Store"}).reindex(Store=store_bus.values)
    if len(dsh_links) > 0:
        tmp = p_link.loc[:, dsh_links]
        tmp = tmp.assign_coords({"Link": n.links.loc[dsh_links, "bus0"].values})
        p_dsh = tmp.groupby("Link").sum().rename({"Link": "Store"}).reindex(Store=store_bus.values)

    # Energy feasibility constraint (direction-dependent)
    count_inv_e = bool(cfg.get("count_inversion_energy", True))

    if direction == "up":
        # r_u <= eta_dsh * (e - e_min*e_nom) - (p_ch / eta_ch) [optional]
        # Use link efficiencies if available, else eta=1.
        eta_dsh = 1.0
        eta_ch = 1.0
        # NOTE: you can refine per-link/per-store efficiencies if needed.

        rhs = eta_dsh * (e - e_min_pu * e_nom)
        if count_inv_e:
            rhs = rhs - (p_ch / eta_ch)

        n.model.add_constraints(lhs, "<=", rhs, f"Reserve_store_energy_{name}")

    else:
        # q_d <= (e_max*e_nom - e) - (p_dsh/eta_ch) [optional]
        eta_ch = 1.0
        rhs = (e_max_pu * e_nom - e)
        if count_inv_e:
            rhs = rhs - (p_dsh / eta_ch)

        n.model.add_constraints(lhs, "<=", rhs, f"Reserve_store_energy_{name}")


# -----------------------------
# Eq. (23)-(28): generator capacity and contribution constraints
# -----------------------------

def add_generator_capacity_constraints(n, snapshots, direction: Direction, tor: List[str], dgen_i: pd.Index):
    """
    Eq. (23)-(24) + fast committable constraints (26)-(27).
    This is the headroom coupling between dispatch and reserve.
    """
    prefix = "r_" if direction == "up" else "q_"
    name = "up" if direction == "up" else "dw"

    if len(dgen_i) == 0 or (prefix + "gen") not in n.model.variables:
        return

    dispatch = n.model["Generator-p"].loc[snapshots, dgen_i]
    p_max_pu = get_as_dense(n, "Generator", "p_max_pu").reindex(index=snapshots, columns=dgen_i)
    p_min_pu = get_as_dense(n, "Generator", "p_min_pu").reindex(index=snapshots, columns=dgen_i)

    com_i = n.get_committable_i("Generator").intersection(dgen_i)
    ext_i = n.generators.query("p_nom_extendable").index.intersection(dgen_i)
    fix_i = dgen_i.difference(ext_i)

    # Non-committable (simpler)
    if direction == "up":
        # r_total + p <= p_max_pu * p_nom
        if len(ext_i) > 0:
            p_nom = n.model["Generator-p_nom"].loc[ext_i].rename({"Generator-ext": "Generator"})
            lhs = n.model.linexpr((1, n.model[prefix + "gen"].loc[snapshots, ext_i]), (1, dispatch.loc[:, ext_i]))
            rhs = p_max_pu[ext_i] * p_nom
            n.model.add_constraints(lhs, "<=", rhs, f"Gen_headroom_ext_{name}")

        if len(fix_i) > 0:
            lhs = n.model.linexpr((1, n.model[prefix + "gen"].loc[snapshots, fix_i]), (1, dispatch.loc[:, fix_i]))
            rhs = p_max_pu[fix_i] * n.generators.p_nom.loc[fix_i]
            n.model.add_constraints(lhs, "<=", rhs, f"Gen_headroom_fix_{name}")

    else:
        # q_total <= p  (downward reserve cannot exceed current dispatch)
        lhs = n.model.linexpr((1, n.model[prefix + "gen"].loc[snapshots, dgen_i]))
        rhs = dispatch
        n.model.add_constraints(lhs, "<=", rhs, f"Gen_downward_limit_{name}")

    # Committable fast reserve constraint: FCR + aFRR bounded by status & headroom (26)-(27)
    if len(com_i) > 0 and ("FCR" in tor and "aFRR" in tor):
        status = n.model["Generator-status"].loc[snapshots, com_i]
        p_mod = n.generators.p_nom_mod.loc[com_i]

        r_fcr = n.model[prefix + "gen_fcr"].loc[snapshots, com_i]
        r_afrr = n.model[prefix + "gen_afrr"].loc[snapshots, com_i]

        if direction == "up":
            lhs = n.model.linexpr((1, r_fcr), (1, r_afrr), (1, dispatch.loc[:, com_i]))
            rhs = (p_mod * status) * p_max_pu[com_i]
            n.model.add_constraints(lhs, "<=", rhs, f"Gen_fast_com_{name}")
        else:
            lhs = n.model.linexpr((1, r_fcr), (1, r_afrr), (-1, dispatch.loc[:, com_i]))
            rhs = -(p_mod * status) * p_min_pu[com_i]
            n.model.add_constraints(lhs, "<=", rhs, f"Gen_fast_com_{name}")


def add_contribution_bounds(n, snapshots, direction: Direction, tor: List[str], dgen_i: pd.Index, stores_i: pd.Index, contrib: pd.DataFrame):
    """
    Eq. (25) + Eq. (33)-(34): reserve contribution upper bounds via mu/ctrb coefficients.
    """
    prefix = "r_" if direction == "up" else "q_"
    name = "up" if direction == "up" else "dw"

    # Generators: r_ToR <= mu_ToR * P_nom
    if len(dgen_i) > 0:
        ext_i = n.generators.query("p_nom_extendable").index.intersection(dgen_i)
        fix_i = dgen_i.difference(ext_i)

        for k in tor:
            mu_col = f"ctrb:{k}"
            if mu_col not in contrib.columns:
                continue

            r_k = n.model[prefix + f"gen_{k.lower()}"].loc[snapshots, dgen_i]

            # Extendable: bound by decision capacity
            if len(ext_i) > 0:
                mu = contrib.loc[n.generators.loc[ext_i, "carrier"], mu_col].values
                mu = pd.Series(mu, index=ext_i)
                p_nom = n.model["Generator-p_nom"].loc[ext_i].rename({"Generator-ext": "Generator"})
                lhs = n.model.linexpr((1, r_k.loc[:, ext_i]), (-mu, p_nom))
                n.model.add_constraints(lhs, "<=", 0, f"Gen_mu_{k}_{name}_ext")

            # Fixed: bound by installed capacity
            if len(fix_i) > 0:
                mu = contrib.loc[n.generators.loc[fix_i, "carrier"], mu_col].values
                mu = pd.Series(mu, index=fix_i)
                rhs = mu * n.generators.p_nom.loc[fix_i]
                n.model.add_constraints(r_k.loc[:, fix_i], "<=", rhs, f"Gen_mu_{k}_{name}_fix")

    # Stores: r_ToR <= mu_ToR * E_nom (and optionally power converter bounds elsewhere)
    if len(stores_i) > 0:
        for k in tor:
            mu_col = f"ctrb:{k}"
            if mu_col not in contrib.columns:
                continue

            r_k = n.model[prefix + f"store_{k.lower()}"].loc[snapshots, stores_i]
            mu = contrib.loc[n.stores.loc[stores_i, "carrier"], mu_col].values
            mu = pd.Series(mu, index=stores_i)

            e_nom = n.model["Store-e_nom"].loc[stores_i].rename({"Store-ext": "Store"})
            lhs = n.model.linexpr((1, r_k), (-mu, e_nom))
            n.model.add_constraints(lhs, "<=", 0, f"Store_mu_{k}_{name}")


# -----------------------------
# Eq. (22): system reserve requirement (operational reserve margin)
# -----------------------------

def add_system_reserve_requirement(
    n,
    snapshots,
    direction: Direction,
    tor: List[str],
    dgen_i: pd.Index,
    stores_i: pd.Index,
    vres_i: pd.Index,
    coeff_matrix_by_tor: Dict[str, pd.DataFrame],
):
    """
    Eq. (22): sum of reserve contributions >= requirement R^ToR(x_t).

    This is where your "coeff_matrix" regression-like requirement enters.
    """
    prefix = "r_" if direction == "up" else "q_"
    name = "up" if direction == "up" else "dw"

    demand = n.loads_t.p_set.sum(axis=1).reindex(snapshots)
    VRD = n.model["VRD"].reindex(snapshot=snapshots)

    for k in tor:
        if k not in coeff_matrix_by_tor:
            continue

        coeff = coeff_matrix_by_tor[k].reindex(index=snapshots)

        # LHS: offered reserve (gens + stores + optional vRES)
        lhs = 0
        if len(dgen_i) > 0:
            lhs = lhs + n.model[prefix + f"gen_{k.lower()}"].loc[snapshots, dgen_i].sum(dim="Generator")
        if len(stores_i) > 0:
            lhs = lhs + n.model[prefix + f"store_{k.lower()}"].loc[snapshots, stores_i].sum(dim="Store")

        if direction == "down" and (prefix + f"vres_{k.lower()}") in n.model.variables and len(vres_i) > 0:
            lhs = lhs + n.model[prefix + f"vres_{k.lower()}"].loc[snapshots, vres_i].sum(dim="Generator")

        # RHS: requirement model
        # Keep this consistent with your earlier expression:
        #   RHS = coeff["Load"]*Demand + coeff["Ramp"]*VRD + coeff["Reserve-const"]
        rhs = (
            coeff["Load"] * demand
            + coeff["Ramp"] * VRD
            + coeff["Reserve-const"]
        )

        n.model.add_constraints(lhs, ">=", rhs, f"Reserve_requirement_{k}_{name}")


# -----------------------------
# Public entry point
# -----------------------------

def apply_reserves(n, snapshots, cfg: dict, reserve_inputs: dict):
    """
    Main entry point to be called from solve_network.py via extra_functionality.

    reserve_inputs expects:
      - contribution_table: path
      - coeff_up: path
      - coeff_dw: path
    """
    if not cfg.get("enabled", False):
        return

    tor = cfg.get("tor", ["FCR", "aFRR", "mFRR", "RR"])
    directions: List[Direction] = cfg.get("directions", ["up", "down"])
    t_rr = int(cfg.get("t_rr", 2))

    contrib = load_contribution_table(reserve_inputs["contribution_table"])
    coeff_up = load_coeff_matrices(reserve_inputs["coeff_up"])
    coeff_dw = load_coeff_matrices(reserve_inputs["coeff_dw"])

    # Asset selection
    vres_i = select_vres_generators(n, cfg)
    dgen_i = select_dispatchable_generators(n, cfg, vres_i)
    stores_i = select_stores(n, cfg)
    ch_links, dsh_links = select_store_links(n, stores_i, cfg)

    # Eq. (38)-(39): VRD variable + constraints
    add_vrd_variable(n, snapshots)
    add_vrd_constraints(n, snapshots, vres_i)

    # Loop directions: up/down
    for direction in directions:
        add_reserve_variables(
            n=n, snapshots=snapshots, direction=direction, tor=tor,
            dgen_i=dgen_i, stores_i=stores_i, vres_i=vres_i, cfg=cfg
        )

        # Eq. (35)-(37): composition
        add_reserve_composition(n, snapshots, direction, tor, t_rr, vres_i, cfg)

        # Eq. (23)-(28): generator headroom & committable fast
        add_generator_capacity_constraints(n, snapshots, direction, tor, dgen_i)

        # Eq. (25) + (33)-(34): contribution bounds
        add_contribution_bounds(n, snapshots, direction, tor, dgen_i, stores_i, contrib)

        # Eq. (29)-(30): store power feasibility
        add_store_power_feasibility(n, snapshots, direction, stores_i, ch_links, dsh_links)

        # Eq. (31)-(32) / (36)-(37): store energy feasibility
        add_store_energy_feasibility(n, snapshots, direction, tor, t_rr, stores_i, ch_links, dsh_links, cfg)

        # Eq. (22): system requirement model using coeff matrices
        coeff_by_tor = coeff_up if direction == "up" else coeff_dw
        add_system_reserve_requirement(n, snapshots, direction, tor, dgen_i, stores_i, vres_i, coeff_by_tor)

    logger.info(
        "Reserves added: ToR=%s, directions=%s, gens=%d, stores=%d, vRES=%d",
        tor, directions, len(dgen_i), len(stores_i), len(vres_i)
    )
