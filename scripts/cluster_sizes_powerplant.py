#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster typical powerplant sizes by Fueltype (and Set/component) from powerplant.csv.

Inputs:
- powerplant.csv with columns at least: Fueltype, Set, Capacity

Outputs:
- Excel:
  - summary_top3: top-3 most frequent size clusters per (Set, Fueltype)
  - clusters_detail: all clusters per (Set, Fueltype)
- Plots:
  - bar charts of cluster counts per (Set, Fueltype)

Clustering:
- 1D greedy clustering on sorted sizes.
- A new size x joins the current cluster if:
    |x - center| <= max(abs_tol, rel_tol * center)
- center can be median or mean.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Clustering logic (1D)
# ----------------------------

def cluster_1d_sizes(
    sizes: np.ndarray,
    abs_tol: float = 50.0,
    rel_tol: float = 0.20,
    center: str = "median",
    max_clusters: int | None = None,
) -> List[Dict]:
    """
    Threshold-based 1D clustering + optional merging to enforce <= max_clusters.

    Step 1: greedy clustering on sorted sizes using tolerance rule:
        |x - c| <= max(abs_tol, rel_tol * c)

    Step 2 (optional): if too many clusters, iteratively merge the closest
    adjacent clusters (by center distance) until <= max_clusters.
    """
    sizes = np.asarray(sizes, dtype=float)
    sizes = sizes[np.isfinite(sizes)]
    sizes = sizes[sizes > 0]
    if sizes.size == 0:
        return []

    sizes_sorted = np.sort(sizes)

    def center_of(arr: np.ndarray) -> float:
        if center == "mean":
            return float(np.mean(arr))
        return float(np.median(arr))

    # ---- Step 1: greedy threshold clustering
    clusters_vals: List[List[float]] = []
    current: List[float] = [float(sizes_sorted[0])]

    def current_center(vals: List[float]) -> float:
        arr = np.asarray(vals, dtype=float)
        return center_of(arr)

    for x in sizes_sorted[1:]:
        c = current_center(current)
        thr = max(abs_tol, rel_tol * c)
        if abs(float(x) - c) <= thr:
            current.append(float(x))
        else:
            clusters_vals.append(current)
            current = [float(x)]
    clusters_vals.append(current)

    # ---- Step 2: merge until <= max_clusters
    if max_clusters is not None and max_clusters > 0:
        # ensure we don't ask for more clusters than available
        while len(clusters_vals) > max_clusters:
            # compute centers for current clusters
            centers = [center_of(np.asarray(v, dtype=float)) for v in clusters_vals]
            # choose closest adjacent pair
            dists = [abs(centers[i + 1] - centers[i]) for i in range(len(centers) - 1)]
            j = int(np.argmin(dists))
            # merge j and j+1
            merged = clusters_vals[j] + clusters_vals[j + 1]
            clusters_vals = clusters_vals[:j] + [merged] + clusters_vals[j + 2:]

    # ---- Build output dicts with std info
    out: List[Dict] = []
    for vals in clusters_vals:
        arr = np.asarray(vals, dtype=float)
        c = center_of(arr)
        out.append(
            {
                "cluster_center": c,
                "count": int(arr.size),
                "min_size": float(arr.min()),
                "max_size": float(arr.max()),
                "mean_size": float(np.mean(arr)),
                "std_size": float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0,
                "members": arr,
            }
        )

    return out



# ----------------------------
# Plotting
# ----------------------------

def save_cluster_barplot(
    df_detail: pd.DataFrame,
    outpath: Path,
    title: str,
    max_bars: int = 25,
    errorbar: str = "std",
) -> None:
    """
    Save a bar plot of cluster counts with optional error bars on the x-label clusters:
    - errorbar='std' uses std_MW from df_detail
    """
    if df_detail.empty:
        return

    center_col = "cluster_center_MW" if "cluster_center_MW" in df_detail.columns else "cluster_center"
    std_col = "std_MW" if "std_MW" in df_detail.columns else None

    d = (
        df_detail.sort_values(["count", center_col], ascending=[False, True])
        .head(max_bars)
        .copy()
    )

    x = np.arange(len(d))
    centers = d[center_col].to_numpy(dtype=float)
    labels = [f"{c:.1f}" for c in centers]

    plt.figure(figsize=(12, 6))
    plt.bar(x, d["count"].to_numpy(), edgecolor="black")

    # Add std as errorbars on a secondary axis? No: keep it simple, show as text in x tick labels.
    # Better: overlay errorbars on counts is misleading, so we annotate std in labels.
    if errorbar == "std" and std_col is not None:
        stds = d[std_col].to_numpy(dtype=float)
        labels = [f"{c:.1f}±{s:.1f}" for c, s in zip(centers, stds)]

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Cluster center size [MW] (±1σ if shown)", fontweight="bold")
    plt.ylabel("Count", fontweight="bold")
    plt.title(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()




# ----------------------------
# Main pipeline
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster typical sizes by Fueltype/Set from powerplant.csv")
    parser.add_argument("--csv", type=str, required=True, help="Path to powerplant.csv")
    parser.add_argument("--outdir", type=str, default="size_clusters_out", help="Output folder")
    parser.add_argument("--abs-tol", type=float, default=50.0, help="Absolute tolerance in MW (e.g., 50)")
    parser.add_argument("--rel-tol", type=float, default=0.20, help="Relative tolerance (e.g., 0.20 for 20%%)")
    parser.add_argument("--center", type=str, default="median", choices=["median", "mean"], help="Cluster center definition")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K clusters by frequency to report in summary")
    parser.add_argument("--max-bars", type=int, default=25, help="Max clusters to show in each plot")
    parser.add_argument("--max-clusters", type=int, default=3, help="Maximum number of clusters per (Set, Fueltype)")
    parser.add_argument("--errorbar", type=str, default="std", choices=["none", "std"], help="Error bars in plots")

    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV (robust to an extra unnamed index column)
    df = pd.read_csv(csv_path)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])

    required = {"Fueltype", "Set", "Capacity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # Clean/standardize
    df["Fueltype"] = df["Fueltype"].astype(str).fillna("unknown")
    df["Set"] = df["Set"].astype(str).fillna("unknown")
    df["Capacity"] = pd.to_numeric(df["Capacity"], errors="coerce")
    df = df.dropna(subset=["Capacity"])
    df = df[df["Capacity"] > 0].copy()

    detail_rows = []
    summary_rows = []

    group_cols = ["Set", "Fueltype"]
    for (set_name, fueltype), g in df.groupby(group_cols):
        sizes = g["Capacity"].to_numpy(dtype=float)

        clusters = cluster_1d_sizes(
            sizes,
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
            center=args.center,
            max_clusters=args.max_clusters,
        )

        # Sort clusters by frequency, then by size
        clusters_sorted = sorted(clusters, key=lambda c: (-c["count"], c["cluster_center"]))

        total_n = int(len(sizes))
        total_cap = float(np.sum(sizes))

        # Build detail table
        for rank, c in enumerate(clusters_sorted, start=1):
            members = c["members"]
            detail_rows.append(
                {
                    "Set": set_name,
                    "Fueltype": fueltype,
                    "cluster_rank_by_count": rank,
                    "cluster_center_MW": c["cluster_center"],
                    "count": c["count"],
                    "count_share": c["count"] / total_n if total_n else np.nan,
                    "min_MW": c["min_size"],
                    "max_MW": c["max_size"],
                    "cluster_total_capacity_MW": float(np.sum(members)),
                    "cluster_capacity_share": float(np.sum(members)) / total_cap if total_cap else np.nan,
                    "mean_MW": c["mean_size"],
                    "std_MW": c["std_size"],
                }
            )

        # Summary top-k
        row = {
            "Set": set_name,
            "Fueltype": fueltype,
            "n_assets": total_n,
            "total_capacity_MW": total_cap,
        }
        for i in range(args.top_k):
            if i < len(clusters_sorted):
                c = clusters_sorted[i]
                row[f"top{i+1}_center_MW"] = c["cluster_center"]
                row[f"top{i+1}_count"] = c["count"]
                row[f"top{i+1}_min_MW"] = c["min_size"]
                row[f"top{i+1}_max_MW"] = c["max_size"]
                row[f"top{i+1}_std_MW"] = c["std_size"]
            else:
                row[f"top{i+1}_center_MW"] = np.nan
                row[f"top{i+1}_count"] = np.nan
                row[f"top{i+1}_min_MW"] = np.nan
                row[f"top{i+1}_max_MW"] = np.nan
                row[f"top{i+1}_std_MW"] = np.nan

        summary_rows.append(row)

    df_detail = pd.DataFrame(detail_rows)
    df_summary = pd.DataFrame(summary_rows)

    # Sort for readability
    df_detail = df_detail.sort_values(["Set", "Fueltype", "cluster_rank_by_count"]).reset_index(drop=True)
    df_summary = df_summary.sort_values(["Set", "Fueltype"]).reset_index(drop=True)

    # Save plots per (Set, Fueltype)
    for (set_name, fueltype), d in df_detail.groupby(["Set", "Fueltype"]):
        safe_set = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(set_name))
        safe_fuel = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(fueltype))
        out_png = plots_dir / f"clusters_{safe_set}__{safe_fuel}.png"
        title = f"Clustered sizes - Set={set_name} | Fueltype={fueltype}"
        save_cluster_barplot(d, out_png, title, max_bars=args.max_bars, errorbar=args.errorbar)


    # Save Excel
    out_xlsx = outdir / "size_clusters_powerplant.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="summary_top3", index=False)
        df_detail.to_excel(writer, sheet_name="clusters_detail", index=False)

    print(f"[OK] Wrote Excel: {out_xlsx}")
    print(f"[OK] Wrote plots in: {plots_dir}")


if __name__ == "__main__":
    main()

# python scripts/cluster_sizes_powerplant.py \
#  --csv resources/basecase/europe/powerplants_s_adm.csv \
#  --outdir results/basecase/europe/out_sizes \
#  --abs-tol 50 --rel-tol 0.20 \
#  --max-clusters 3 \
#  --errorbar std

