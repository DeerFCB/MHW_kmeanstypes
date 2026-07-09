#!/usr/bin/env python
# coding: utf-8
"""
Final plotting script for 00_02 EOF-KMeans classification and early SI figures.

Outputs
-------
Main/Fig01_type_composite_and_onset_stage_evolution.{pdf,png}
Main/Fig02_month_and_characteristics.{pdf,png}
SI/FigS04_EOF_raw_vs_detrended_PCcorr.{pdf,png}
SI/FigS05_cumulative_variance_raw_vs_detrended.{pdf,png}
SI/FigS06_cluster_evaluation_metrics.{pdf,png}
SI/FigS06b_silhouette_profiles_raw_vs_detrended.{pdf,png}
SI/FigS07_full_onset_peak_decay_type_composites.{pdf,png}
SI/FigS08_season_standardized_type_maps.{pdf,png}
SI/FigS09_climate_indices_by_type.{pdf,png}
Tables/TableS1_full_stage_pattern_correlations.csv
Tables/TableS2_standardized_clustering_summary.xlsx

Run from the new code folder, for example:
    cd /Users/luty8/MHW_Project/code/projects/Kmeans_eof_2baseline/Revised_0705
    python scripts/00_02_make_classification_figures.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as tdist

warnings.filterwarnings("ignore")

THIS_FILE = Path(__file__).resolve()
REVISED_DIR = THIS_FILE.parents[1]
sys.path.insert(0, str(REVISED_DIR))

from src.figstyle import (  # noqa: E402
    set_mpl_style,
    save_figure,
    add_panel_label,
    thin_spines,
    apply_grid,
    FIG_W_FULL,
    FIG_W_HALF,
    TYPE_COLORS,
    TYPE_COLORS_LIST,
    CMAP_SST,
    PANEL_LABELS,
)
from src.map_utils import (  # noqa: E402
    normalize_longitude,
    lon_lat_mesh,
    add_gom_base,
    map_gridlines,
    GOM_EXTENT_WIDE,
)

# -----------------------------
# Classification settings
# -----------------------------
K_MAIN = 4
N_MODES_EOF = 10
N_PC_MAIN = 4
RANDOM_STATE = 42
N_INIT = 100
SWAP_MAP_MAIN = {0: 2, 1: 0, 2: 1, 3: 3}
PEAK_HALF_WIDTH = 3
PHASE_STAGES = (0.0, 0.5, 1.0)
PHASE_STAGE_LABELS = {0.0: "Start", 0.5: "50% of onset stage", 1.0: "Peak"}
CORE_CONTOUR_LEVEL = 0.8
STD_BASELINE_START = "1982-01-01"
STD_BASELINE_END = "2011-12-31"
STD_SMOOTH_WINDOW = 31
STD_MIN_SIGMA = 0.05

CLIMATE_INDICES = [
    "NAO", "PNA", "EP/NP", "WP",
    "Niño 1+2", "Niño 3.4", "Niño 4", "ONI", "MEI V2",
    "TNA", "AMM", "CAR",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make final 00_02 classification figures.")
    parser.add_argument("--base", type=Path, default=Path("/Users/luty8/MHW_Project"), help="MHW project root")
    parser.add_argument("--main-dir", type=Path, default=None, help="Main-text figure output directory")
    parser.add_argument("--si-dir", type=Path, default=None, help="SI figure output directory")
    parser.add_argument("--table-dir", type=Path, default=None, help="Table output directory")
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    return parser.parse_args()


# ============================================================
# Data and classification helpers
# ============================================================

def mhw_ds_to_df(mhw_ds: xr.Dataset) -> pd.DataFrame:
    data = {}
    for var in mhw_ds.data_vars:
        da = mhw_ds[var]
        if "events" in da.dims and da.ndim == 1:
            data[var] = da.values
        elif "event" in da.dims and da.ndim == 1:
            data[var] = da.values
    df = pd.DataFrame(data)
    for col in df.columns:
        if "time" in col:
            df[col] = pd.to_datetime(df[col])
    return df


def compute_event_mean_ssta_from_df(
    sst_da: xr.DataArray,
    events_df: pd.DataFrame,
    time_start_col="time_start",
    time_end_col="time_end",
    min_duration=3,
) -> xr.DataArray:
    event_means = []
    event_ids = []
    for idx, row in events_df.iterrows():
        ts = pd.Timestamp(row[time_start_col])
        te = pd.Timestamp(row[time_end_col])
        if pd.isna(ts) or pd.isna(te):
            continue
        if (te - ts).days + 1 < min_duration:
            continue
        da_evt = sst_da.sel(time=slice(ts, te))
        if da_evt.sizes.get("time", 0) == 0:
            continue
        event_means.append(da_evt.mean("time", skipna=True).expand_dims(event=[int(idx)]))
        event_ids.append(int(idx))
    if not event_means:
        raise RuntimeError("No valid event means computed.")
    out = xr.concat(event_means, dim="event")
    out = out.assign_coords(event=np.asarray(event_ids, dtype=int))
    out.name = "sst_event_mean"
    return out


def compute_event_stage_mean_ssta_from_df(
    sst_da: xr.DataArray,
    events_df: pd.DataFrame,
    stage="full",
    peak_half_width=3,
) -> xr.DataArray:
    stage = stage.lower()
    if stage not in ["full", "onset", "peak", "decay"]:
        raise ValueError("stage must be one of full/onset/peak/decay")
    maps = []
    ids = []
    for idx, row in events_df.iterrows():
        t0 = pd.Timestamp(row["time_start"])
        tp = pd.Timestamp(row["time_peak"])
        te = pd.Timestamp(row["time_end"])
        if pd.isna(t0) or pd.isna(tp) or pd.isna(te):
            continue
        if stage == "full":
            start, end = t0, te
        elif stage == "onset":
            start, end = t0, tp
        elif stage == "peak":
            start, end = tp - pd.Timedelta(days=peak_half_width), tp + pd.Timedelta(days=peak_half_width)
        else:
            start, end = tp, te
        if end < start:
            continue
        da = sst_da.sel(time=slice(start, end))
        if da.sizes.get("time", 0) == 0:
            continue
        maps.append(da.mean("time", skipna=True).expand_dims(event=[int(idx)]))
        ids.append(int(idx))
    if not maps:
        raise RuntimeError(f"No valid stage maps for stage={stage}")
    out = xr.concat(maps, dim="event").assign_coords(event=np.asarray(ids, dtype=int))
    out.name = f"sst_event_mean_{stage}"
    return out


def eof_from_event_means(em_da: xr.DataArray, n_modes=10, use_weight=True):
    """EOF/PCA for event-mean maps.

    This intentionally mirrors the original 00_02 notebook logic:
    grid cells that are all-NaN across events are excluded, while occasional
    NaNs inside retained ocean cells are filled with zero before PCA. This keeps
    the EOF variance and spatial patterns consistent with the revision notebook.
    """
    da = em_da.transpose("event", "lat", "lon")
    if use_weight:
        w = np.sqrt(np.cos(np.deg2rad(da["lat"])))
        da_w = da * w
    else:
        da_w = da

    X = da_w.stack(space=("lat", "lon"))  # event x space
    X_values = X.values
    valid = ~np.all(np.isnan(X_values), axis=0)
    X_valid = X[:, valid]
    X_filled = X_valid.fillna(0.0)

    X_mat = X_filled.values
    X_mat_anom = X_mat - X_mat.mean(axis=0, keepdims=True)

    pca = PCA(n_components=n_modes)
    pcs = pca.fit_transform(X_mat_anom)
    eof_valid = pca.components_
    expvar = pca.explained_variance_ratio_

    eof_full = np.full((n_modes, X.sizes["space"]), np.nan, dtype=np.float32)
    eof_full[:, valid] = eof_valid.astype(np.float32)
    eofs_da = xr.DataArray(
        eof_full.reshape((n_modes, da.sizes["lat"], da.sizes["lon"])),
        dims=("mode", "lat", "lon"),
        coords={"mode": np.arange(1, n_modes + 1), "lat": da["lat"], "lon": da["lon"]},
        name="eof",
    )
    pcs_da = xr.DataArray(
        pcs,
        dims=("event", "mode"),
        coords={"event": da["event"], "mode": np.arange(1, n_modes + 1)},
        name="pc",
    )
    return eofs_da, pcs_da, expvar


def compute_cluster_mean(event_mean_da: xr.DataArray, labels, K=4) -> xr.DataArray:
    labels = np.asarray(labels)
    means = []
    for k in range(K):
        da_k = event_mean_da.sel(event=event_mean_da.event[labels == k])
        means.append(da_k.mean("event", skipna=True) if da_k.sizes["event"] else event_mean_da.isel(event=0) * np.nan)
    out = xr.concat(means, dim="cluster").assign_coords(cluster=np.arange(K))
    return out


def run_eof_kmeans_classification(event_mean_da, K=4, n_modes=10, n_pc=4, swap_map=None):
    eofs_da, pcs_da, expvar = eof_from_event_means(event_mean_da, n_modes=n_modes, use_weight=True)
    pc_input = pcs_da.sel(mode=slice(1, n_pc)).values
    km = KMeans(n_clusters=K, n_init=N_INIT, random_state=RANDOM_STATE)
    labels = km.fit_predict(pc_input)
    if swap_map is not None:
        labels = np.array([swap_map[int(i)] for i in labels], dtype=int)
    cluster_mean = compute_cluster_mean(event_mean_da, labels, K=K)
    print("Cluster sizes:", np.bincount(labels, minlength=K))
    print("Explained variance first 4 modes (%):", np.round(expvar[:n_pc] * 100, 2))
    print("Cumulative variance used in K-means (%):", np.round(expvar[:n_pc].sum() * 100, 2))
    return {"eofs": eofs_da, "pcs": pcs_da, "expvar": expvar, "labels": labels, "cluster_mean": cluster_mean}


def weighted_pattern_corr(a: xr.DataArray, b: xr.DataArray) -> float:
    aa = a.values
    bb = b.values
    lat_vals = a["lat"].values
    w2d = np.broadcast_to(np.cos(np.deg2rad(lat_vals))[:, None], aa.shape)
    mask = np.isfinite(aa) & np.isfinite(bb) & np.isfinite(w2d)
    if mask.sum() < 10:
        return np.nan
    x = aa[mask]
    y = bb[mask]
    w = w2d[mask]
    w = w / np.nansum(w)
    xm = np.nansum(w * x)
    ym = np.nansum(w * y)
    xa = x - xm
    ya = y - ym
    denom = np.sqrt(np.nansum(w * xa**2) * np.nansum(w * ya**2))
    return float(np.nansum(w * xa * ya) / denom) if denom else np.nan


def align_labels_to_event_da(labels, event_da: xr.DataArray, reference_event_ids) -> np.ndarray:
    label_series = pd.Series(np.asarray(labels, dtype=int), index=np.asarray(reference_event_ids, dtype=int))
    return label_series.loc[np.asarray(event_da.event.values, dtype=int)].values


def make_stage_type_composites(sst_da, df_det, labels_det, reference_event_ids, K=4):
    stage_event_maps = {
        "Full duration": compute_event_stage_mean_ssta_from_df(sst_da, df_det, stage="full", peak_half_width=PEAK_HALF_WIDTH),
        "Onset stage (start–peak)": compute_event_stage_mean_ssta_from_df(sst_da, df_det, stage="onset", peak_half_width=PEAK_HALF_WIDTH),
        f"Peak ±{PEAK_HALF_WIDTH} d": compute_event_stage_mean_ssta_from_df(sst_da, df_det, stage="peak", peak_half_width=PEAK_HALF_WIDTH),
        "Decay stage (peak–end)": compute_event_stage_mean_ssta_from_df(sst_da, df_det, stage="decay", peak_half_width=PEAK_HALF_WIDTH),
    }
    stage_composites = {}
    for stage_name, em in stage_event_maps.items():
        labels_stage = align_labels_to_event_da(labels_det, em, reference_event_ids)
        stage_composites[stage_name] = compute_cluster_mean(em, labels_stage, K=K)
    return stage_event_maps, stage_composites


def stage_pattern_correlation_table(stage_composites) -> pd.DataFrame:
    full = stage_composites["Full duration"]
    rows = []
    for typ in range(full.sizes["cluster"]):
        row = {"Type": typ + 1}
        for stage_name, comp in stage_composites.items():
            if stage_name == "Full duration":
                continue
            row[f"Full vs {stage_name}"] = weighted_pattern_corr(full.sel(cluster=typ), comp.sel(cluster=typ))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_phase_normalized_onset_stage_composites(sst_da, events_df, labels, reference_event_ids, stages=PHASE_STAGES, K=4):
    label_series = pd.Series(np.asarray(labels, dtype=int), index=np.asarray(reference_event_ids, dtype=int))
    comp_by_type = []
    records = []
    tmin = pd.Timestamp(sst_da["time"].values[0]).normalize()
    tmax = pd.Timestamp(sst_da["time"].values[-1]).normalize()
    for cl in range(K):
        maps_by_phase = []
        event_ids_this_type = label_series.index[label_series.values == cl]
        for frac in stages:
            event_maps = []
            offsets = []
            for event_id in event_ids_this_type:
                row = events_df.loc[int(event_id)]
                t0 = pd.Timestamp(row["time_start"]).normalize()
                tp = pd.Timestamp(row["time_peak"]).normalize()
                if pd.isna(t0) or pd.isna(tp) or tp < t0:
                    continue
                onset_days = (tp - t0).days
                target_offset = 0 if onset_days == 0 else int(np.round(float(frac) * onset_days))
                target_time = t0 + pd.Timedelta(days=target_offset)
                if target_time < tmin or target_time > tmax:
                    continue
                try:
                    da = sst_da.sel(time=target_time, method="nearest", tolerance=np.timedelta64(1, "D"))
                except Exception:
                    continue
                event_maps.append(da.expand_dims(event=[int(event_id)]))
                offsets.append(target_offset)
            if not event_maps:
                raise RuntimeError(f"No SST maps for Type {cl + 1}, phase={frac}")
            da_phase = xr.concat(event_maps, dim="event").mean("event", skipna=True).expand_dims(phase=[float(frac)])
            maps_by_phase.append(da_phase)
            records.append({"type": cl + 1, "cluster": cl, "phase": float(frac),
                            "phase_label": PHASE_STAGE_LABELS.get(float(frac), f"{frac:g}"),
                            "n_event": len(event_maps),
                            "median_target_offset_days": float(np.nanmedian(offsets)) if offsets else np.nan})
        comp_by_type.append(xr.concat(maps_by_phase, dim="phase").expand_dims(cluster=[cl]))
    out = xr.concat(comp_by_type, dim="cluster")
    out.name = "sst_resid_phase_normalized_onset_stage_composite"
    return out, pd.DataFrame(records)


def compute_doy_std_for_standardization(sst_resid, baseline_start=STD_BASELINE_START, baseline_end=STD_BASELINE_END):
    base = sst_resid.sel(time=slice(baseline_start, baseline_end))
    doy = base["time"].dt.dayofyear
    sigma = base.groupby(doy).std("time", skipna=True)
    sigma = sigma.rename({"dayofyear": "doy"}) if "dayofyear" in sigma.dims else sigma
    if "doy" not in sigma.dims:
        sigma = sigma.rename({sigma.dims[0]: "doy"})
    sigma_smooth = sigma.rolling(doy=STD_SMOOTH_WINDOW, center=True, min_periods=1).mean()
    sigma_smooth = sigma_smooth.where(sigma_smooth >= STD_MIN_SIGMA, STD_MIN_SIGMA)
    return sigma_smooth


def standardize_sst_resid_by_doy(sst_resid, sigma_doy):
    sig = sigma_doy.sel(doy=sst_resid["time"].dt.dayofyear).assign_coords(time=sst_resid["time"])
    if "doy" in sig.coords:
        sig = sig.drop_vars("doy")
    z = (sst_resid / sig).astype("float32")
    z.name = "sst_resid_z"
    return z


def make_contingency_and_match(labels_ref, labels_sens, K=4, ref_name="Original type", sens_name="Standardized cluster"):
    labels_ref = np.asarray(labels_ref, dtype=int)
    labels_sens = np.asarray(labels_sens, dtype=int)
    tab = pd.crosstab(pd.Series(labels_ref, name=ref_name), pd.Series(labels_sens, name=sens_name))
    tab = tab.reindex(index=np.arange(K), columns=np.arange(K), fill_value=0)
    row_ind, col_ind = linear_sum_assignment(-tab.values)
    sens_to_ref = {int(c): int(r) for r, c in zip(row_ind, col_ind)}
    labels_sens_matched = np.array([sens_to_ref[int(x)] for x in labels_sens], dtype=int)
    ari = adjusted_rand_score(labels_ref, labels_sens)
    return tab, sens_to_ref, labels_sens_matched, ari



def find_climate_index_excel(base: Path) -> Path | None:
    candidates = [
        base.parent / "Documents" / "MHW" / "Climate_index" / "psl_indices_1982_2024.xlsx",
        base / "data" / "processed" / "climate_indices" / "psl_indices_1982_2024.xlsx",
        base / "data" / "core" / "climate_indices" / "psl_indices_1982_2024.xlsx",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def prepare_climate_df(path_climate_excel: Path | None) -> pd.DataFrame | None:
    if path_climate_excel is None or not Path(path_climate_excel).exists():
        print(f"[WARN] Climate indices file not found: {path_climate_excel}")
        return None
    df = pd.read_excel(path_climate_excel)
    lower_cols = {str(c).lower(): c for c in df.columns}
    if "date" in lower_cols:
        df[lower_cols["date"]] = pd.to_datetime(df[lower_cols["date"]])
        df = df.set_index(lower_cols["date"]).sort_index()
    elif "time" in lower_cols:
        df[lower_cols["time"]] = pd.to_datetime(df[lower_cols["time"]])
        df = df.set_index(lower_cols["time"]).sort_index()
    elif "year" in lower_cols and "month" in lower_cols:
        ycol = lower_cols["year"]
        mcol = lower_cols["month"]
        df["date"] = pd.to_datetime(dict(year=df[ycol].astype(int), month=df[mcol].astype(int), day=15))
        df = df.set_index("date").sort_index()
    else:
        try:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
        except Exception as exc:
            raise ValueError("Could not infer date index for climate_df.") from exc
    return df


def build_event_climate_index_df(df_events: pd.DataFrame, event_ids, labels, climate_df: pd.DataFrame, climate_indices) -> tuple[pd.DataFrame, list[str]]:
    available_indices = [idx for idx in climate_indices if idx in climate_df.columns]
    if not available_indices:
        raise ValueError("None of CLIMATE_INDICES found in climate_df columns.")
    label_series = pd.Series(np.asarray(labels, dtype=int), index=np.asarray(event_ids, dtype=int))
    events_source = df_events.loc[df_events.index.intersection(label_series.index)].copy()
    peak_time_col = "time_peak" if "time_peak" in events_source.columns else "time_start"
    records = []
    for event_id, row in events_source.iterrows():
        peak_time = pd.to_datetime(row[peak_time_col])
        if pd.isna(peak_time):
            continue
        closest_idx = climate_df.index.get_indexer([peak_time], method="nearest")[0]
        if closest_idx < 0:
            continue
        closest_date = climate_df.index[closest_idx]
        rec = {
            "event_id": int(event_id),
            "event_date": peak_time,
            "closest_climate_date": closest_date,
            "cluster": int(label_series.loc[int(event_id)]),
        }
        for idx_name in available_indices:
            val = climate_df.loc[closest_date, idx_name]
            rec[idx_name] = float(val) if pd.notna(val) else np.nan
        records.append(rec)
    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out, available_indices
    return out.set_index("event_id").sort_index(), available_indices


def summarize_index_significance(df: pd.DataFrame, indices, alpha=0.05) -> pd.DataFrame:
    rows = []
    for idx_name in indices:
        vals = df[idx_name].dropna().values if idx_name in df.columns else np.array([])
        n = len(vals)
        if n < 3:
            rows.append({"Index": idx_name, "N": n, "Mean": np.nan, "Std": np.nan, "p_value": np.nan, "Significance": "insufficient sample"})
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1))
        se = std / np.sqrt(n)
        p = np.nan if se == 0 or not np.isfinite(se) else 2 * tdist.sf(abs(mean / se), df=n - 1)
        if np.isnan(p):
            sig = "undefined"
        elif p < alpha and mean > 0:
            sig = "significantly positive"
        elif p < alpha and mean < 0:
            sig = "significantly negative"
        else:
            sig = "not significant"
        rows.append({"Index": idx_name, "N": n, "Mean": mean, "Std": std, "p_value": p, "Significance": sig})
    return pd.DataFrame(rows).set_index("Index")


def prepare_significant_index_order(all_signif: pd.DataFrame, cluster_results: dict, available_indices: list[str]) -> list[str]:
    sig_indices_union = set()
    for idx_name in available_indices:
        if idx_name in all_signif.index and all_signif.loc[idx_name, "Significance"] in ["significantly positive", "significantly negative"]:
            sig_indices_union.add(idx_name)
        for cl, df_res in cluster_results.items():
            if idx_name in df_res.index and df_res.loc[idx_name, "Significance"] in ["significantly positive", "significantly negative"]:
                sig_indices_union.add(idx_name)
    return [idx for idx in available_indices if idx in sig_indices_union]

def load_and_compute(base: Path, table_dir: Path):
    core = base / "data/core"
    path_out = core / "eof_kmeans"
    path_out.mkdir(parents=True, exist_ok=True)
    path_sst = core / "oisst" / "oisst_sst_full_anom_trend_resid_1982-2024_xmhw.nc"
    path_mhw_raw = core / "oisst" / "mhw_basin_raw_1982-2024_xmhw.nc"
    path_mhw_det = core / "oisst" / "mhw_basin_detrended_1982-2024_xmhw.nc"
    if not path_sst.exists():
        raise FileNotFoundError(path_sst)
    if not path_mhw_det.exists():
        raise FileNotFoundError(path_mhw_det)

    ds_sst = xr.open_dataset(path_sst)
    ds_sst = normalize_longitude(ds_sst)
    sst_anom = ds_sst["sst_anom"]
    sst_resid = ds_sst["sst_resid"]
    df_det = mhw_ds_to_df(xr.open_dataset(path_mhw_det))
    df_raw = mhw_ds_to_df(xr.open_dataset(path_mhw_raw)) if path_mhw_raw.exists() else df_det.copy()

    # Main classification uses detrended event list and detrended fields.
    event_mean_det = compute_event_mean_ssta_from_df(sst_resid, df_det, min_duration=3)
    reference_event_ids = event_mean_det.event.values
    main = run_eof_kmeans_classification(event_mean_det, K=K_MAIN, n_modes=N_MODES_EOF, n_pc=N_PC_MAIN, swap_map=SWAP_MAP_MAIN)
    labels_det = main["labels"]
    cluster_mean_det = main["cluster_mean"]

    # Save reusable main outputs.
    ds_main = xr.Dataset({
        "cluster_mean_ssta": cluster_mean_det,
        "eof": main["eofs"],
        "pc": main["pcs"],
        "explained_variance": (("mode",), main["expvar"]),
        "cluster_label_K4": (("event",), labels_det),
    }).assign_coords(event=event_mean_det.event)
    ds_main.attrs["K"] = K_MAIN
    ds_main.attrs["n_pc_used_for_kmeans"] = N_PC_MAIN
    ds_main.to_netcdf(path_out / "detrend_labels_clustermean_EOFKMeans_K4_revision.nc")

    # Same-event raw vs detrended EOF sensitivity, so PC correlations are interpretable.
    event_mean_raw_same_events = compute_event_mean_ssta_from_df(sst_anom, df_det, min_duration=3)
    event_mean_raw_same_events = event_mean_raw_same_events.sel(event=event_mean_det.event)
    eofs_raw_same, pcs_raw_same, expvar_raw_same = eof_from_event_means(event_mean_raw_same_events, n_modes=N_MODES_EOF, use_weight=True)
    eofs_det_same, pcs_det_same, expvar_det_same = eof_from_event_means(event_mean_det, n_modes=N_MODES_EOF, use_weight=True)

    # Optional original raw-event metrics use raw events, matching the old notebook's K-selection logic.
    event_mean_raw_eventlist = compute_event_mean_ssta_from_df(sst_anom, df_raw, min_duration=3)
    eofs_raw_event, pcs_raw_event, expvar_raw_event = eof_from_event_means(event_mean_raw_eventlist, n_modes=N_MODES_EOF, use_weight=True)

    # K metrics: same as old notebook, raw-event vs detrended-event.
    cluster_rows = []
    silhouette_inputs = {}
    for method, pcs_da in [("raw", pcs_raw_event), ("detrend", main["pcs"] )]:
        pc_input = pcs_da.sel(mode=slice(1, N_PC_MAIN)).values
        silhouette_inputs[method] = pc_input
        for K in [2, 3, 4, 5, 6]:
            km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=50)
            lab = km.fit_predict(pc_input)
            cluster_rows.append({
                "method": method,
                "K": K,
                "SSE": km.inertia_,
                "Silhouette": silhouette_score(pc_input, lab),
                "CH": calinski_harabasz_score(pc_input, lab),
                "DB": davies_bouldin_score(pc_input, lab),
            })
    cluster_stats_df = pd.DataFrame(cluster_rows)
    cluster_stats_df.to_csv(table_dir / "kmeans_metrics_raw_detrend_revision.csv", index=False)

    # Stage composites and phase-normalized onset evolution.
    stage_event_maps, stage_composites = make_stage_type_composites(sst_resid, df_det, labels_det, reference_event_ids, K=K_MAIN)
    stage_corr_df = stage_pattern_correlation_table(stage_composites)
    stage_corr_df.to_csv(table_dir / "TableS1_full_stage_pattern_correlations.csv", index=False)
    stage_ds = xr.Dataset({})
    for stage_name, comp in stage_composites.items():
        key = stage_name.lower().replace(" ", "_").replace("±", "pm").replace("-", "_").replace("(", "").replace(")", "")
        stage_ds[f"ssta_{key}"] = comp
    stage_ds.to_netcdf(path_out / "full_onset_peak_type_composites_same_labels.nc")

    sst_phase_comp, df_phase_counts = compute_phase_normalized_onset_stage_composites(
        sst_resid, df_det, labels_det, reference_event_ids, stages=PHASE_STAGES, K=K_MAIN
    )
    sst_phase_comp.to_dataset().to_netcdf(path_out / "sst_phase_normalized_start_50_peak_by_full_duration_type.nc")
    df_phase_counts.to_csv(table_dir / "phase_normalized_event_counts.csv", index=False)

    # Season-standardized clustering sensitivity.
    sigma_doy = compute_doy_std_for_standardization(sst_resid)
    sst_z = standardize_sst_resid_by_doy(sst_resid, sigma_doy)
    event_mean_z = compute_event_mean_ssta_from_df(sst_z, df_det, min_duration=3).sel(event=event_mean_det.event)
    std_cls = run_eof_kmeans_classification(event_mean_z, K=K_MAIN, n_modes=N_MODES_EOF, n_pc=N_PC_MAIN, swap_map=None)
    labels_ref_for_z = labels_det
    labels_z_raw = std_cls["labels"]
    tab_std, sens_to_ref, labels_z_matched, ari_std = make_contingency_and_match(labels_ref_for_z, labels_z_raw, K=K_MAIN)
    cluster_mean_z_raw = compute_cluster_mean(event_mean_z, labels_z_raw, K=K_MAIN)
    # Reorder standardized cluster means to matched Type 1-4 order.
    ref_to_sens = {ref: sens for sens, ref in sens_to_ref.items()}
    matched_means = []
    for ref in range(K_MAIN):
        sens = ref_to_sens[ref]
        matched_means.append(cluster_mean_z_raw.sel(cluster=sens).expand_dims(cluster=[ref]))
    cluster_mean_z_matched = xr.concat(matched_means, dim="cluster")

    std_corr_rows = []
    for typ in range(K_MAIN):
        std_corr_rows.append({
            "Type": typ + 1,
            "Pattern corr: original SSTA vs standardized-z composite": weighted_pattern_corr(
                cluster_mean_det.sel(cluster=typ), cluster_mean_z_matched.sel(cluster=typ)
            ),
        })
    std_corr_df = pd.DataFrame(std_corr_rows)

    with pd.ExcelWriter(table_dir / "TableS2_standardized_clustering_summary.xlsx") as writer:
        tab_std.to_excel(writer, sheet_name="contingency_0based")
        pd.DataFrame({"ARI": [ari_std]}).to_excel(writer, sheet_name="ARI", index=False)
        std_corr_df.to_excel(writer, sheet_name="pattern_correlation", index=False)
    tab_std.to_csv(table_dir / "TableS2_original_vs_standardized_contingency.csv")
    pd.DataFrame({"ARI": [ari_std]}).to_csv(table_dir / "TableS2_standardized_clustering_ARI.csv", index=False)
    std_corr_df.to_csv(table_dir / "TableS2_standardized_pattern_correlations.csv", index=False)
    xr.Dataset({
        "cluster_mean_std_matched": cluster_mean_z_matched,
        "cluster_label_standardized_raw": (("event",), labels_z_raw),
        "cluster_label_standardized_matched": (("event",), labels_z_matched),
        "cluster_label_original": (("event",), labels_ref_for_z),
    }).assign_coords(event=event_mean_det.event).to_netcdf(path_out / "season_standardized_clustering_sensitivity.nc")

    climate_path = find_climate_index_excel(base)
    climate_df = prepare_climate_df(climate_path)
    event_idx_df = None
    available_indices = []
    ordered_sig_indices = []
    cluster_results = {}
    all_index_signif = None
    if climate_df is not None:
        try:
            event_idx_df, available_indices = build_event_climate_index_df(df_det, reference_event_ids, labels_det, climate_df, CLIMATE_INDICES)
            if event_idx_df is not None and not event_idx_df.empty:
                all_index_signif = summarize_index_significance(event_idx_df, available_indices)
                for cl in range(K_MAIN):
                    cluster_results[cl] = summarize_index_significance(event_idx_df[event_idx_df["cluster"] == cl], available_indices)
                ordered_sig_indices = prepare_significant_index_order(all_index_signif, cluster_results, available_indices)
                if not ordered_sig_indices:
                    ordered_sig_indices = available_indices.copy()
        except Exception as exc:
            print(f"[WARN] Climate-index processing failed: {exc}")
            event_idx_df = None
            available_indices = []
            ordered_sig_indices = []
            cluster_results = {}
            all_index_signif = None

    return {
        "df_det": df_det,
        "labels_det": labels_det,
        "cluster_mean_det": cluster_mean_det,
        "event_mean_det": event_mean_det,
        "reference_event_ids": reference_event_ids,
        "sst_phase_comp": sst_phase_comp,
        "stage_composites": stage_composites,
        "stage_corr_df": stage_corr_df,
        "cluster_stats_df": cluster_stats_df,
        "silhouette_inputs": silhouette_inputs,
        # Same-event EOFs are kept only for diagnostics if needed.
        "eofs_raw_same": eofs_raw_same,
        "pcs_raw_same": pcs_raw_same,
        "expvar_raw_same": expvar_raw_same,
        "eofs_det_same": eofs_det_same,
        "pcs_det_same": pcs_det_same,
        "expvar_det_same": expvar_det_same,
        # Original-notebook EOF sensitivity: raw-event list versus detrended-event list.
        "eofs_raw_event": eofs_raw_event,
        "pcs_raw_event": pcs_raw_event,
        "expvar_raw_event": expvar_raw_event,
        "cluster_mean_z_matched": cluster_mean_z_matched,
        "tab_std": tab_std,
        "ari_std": ari_std,
        "std_corr_df": std_corr_df,
        "event_idx_df": event_idx_df,
        "available_indices": available_indices,
        "ordered_sig_indices": ordered_sig_indices,
        "cluster_results": cluster_results,
        "all_index_signif": all_index_signif,
    }


# ============================================================
# Plot helpers
# ============================================================

def plot_map_panel(ax, da, *, vmin=None, vmax=None, levels=None, cmap=CMAP_SST, contour_level=None,
                   row=0, col=0, nrow=1, ncol=1, label=None, extent=GOM_EXTENT_WIDE):
    da = normalize_longitude(da)
    da, lon, lat, lon2d, lat2d = lon_lat_mesh(da)
    if levels is not None:
        im = ax.contourf(lon2d, lat2d, da.values, levels=levels, cmap=cmap, extend="both", transform=ccrs.PlateCarree())
    else:
        im = ax.pcolormesh(lon, lat, da, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", transform=ccrs.PlateCarree())
    if contour_level is not None:
        da_min = float(np.nanmin(da.values))
        da_max = float(np.nanmax(da.values))
        if da_min <= contour_level <= da_max:
            ax.contour(lon, lat, da, levels=[contour_level], colors="k", linewidths=0.60, transform=ccrs.PlateCarree())
    add_gom_base(ax, extent=extent)
    map_gridlines(ax, row, col, nrow, ncol)
    if label:
        add_panel_label(ax, label, fontsize=7.0)
    return im


def plot_fig01_type_evolution(data: dict, main_dir: Path, show=False):
    cluster_mean = data["cluster_mean_det"]
    phase_comp = data["sst_phase_comp"]
    labels = data["labels_det"]
    K = 4
    columns = [("Full-duration\ncomposite", None)] + [(PHASE_STAGE_LABELS[float(ph)], float(ph)) for ph in phase_comp["phase"].values]
    ncols = len(columns)
    nrows = K
    type_counts = {cl: int(np.sum(labels == cl)) for cl in range(K)}

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(FIG_W_FULL, 4.75),
        subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=False
    )
    axes = np.atleast_2d(axes)
    pcm = None
    for i in range(K):
        for j, (title, phase) in enumerate(columns):
            ax = axes[i, j]
            da = cluster_mean.sel(cluster=i) if phase is None else phase_comp.sel(cluster=i, phase=phase)
            pcm = plot_map_panel(
                ax, da, vmin=-0.5, vmax=1.5, cmap=CMAP_SST,
                contour_level=CORE_CONTOUR_LEVEL, row=i, col=j, nrow=nrows, ncol=ncols,
                label=PANEL_LABELS[i * ncols + j]
            )
            if i == 0:
                ax.set_title(title, fontsize=8.1, pad=3.5, fontweight="semibold")

    fig.subplots_adjust(left=0.145, right=0.885, top=0.89, bottom=0.075, wspace=0.03, hspace=0.045)

    # Create a slightly larger gap between the first column and the onset-evolution columns.
    extra_gap = 0.014
    shrink = 0.004
    for i in range(nrows):
        for j in range(1, ncols):
            pos = axes[i, j].get_position()
            axes[i, j].set_position([pos.x0 + extra_gap, pos.y0, pos.width - shrink, pos.height])

    # Put row labels farther to the left in figure coordinates, avoiding overlap with latitude labels.
    for i in range(K):
        pos = axes[i, 0].get_position()
        x = pos.x0 - 0.055
        y = 0.5 * (pos.y0 + pos.y1)
        fig.text(x, y, f"Type {i+1}\n(n={type_counts.get(i,0)})", ha="center", va="center",
                 rotation=90, fontsize=8.2, fontweight="semibold")

    # Group titles and separator between the classification composite and onset-evolution columns.
    pos0 = axes[0, 0].get_position()
    pos1 = axes[0, 1].get_position()
    pos_last = axes[0, -1].get_position()
    group1_x = 0.5 * (pos0.x0 + pos0.x1)
    group2_x = 0.5 * (pos1.x0 + pos_last.x1)
    #fig.text(group1_x, 0.965, "Full-duration classification", ha="center", va="bottom", fontsize=8.6, fontweight="semibold")
    fig.text(group2_x, 0.955, "Phase-normalized onset-stage evolution", ha="center", va="bottom", fontsize=8.6, fontweight="semibold")
    x_sep = 0.5 * (pos0.x1 + pos1.x0)
    fig.add_artist(Line2D([x_sep, x_sep], [0.075, 0.895], transform=fig.transFigure, color="0.25", lw=0.8))

    cbar_ax = fig.add_axes([0.915, 0.18, 0.017, 0.64])
    cbar = fig.colorbar(pcm, cax=cbar_ax, orientation="vertical", extend="both")
    cbar.set_label("SST residual anomaly (°C)", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.0, width=0.6, length=2.2)
    cbar.outline.set_linewidth(0.6)
    save_figure(fig, main_dir, "Fig01_type_composite_and_onset_stage_evolution")
    if show:
        plt.show()
    plt.close(fig)


def plot_fig02_event_characteristics(data: dict, main_dir: Path, show=False):
    df = data["df_det"].copy()
    labels = np.asarray(data["labels_det"])
    df["type"] = labels
    df["start_month"] = pd.to_datetime(df["time_start"]).dt.month
    if "duration_days" not in df.columns:
        df["duration_days"] = df["duration"]
    vars_to_plot = [
        ("duration_days", "Duration", "Days"),
        ("intensity_max", "Maximum intensity", "°C"),
        ("intensity_mean", "Mean intensity", "°C"),
        ("intensity_cumulative", "Cumulative intensity", "°C·days"),
    ]
    months = np.arange(1, 13)
    month_ymax = 0
    for cl in range(4):
        counts = df.loc[df["type"] == cl, "start_month"].value_counts().reindex(months, fill_value=0)
        month_ymax = max(month_ymax, int(counts.max()))
    month_ymax = max(1, int(np.ceil(month_ymax * 1.18)))

    fig, axes = plt.subplots(2, 4, figsize=(FIG_W_FULL, 4.25), gridspec_kw={"height_ratios": [0.95, 1.15]})
    for j, cl in enumerate(range(4)):
        ax = axes[0, j]
        counts = df.loc[df["type"] == cl, "start_month"].value_counts().reindex(months, fill_value=0)
        ax.bar(months, counts.values, color=TYPE_COLORS[cl + 1], alpha=0.92, edgecolor="black", linewidth=0.55)
        ax.set_xlim(0.3, 12.7)
        ax.set_ylim(0, month_ymax)
        ax.set_xticks([1, 3, 5, 7, 9, 11])
        ax.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov"], rotation=30, ha="right")
        ax.set_xlabel("Start month")
        ax.set_ylabel("Event count" if j == 0 else "")
        add_panel_label(ax, PANEL_LABELS[j], f"Type {cl + 1}", f"n = {int((df['type']==cl).sum())}", fontsize=7.1, boxed=True, x=0.02, y=0.97)
        apply_grid(ax, axis="y", alpha=0.18)
        thin_spines(ax)
    for j, (var, title, ylabel) in enumerate(vars_to_plot):
        ax = axes[1, j]
        data_by_type = [df.loc[df["type"] == cl, var].dropna().values for cl in range(4)]
        bp = ax.boxplot(
            data_by_type, patch_artist=True, labels=["T1", "T2", "T3", "T4"],
            boxprops=dict(linewidth=0.75, color="black"),
            medianprops=dict(linewidth=0.95, color="black"),
            whiskerprops=dict(linewidth=0.75, color="black"),
            capprops=dict(linewidth=0.75, color="black"),
            flierprops=dict(marker="o", markersize=2.6, markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.7),
        )
        for patch, cl in zip(bp["boxes"], range(4)):
            patch.set_facecolor(TYPE_COLORS[cl + 1])
            patch.set_alpha(0.86)
        # Put labels inside the axes; this avoids collisions with the row above and boxplot outliers.
        add_panel_label(ax, PANEL_LABELS[4 + j], title, boxed=True, x=0.02, y=0.97, fontsize=7.2)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.08 * (ymax - ymin))
        ax.set_xlabel("MHW type")
        ax.set_ylabel(ylabel)
        apply_grid(ax, axis="y", alpha=0.18)
        thin_spines(ax)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.975, bottom=0.105, wspace=0.33, hspace=0.30)
    save_figure(fig, main_dir, "Fig02_month_and_characteristics")
    if show:
        plt.show()
    plt.close(fig)


def pattern_corr_simple(a: xr.DataArray, b: xr.DataArray) -> float:
    """Unweighted pattern correlation matching the original notebook sensitivity plot."""
    A = a.values.ravel()
    B = b.values.ravel()
    mask = np.isfinite(A) & np.isfinite(B)
    if mask.sum() < 10:
        return np.nan
    A = A[mask] - A[mask].mean()
    B = B[mask] - B[mask].mean()
    denom = np.sqrt(np.dot(A, A) * np.dot(B, B))
    return float(np.dot(A, B) / denom) if denom else np.nan


def eof_alignment(eofs_raw, pcs_raw, expvar_raw, eofs_det, pcs_det, expvar_det, n=4, n_match=5):
    """Align raw EOF modes to detrended modes using the original notebook's greedy pairing.

    EOF signs are arbitrary. For each raw/detrend pair, choose the sign that maximizes
    positive pattern correlation, then greedily assign one-to-one pairs by descending
    absolute correlation. Detrended mode order is preserved in the returned records.
    """
    n_match = min(n_match, eofs_raw.sizes["mode"], eofs_det.sizes["mode"])
    corr_mat = np.full((n_match, n_match), np.nan)
    sign_mat = np.ones((n_match, n_match), dtype=float)
    for i in range(n_match):
        for j in range(n_match):
            r = pattern_corr_simple(eofs_raw.sel(mode=i + 1), eofs_det.sel(mode=j + 1))
            if np.isnan(r):
                continue
            if r < 0:
                sign_mat[i, j] = -1.0
                corr_mat[i, j] = -r
            else:
                sign_mat[i, j] = 1.0
                corr_mat[i, j] = r

    all_pairs = []
    for i in range(n_match):
        for j in range(n_match):
            if np.isfinite(corr_mat[i, j]):
                all_pairs.append((i + 1, j + 1, corr_mat[i, j]))
    all_pairs.sort(key=lambda x: x[2], reverse=True)

    raw_to_det = {}
    used_det = set()
    for raw_mode, det_mode, corr_val in all_pairs:
        if raw_mode not in raw_to_det and det_mode not in used_det:
            raw_to_det[raw_mode] = det_mode
            used_det.add(det_mode)
    det_to_raw = {det: raw for raw, det in raw_to_det.items()}

    records = []
    for det_mode in range(1, n + 1):
        raw_mode = det_to_raw.get(det_mode, det_mode)
        sign = sign_mat[raw_mode - 1, det_mode - 1] if raw_mode <= n_match and det_mode <= n_match else 1.0
        raw_eof = eofs_raw.sel(mode=raw_mode) * sign
        raw_pc = pcs_raw.sel(mode=raw_mode) * sign
        det_pc = pcs_det.sel(mode=det_mode)
        common = np.intersect1d(raw_pc.event.values, det_pc.event.values)
        pc_r = np.nan
        if len(common) > 3:
            try:
                pc_r = float(np.corrcoef(raw_pc.sel(event=common).values, det_pc.sel(event=common).values)[0, 1])
            except Exception:
                pc_r = np.nan
        records.append({
            "det_mode": det_mode,
            "raw_mode": raw_mode,
            "map_r": pattern_corr_simple(raw_eof, eofs_det.sel(mode=det_mode)),
            "pc_r": pc_r,
            "sign": sign,
            "raw_eof_aligned": raw_eof,
            "raw_var": float(expvar_raw[raw_mode - 1]),
            "det_var": float(expvar_det[det_mode - 1]),
        })
    return records


def plot_figS04_eof_compare(data: dict, si_dir: Path, show=False):
    # Use the same raw-event-list versus detrended-event-list comparison as the original notebook,
    # so the explained variance and EOF patterns match the previous SI figure.
    eofs_raw = data["eofs_raw_event"]
    pcs_raw = data["pcs_raw_event"]
    expvar_raw = data["expvar_raw_event"]
    eofs_det = data["eofs_det_same"]
    pcs_det = data["pcs_det_same"]
    expvar_det = data["expvar_det_same"]
    records = eof_alignment(eofs_raw, pcs_raw, expvar_raw, eofs_det, pcs_det, expvar_det, n=4)
    vals = []
    for rec in records:
        vals.append(rec["raw_eof_aligned"].values)
        vals.append(eofs_det.sel(mode=rec["det_mode"]).values)
    vmax = np.nanmax(np.abs(np.concatenate([v.ravel() for v in vals])))
    vmax = np.ceil(vmax / 0.01) * 0.01

    fig, axes = plt.subplots(2, 4, figsize=(FIG_W_FULL, 3.30), subplot_kw={"projection": ccrs.PlateCarree()})
    for j, rec in enumerate(records):
        raw_da = rec["raw_eof_aligned"]
        det_da = eofs_det.sel(mode=rec["det_mode"])
        im = plot_map_panel(axes[0, j], raw_da, vmin=-vmax, vmax=vmax, cmap=CMAP_SST,
                            row=0, col=j, nrow=2, ncol=4, extent=GOM_EXTENT_WIDE)
        axes[0, j].set_title(f"raw EOF{rec['raw_mode']} aligned to detrend {rec['det_mode']}\n({rec['raw_var']*100:.1f}% var)", fontsize=7.6, pad=2.7)
        im = plot_map_panel(axes[1, j], det_da, vmin=-vmax, vmax=vmax, cmap=CMAP_SST,
                            row=1, col=j, nrow=2, ncol=4, extent=GOM_EXTENT_WIDE)
        axes[1, j].set_title(
            f"detrend EOF{rec['det_mode']}\n({rec['det_var']*100:.1f}% var; map r={rec['map_r']:.2f})",
            fontsize=7.3, pad=2.7,
        )
    fig.subplots_adjust(left=0.075, right=0.90, top=0.94, bottom=0.13, wspace=0.07, hspace=0.12)
    cax = fig.add_axes([0.92, 0.20, 0.018, 0.62])
    cbar = fig.colorbar(im, cax=cax, orientation="vertical", extend="both")
    cbar.set_label("EOF amplitude", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.0)
    cbar.outline.set_linewidth(0.6)
    save_figure(fig, si_dir, "FigS04_EOF_raw_vs_detrended")
    if show:
        plt.show()
    plt.close(fig)


def plot_figS05_cumulative_variance(data: dict, si_dir: Path, show=False):
    exp_raw = data["expvar_raw_event"]
    exp_det = data["expvar_det_same"]
    modes = np.arange(1, min(10, len(exp_raw), len(exp_det)) + 1)
    fig, ax = plt.subplots(figsize=(FIG_W_HALF, 2.45))
    ax.plot(modes, np.cumsum(exp_raw[:len(modes)]) * 100, marker="o", ms=3.2, lw=1.0, color="#0072B2", label="deseasoned only")
    ax.plot(modes, np.cumsum(exp_det[:len(modes)]) * 100, marker="s", ms=3.0, lw=1.0, color="#E69F00", label="detrended")
    ax.set_xlabel("EOF mode")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_xlim(0.8, len(modes) + 0.2)
    ax.set_ylim(0, 100)
    ax.set_xticks(modes)
    ax.legend(frameon=True, loc="lower right")
    apply_grid(ax, alpha=0.20)
    thin_spines(ax)
    save_figure(fig, si_dir, "FigS05_cumulative_variance_raw_vs_detrended")
    if show:
        plt.show()
    plt.close(fig)


def plot_figS06_cluster_metrics(data: dict, si_dir: Path, show=False):
    df = data["cluster_stats_df"]
    metrics = [("SSE", "SSE (elbow)"), ("Silhouette", "Silhouette"), ("CH", "Calinski–Harabasz"), ("DB", "Davies–Bouldin")]
    method_colors = {"raw": "#0072B2", "detrend": "#E69F00"}
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W_FULL, 4.4))
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, metrics):
        for method in ["raw", "detrend"]:
            sub = df[df["method"] == method].sort_values("K")
            ax.plot(sub["K"], sub[metric], marker="o", ms=3.2, lw=1.0, color=method_colors[method], label=method)
        ax.set_title(title)
        ax.set_xlabel("Number of clusters K")
        ax.set_xticks([2, 3, 4, 5, 6])
        apply_grid(ax, alpha=0.20)
        thin_spines(ax)
    axes[0].legend(frameon=True)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.95, bottom=0.10, wspace=0.25, hspace=0.35)
    save_figure(fig, si_dir, "FigS06_cluster_evaluation_metrics")
    if show:
        plt.show()
    plt.close(fig)


def plot_figS06b_silhouette_profiles(data: dict, si_dir: Path, show=False):
    method_colors = {"raw": "#0072B2", "detrend": "#E69F00"}
    fig, axes = plt.subplots(5, 2, figsize=(FIG_W_FULL, 8.2), sharex=True)
    for r, K in enumerate([2, 3, 4, 5, 6]):
        for c, method in enumerate(["raw", "detrend"]):
            ax = axes[r, c]
            X = data["silhouette_inputs"][method]
            km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=50)
            lab = km.fit_predict(X)
            sil = silhouette_samples(X, lab)
            mean_s = silhouette_score(X, lab)
            y_lower = 10
            for cl in range(K):
                vals = np.sort(sil[lab == cl])
                size = vals.shape[0]
                y_upper = y_lower + size
                ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.75)
                ax.text(-0.08, y_lower + 0.5 * size, f"C{cl}", fontsize=6.0, va="center")
                y_lower = y_upper + 10
            ax.axvline(mean_s, color="red", ls="--", lw=0.8)
            ax.set_title(f"{method} K={K}\nmean S={mean_s:.3f}", fontsize=7.0)
            ax.set_yticks([])
            ax.set_xlim(-0.15, 0.55)
            thin_spines(ax)
    axes[-1, 0].set_xlabel("Silhouette coefficient")
    axes[-1, 1].set_xlabel("Silhouette coefficient")
    fig.subplots_adjust(left=0.08, right=0.995, top=0.965, bottom=0.06, wspace=0.10, hspace=0.58)
    save_figure(fig, si_dir, "FigS06b_silhouette_profiles_raw_vs_detrended")
    if show:
        plt.show()
    plt.close(fig)


def plot_figS07_stage_composites(data: dict, si_dir: Path, show=False):
    comps = data["stage_composites"]
    stage_names = list(comps.keys())
    K = 4
    n_stage = len(stage_names)
    all_vals = np.concatenate([c.values.ravel() for c in comps.values()])
    vmax = np.ceil(np.nanmax(np.abs(all_vals)) / 0.2) * 0.2
    levels = np.arange(-vmax, vmax + 0.001, 0.2)
    fig, axes = plt.subplots(K, n_stage, figsize=(FIG_W_FULL, 4.95), subplot_kw={"projection": ccrs.PlateCarree()})
    axes = np.atleast_2d(axes)
    im = None
    for i in range(K):
        for j, stage in enumerate(stage_names):
            da = comps[stage].sel(cluster=i)
            im = plot_map_panel(axes[i, j], da, levels=levels, cmap=CMAP_SST, contour_level=CORE_CONTOUR_LEVEL,
                                row=i, col=j, nrow=K, ncol=n_stage, label=PANEL_LABELS[i * n_stage + j])
            if i == 0:
                axes[i, j].set_title(stage, fontsize=8.1, pad=3.0, fontweight="semibold")
            if j == 0:
                axes[i, j].text(-0.24, 0.5, f"Type {i+1}", transform=axes[i, j].transAxes, rotation=90,
                                ha="center", va="center", fontsize=7.8, fontweight="semibold")
    fig.subplots_adjust(left=0.13, right=0.89, top=0.94, bottom=0.125, wspace=0.035, hspace=0.045)
    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.045, pad=0.055, aspect=42)
    cbar.set_label("SST residual anomaly (°C)", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.0)
    cbar.outline.set_linewidth(0.6)
    save_figure(fig, si_dir, "FigS07_full_onset_peak_decay_type_composites")
    if show:
        plt.show()
    plt.close(fig)


def plot_figS08_standardized(data: dict, si_dir: Path, show=False):
    z = data["cluster_mean_z_matched"]
    all_vals = z.values.ravel()
    vmax = np.ceil(np.nanmax(np.abs(all_vals)) / 0.25) * 0.25
    levels = np.arange(-vmax, vmax + 0.001, 0.25)
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W_FULL, 5.2), subplot_kw={"projection": ccrs.PlateCarree()})
    axes = axes.ravel()
    im = None
    for i, ax in enumerate(axes):
        im = plot_map_panel(ax, z.sel(cluster=i), levels=levels, cmap=CMAP_SST,
                            row=i // 2, col=i % 2, nrow=2, ncol=2,
                            label=f"{PANEL_LABELS[i]} Matched Type {i+1}")
    fig.subplots_adjust(left=0.07, right=0.995, top=0.965, bottom=0.16, wspace=0.06, hspace=0.07)
    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.06, aspect=38)
    cbar.set_label("Season-standardized SST residual (z score)", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.0)
    cbar.outline.set_linewidth(0.6)
    save_figure(fig, si_dir, "FigS08_season_standardized_type_maps")
    if show:
        plt.show()
    plt.close(fig)



def plot_figS09_climate_indices(data: dict, si_dir: Path, show=False):
    event_idx_df = data.get("event_idx_df")
    ordered = data.get("ordered_sig_indices", [])
    cluster_results = data.get("cluster_results", {})
    if event_idx_df is None or getattr(event_idx_df, "empty", True):
        print("[INFO] Skipping FigS09 climate indices: no climate-index data available.")
        return
    if not ordered:
        ordered = [idx for idx in data.get("available_indices", []) if idx in event_idx_df.columns]
    if not ordered:
        print("[INFO] Skipping FigS09 climate indices: no usable indices found.")
        return

    y = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 4, figsize=(FIG_W_FULL, 2.75), sharex=True, constrained_layout=False)
    axes = np.atleast_1d(axes)
    x_vals = []
    for idx_name in ordered:
        vals = event_idx_df[idx_name].dropna().values if idx_name in event_idx_df.columns else np.array([])
        if len(vals):
            x_vals.append(vals)
    if x_vals:
        x_all = np.concatenate(x_vals)
        xmax = max(abs(np.nanmin(x_all)), abs(np.nanmax(x_all)))
        xmax = max(1.0, np.ceil((xmax + 0.25) / 0.5) * 0.5)
    else:
        xmax = 1.0

    for i, ax in enumerate(axes):
        df_group = event_idx_df[event_idx_df["cluster"] == i]
        signif_df = cluster_results.get(i)
        if signif_df is None or signif_df.empty:
            idx_for_group = [idx for idx in ordered if idx in df_group.columns]
        else:
            idx_for_group = [
                idx for idx in ordered
                if idx in signif_df.index and signif_df.loc[idx, "Significance"] in ["significantly positive", "significantly negative"]
            ]
        valid_pairs = [(idx, df_group[idx].dropna().values) for idx in idx_for_group if idx in df_group.columns and len(df_group[idx].dropna()) > 0]
        if valid_pairs:
            data_boxes = [vals for _, vals in valid_pairs]
            positions = [ordered.index(idx) for idx, _ in valid_pairs]
            bp = ax.boxplot(
                data_boxes, positions=positions, vert=False, widths=0.54, patch_artist=True, showfliers=False,
                boxprops=dict(linewidth=0.8, color="0.2"), medianprops=dict(linewidth=1.0, color="darkorange"),
                whiskerprops=dict(linewidth=0.8, color="0.2"), capprops=dict(linewidth=0.8, color="0.2"),
            )
            for box in bp["boxes"]:
                box.set_facecolor(TYPE_COLORS_LIST[i])
                box.set_alpha(0.90)
        ax.axvline(0, color="0.35", lw=0.8, ls="--")
        ax.set_xlim(-xmax, xmax)
        ax.set_xlabel("Index value", fontsize=7.8)
        ax.set_yticks(y)
        ax.set_ylim(-0.5, len(ordered) - 0.5)
        ax.invert_yaxis()
        if i == 0:
            ax.set_yticklabels(ordered, fontsize=7.0)
            ax.tick_params(axis="y", labelleft=True)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", labelleft=False)
        add_panel_label(ax, PANEL_LABELS[i], title=f"Type {i+1}", fontsize=7.3)
        apply_grid(ax, axis="x", alpha=0.18)
        thin_spines(ax)

    fig.subplots_adjust(left=0.12, right=0.995, top=0.95, bottom=0.18, wspace=0.14)
    save_figure(fig, si_dir, "FigS09_climate_indices_by_type")
    if show:
        plt.show()
    plt.close(fig)

def main():
    args = parse_args()
    set_mpl_style()
    main_dir = args.main_dir or (REVISED_DIR / "output" / "main")
    si_dir = args.si_dir or (REVISED_DIR / "output" / "si")
    table_dir = args.table_dir or (REVISED_DIR / "output" / "tables")
    table_dir.mkdir(parents=True, exist_ok=True)
    data = load_and_compute(args.base, table_dir)
    plot_fig01_type_evolution(data, main_dir, show=args.show)
    plot_fig02_event_characteristics(data, main_dir, show=args.show)
    plot_figS04_eof_compare(data, si_dir, show=args.show)
    plot_figS05_cumulative_variance(data, si_dir, show=args.show)
    plot_figS06_cluster_metrics(data, si_dir, show=args.show)
    plot_figS06b_silhouette_profiles(data, si_dir, show=args.show)
    plot_figS07_stage_composites(data, si_dir, show=args.show)
    plot_figS08_standardized(data, si_dir, show=args.show)
    plot_figS09_climate_indices(data, si_dir, show=args.show)
    print("\nStage pattern correlations:")
    print(data["stage_corr_df"])
    print("\nStandardized clustering contingency table:")
    print(data["tab_std"])
    print("Standardized clustering ARI:", data["ari_std"])
    print(data["std_corr_df"])


if __name__ == "__main__":
    main()
