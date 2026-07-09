#!/usr/bin/env python
# coding: utf-8
"""
Final plotting script for 00_01 OISST basin-mean preprocessing and 2023 context.

Outputs
-------
SI/FigS02_basin_mean_preprocessing.{pdf,png}
SI/FigS03_2023_context_detrended_vs_deseasoned.{pdf,png}

Run from the new code folder, for example:
    cd /Users/luty8/MHW_Project/code/projects/Kmeans_eof_2baseline/Revised_0705
    python scripts/00_01_make_detection_context_figures.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

try:
    import yaml
except Exception:
    yaml = None

warnings.filterwarnings("ignore")

# Allow running from scripts/ or project root.
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
    METHOD_COLORS,
)

try:
    from xmhw.xmhw import threshold as xmhw_threshold
    HAS_XMHW = True
except Exception:
    xmhw_threshold = None
    HAS_XMHW = False

CLIM_START = 1982
CLIM_END = 2011
TREND_START = 1982
TREND_END = 2024
YEARS = np.arange(1982, 2025)
PCTILE = 90
WINDOW_HALF_WIDTH = 5
SMOOTH_WIDTH = 31
MIN_DURATION = 5
MAX_GAP = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make final 00_01/SI detection-context figures.")
    parser.add_argument("--base", type=Path, default=Path("/Users/luty8/MHW_Project"), help="MHW project root")
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Figure output directory. Default: <base>/code/projects/Kmeans_eof_2baseline/Revised_0705/output/si",
    )
    parser.add_argument("--basin-file", type=Path, default=None, help="Optional explicit basin-mean OISST NetCDF file")
    parser.add_argument("--grid-file", type=Path, default=None, help="Optional explicit gridded GoM OISST NetCDF file used to compute basin mean if needed")
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    return parser.parse_args()


def guess_data_var(ds: xr.Dataset, preferred=("sst", "sst_mean", "sst_basin", "analysed_sst")) -> str:
    for name in preferred:
        if name in ds.data_vars:
            return name
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise ValueError(f"Could not guess variable from {list(ds.data_vars)}")


def circular_smooth_1d(values, window=31):
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values
    if window % 2 == 0:
        raise ValueError("smooth window must be odd")
    half = window // 2
    padded = np.concatenate([values[-half:], values, values[:half]])
    out = pd.Series(padded).rolling(window, center=True, min_periods=1).mean().values
    return out[half:half + len(values)]


def doy_climatology_threshold_fallback(
    da: xr.DataArray,
    climatology_period=(1982, 2011),
    pctile=90,
    window_half_width=5,
    smooth_width=31,
) -> xr.Dataset:
    start, end = climatology_period
    base = da.sel(time=slice(f"{start}-01-01", f"{end}-12-31"))
    vals = base.values.astype(float)
    doy = base["time"].dt.dayofyear.values.astype(int)
    seas = np.full(366, np.nan, dtype=float)
    thresh = np.full(366, np.nan, dtype=float)
    for d in range(1, 367):
        diff = np.abs(((doy - d + 183) % 366) - 183)
        m = diff <= window_half_width
        if np.any(m):
            seas[d - 1] = np.nanmean(vals[m])
            thresh[d - 1] = np.nanpercentile(vals[m], pctile)
    seas = pd.Series(seas).interpolate(limit_direction="both").values
    thresh = pd.Series(thresh).interpolate(limit_direction="both").values
    seas = circular_smooth_1d(seas, window=smooth_width)
    thresh = circular_smooth_1d(thresh, window=smooth_width)
    return xr.Dataset(
        {
            "seas": ("doy", seas.astype("float32")),
            "thresh": ("doy", thresh.astype("float32")),
        },
        coords={"doy": np.arange(1, 367)},
    )


def compute_threshold_dataset(da: xr.DataArray, climatology_period=(1982, 2011), pctile=90) -> xr.Dataset:
    if HAS_XMHW:
        try:
            return xmhw_threshold(da, climatologyPeriod=list(climatology_period), pctile=pctile, tdim="time")
        except Exception as exc:
            print(f"xmhw.threshold failed; using fallback. Error: {exc!r}")
    return doy_climatology_threshold_fallback(
        da,
        climatology_period=climatology_period,
        pctile=pctile,
        window_half_width=WINDOW_HALF_WIDTH,
        smooth_width=SMOOTH_WIDTH,
    )


def doy_field_on_time(doy_da: xr.DataArray, time_coord: xr.DataArray) -> xr.DataArray:
    out = doy_da.sel(doy=time_coord.dt.dayofyear).assign_coords(time=time_coord)
    if "doy" in out.coords:
        out = out.drop_vars("doy")
    return out


def fit_annual_linear_trend(anom_da: xr.DataArray, trend_start=1982, trend_end=2024):
    anom_sel = anom_da.sel(time=slice(f"{trend_start}-01-01", f"{trend_end}-12-31"))
    ann = anom_sel.resample(time="YS").mean(skipna=True)
    ann_mid = ann.copy()
    ann_mid["time"] = ann_mid["time"] + np.timedelta64(182, "D")
    t0 = ann_mid.time[0]
    t_ann_days = ((ann_mid.time - t0) / np.timedelta64(1, "D")).astype(float)
    coef = np.polyfit(t_ann_days.values, ann_mid.values, 1)
    trend_ann = xr.DataArray(
        np.polyval(coef, t_ann_days.values).astype("float32"),
        coords={"time": ann_mid.time},
        dims="time",
        name="sst_trend_ann",
    )
    t_daily_days = ((anom_da.time - t0) / np.timedelta64(1, "D")).astype(float)
    trend_daily = xr.DataArray(
        np.polyval(coef, t_daily_days.values).astype("float32"),
        coords={"time": anom_da.time},
        dims="time",
        name="sst_trend",
    )
    return ann_mid, trend_ann, trend_daily, coef


def detect_events_simple(series_da: xr.DataArray, thresh_on_time: xr.DataArray,
                         min_duration=5, max_gap=2) -> pd.DataFrame:
    s = pd.Series(series_da.values, index=pd.to_datetime(series_da.time.values)).astype(float)
    th = pd.Series(thresh_on_time.values, index=pd.to_datetime(thresh_on_time.time.values)).astype(float)
    above = (s > th).fillna(False)
    events = []
    in_evt = False
    start = None
    last_true = None
    gap = 0
    for dt, flag in above.items():
        if flag:
            if not in_evt:
                in_evt = True
                start = dt
            last_true = dt
            gap = 0
        else:
            if in_evt:
                gap += 1
                if gap > max_gap:
                    events.append((start, last_true))
                    in_evt = False
                    start = None
                    last_true = None
                    gap = 0
    if in_evt and start is not None:
        events.append((start, last_true))

    rows = []
    for start, end in events:
        duration = (end - start).days + 1
        if duration < min_duration:
            continue
        seg = s.loc[start:end]
        peak = seg.idxmax()
        rows.append({
            "event_id": len(rows),
            "time_start": pd.Timestamp(start),
            "time_peak": pd.Timestamp(peak),
            "time_end": pd.Timestamp(end),
            "duration": int(duration),
            "intensity_max": float(seg.max()),
            "intensity_mean": float(seg.mean()),
            "intensity_cumulative": float(seg.sum()),
        })
    return pd.DataFrame(rows)


def load_event_table_from_nc(path: Path) -> pd.DataFrame:
    ds = xr.open_dataset(path)
    dim = "events" if "events" in ds.dims else ("event" if "event" in ds.dims else list(ds.dims)[0])
    n = ds.sizes[dim]

    def get_time(options):
        for v in options:
            if v in ds.variables:
                return pd.to_datetime(ds[v].values)
        raise KeyError(f"Missing any of {options}")

    df = pd.DataFrame({
        "event_id": np.arange(n),
        "time_start": get_time(["time_start", "start", "date_start"]),
        "time_peak": get_time(["time_peak", "peak", "date_peak"]),
        "time_end": get_time(["time_end", "end", "date_end"]),
    })
    if "duration" in ds.variables:
        df["duration"] = np.asarray(ds["duration"].values).astype(float)
    else:
        df["duration"] = (df["time_end"] - df["time_start"]).dt.days + 1
    for v in ["intensity_max", "intensity_mean", "intensity_cumulative"]:
        df[v] = np.asarray(ds[v].values).astype(float) if v in ds.variables else np.nan
    df["start_year"] = df["time_start"].dt.year
    df["start_month"] = df["time_start"].dt.month
    return df


def annual_metrics_from_events(event_df: pd.DataFrame, years=YEARS) -> pd.DataFrame:
    rows = []
    for year in years:
        y0 = pd.Timestamp(f"{year}-01-01")
        y1 = pd.Timestamp(f"{year}-12-31")
        df_y_start = event_df[event_df["time_start"].dt.year == year]
        mhw_days = set()
        for _, r in event_df.iterrows():
            s = max(pd.Timestamp(r["time_start"]), y0)
            e = min(pd.Timestamp(r["time_end"]), y1)
            if s <= e:
                mhw_days.update(pd.date_range(s, e, freq="D"))
        rows.append({
            "year": year,
            "event_count": len(df_y_start),
            "cumulative_duration": len(mhw_days),
            "max_intensity": df_y_start["intensity_max"].max() if len(df_y_start) else np.nan,
        })
    return pd.DataFrame(rows)



def get_config_paths(base: Path) -> dict:
    """Read code/config/data.yml if present. Returns absolute Path objects where possible."""
    cfg_path = base / "code" / "config" / "data.yml"
    if yaml is None or not cfg_path.exists():
        return {}
    try:
        cfg = yaml.safe_load(cfg_path.read_text())
        files = cfg.get("files", {}).get("base", {})
        out = {}
        for key in ["oisst_basin_avg_daily_gom", "oisst_sst_daily_gom"]:
            if key in files and files[key] is not None:
                out[key] = Path(files[key]).expanduser()
        return out
    except Exception as exc:
        print(f"Warning: could not read {cfg_path}: {exc}")
        return {}


def find_first_existing(candidates, label: str) -> Path | None:
    for c in candidates:
        if c is None:
            continue
        c = Path(c).expanduser()
        if c.exists():
            print(f"Using {label}: {c}")
            return c
    return None


def compute_basin_mean_from_grid(grid_path: Path, output_path: Path) -> Path:
    """Compute a simple cosine-latitude weighted GoM basin mean from a gridded GoM OISST file."""
    print(f"Computing basin-mean OISST from gridded file: {grid_path}")
    ds = xr.open_dataset(grid_path)
    var = guess_data_var(ds, preferred=("sst", "analysed_sst", "sst_raw", "sst_mean"))
    da = ds[var].sortby("time").astype("float32")
    lat_name = "lat" if "lat" in da.coords else "latitude"
    # If the grid file is already clipped to GoM, a weighted mean over non-NaN ocean points is enough.
    weights = np.cos(np.deg2rad(da[lat_name]))
    sst_basin = da.weighted(weights).mean(dim=[d for d in da.dims if d.lower() in ("lat", "latitude", "lon", "longitude")], skipna=True)
    sst_basin.name = "sst"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sst_basin.to_dataset().to_netcdf(output_path)
    print(f"Saved computed basin-mean file: {output_path}")
    return output_path

def build_data(base: Path, basin_file: Path | None = None, grid_file: Path | None = None):
    core = base / "data/core"
    path_out = core / "oisst"
    path_out.mkdir(parents=True, exist_ok=True)

    cfg_paths = get_config_paths(base)
    cached_basin = path_out / "oisst_basin_avg_daily_gom.nc"
    old_ts_file = path_out / "oisst_basin_mean_ts_1982-2024_xmhw.nc"
    path_basin = find_first_existing(
        [
            basin_file,
            cfg_paths.get("oisst_basin_avg_daily_gom"),
            cached_basin,
            old_ts_file,
            base / "data/base/oisst/oisst_basin_avg_daily_gom.nc",
        ],
        "basin-mean OISST file",
    )

    if path_basin is None:
        path_grid = find_first_existing(
            [
                grid_file,
                cfg_paths.get("oisst_sst_daily_gom"),
                base / "data/base/oisst/oisst_sst_daily_gom.nc",
            ],
            "gridded GoM OISST file",
        )
        if path_grid is not None:
            path_basin = compute_basin_mean_from_grid(path_grid, cached_basin)

    if path_basin is None:
        raise FileNotFoundError(
            "Cannot find basin-mean OISST input. Either add the path to code/config/data.yml, "
            "or rerun with --basin-file /path/to/oisst_basin_avg_daily_gom.nc, "
            "or provide --grid-file /path/to/oisst_sst_daily_gom.nc."
        )

    out_event_file = path_out / "mhw_basin_detrended_1982-2024_xmhw.nc"

    ds_basin = xr.open_dataset(path_basin)
    var = guess_data_var(ds_basin, preferred=("sst", "sst_raw", "sst_mean", "sst_basin"))
    sst_raw = ds_basin[var].sortby("time").astype("float32")
    sst_raw.name = "sst_raw"

    clim_raw = compute_threshold_dataset(sst_raw, climatology_period=(CLIM_START, CLIM_END), pctile=PCTILE)
    seas_on_time = doy_field_on_time(clim_raw["seas"], sst_raw["time"])
    sst_anom = (sst_raw - seas_on_time).astype("float32")
    sst_anom.name = "sst_anom"

    sst_ann_mid, trend_ann, sst_trend, coef = fit_annual_linear_trend(
        sst_anom, trend_start=TREND_START, trend_end=TREND_END
    )
    sst_resid = (sst_anom - sst_trend).astype("float32")
    sst_resid.name = "sst_resid"

    clim_resid = compute_threshold_dataset(sst_resid, climatology_period=(CLIM_START, CLIM_END), pctile=PCTILE)
    thr_resid = doy_field_on_time(clim_resid["thresh"], sst_resid["time"])
    clim_anom = compute_threshold_dataset(sst_anom, climatology_period=(CLIM_START, CLIM_END), pctile=PCTILE)
    thr_anom = doy_field_on_time(clim_anom["thresh"], sst_anom["time"])

    if out_event_file.exists():
        event_df_det = load_event_table_from_nc(out_event_file)
        print(f"Using existing detrended MHW event file: {out_event_file}")
    else:
        event_df_det = detect_events_simple(sst_resid, thr_resid, min_duration=MIN_DURATION, max_gap=MAX_GAP)
        print("Existing detrended event file not found; using fallback detector.")
    event_df_anom = detect_events_simple(sst_anom, thr_anom, min_duration=MIN_DURATION, max_gap=MAX_GAP)

    annual_det = annual_metrics_from_events(event_df_det, years=YEARS)
    annual_anom = annual_metrics_from_events(event_df_anom, years=YEARS)

    # Save small processed files for later reuse / reproducibility.
    xr.Dataset({
        "sst_raw": sst_raw,
        "sst_anom": sst_anom,
        "sst_trend": sst_trend,
        "sst_resid": sst_resid,
        "sst_thresh_resid": clim_resid["thresh"],
    }).to_netcdf(path_out / "oisst_basin_mean_ts_1982-2024_xmhw.nc")
    event_df_det.to_csv(path_out / "mhw_basin_detrended_event_table_for_context.csv", index=False)
    event_df_anom.to_csv(path_out / "mhw_basin_deseasoned_only_event_table_simple_context.csv", index=False)
    annual_det.to_csv(path_out / "mhw_annual_metrics_detrended_1982-2024.csv", index=False)
    annual_anom.to_csv(path_out / "mhw_annual_metrics_deseasoned_only_simple_1982-2024.csv", index=False)

    return {
        "sst_raw": sst_raw,
        "sst_anom": sst_anom,
        "sst_ann_mid": sst_ann_mid,
        "trend_ann": trend_ann,
        "sst_resid": sst_resid,
        "thr_resid": thr_resid,
        "thr_anom": thr_anom,
        "event_df_det": event_df_det,
        "event_df_anom": event_df_anom,
        "annual_det": annual_det,
        "annual_anom": annual_anom,
    }


def plot_preprocessing(data: dict, fig_dir: Path, show=False):
    sst_raw = data["sst_raw"]
    sst_anom = data["sst_anom"]
    sst_ann_mid = data["sst_ann_mid"]
    trend_ann = data["trend_ann"]
    sst_resid = data["sst_resid"]

    fig = plt.figure(figsize=(FIG_W_FULL, 4.10))
    gs = fig.add_gridspec(2, 2, hspace=0.26, wspace=0.26)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(sst_raw["time"].values, sst_raw.values, color=METHOD_COLORS["daily_residual"], lw=0.52)
    add_panel_label(ax1, "(a)", "GoM basin-mean SST", x=0.02, y=0.97)
    ax1.set_xlabel("")
    ax1.set_ylabel("SST (°C)")
    apply_grid(ax1, alpha=0.20)
    thin_spines(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    vals = sst_raw.values[np.isfinite(sst_raw.values)]
    ax2.hist(vals, bins=50, color=METHOD_COLORS["daily_residual"], alpha=0.72, edgecolor="black", lw=0.28)
    add_panel_label(ax2, "(b)", "Distribution of raw SST", x=0.02, y=0.97)
    ax2.set_xlabel("SST (°C)")
    ax2.set_ylabel("Frequency")
    apply_grid(ax2, alpha=0.16)
    thin_spines(ax2)

    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3.plot(sst_anom["time"].values, sst_anom.values, color=METHOD_COLORS["daily_anomaly"], alpha=0.42, lw=0.48, label="Daily anomaly")
    ax3.plot(sst_ann_mid["year"].values if "year" in sst_ann_mid.coords else sst_ann_mid["time"].values, sst_ann_mid.values,
             color=METHOD_COLORS["annual_mean"], marker="o", ms=2.2, lw=0.65, alpha=0.75, label="Annual mean anomaly")
    ax3.plot(trend_ann["year"].values if "year" in trend_ann.coords else trend_ann["time"].values, trend_ann.values,
             color=METHOD_COLORS["trend"], lw=1.65, alpha=0.88, label="Annual trend fit")
    ax3.plot(sst_resid["time"].values, sst_resid.values, color=METHOD_COLORS["daily_residual"], lw=0.48, alpha=0.66, label="Daily residual")
    ax3.axhline(0, color="0.45", lw=0.65)
    add_panel_label(ax3, "(c)", "SST anomaly, trend, and detrended residual", x=0.02, y=0.97)
    ax3.set_xlabel("Year")
    ax3.set_ylabel("SST anomaly / residual (°C)")
    ax3.legend(ncol=2, fontsize=6.2, frameon=True, loc="lower center", bbox_to_anchor=(0.5, 0.03))
    apply_grid(ax3, alpha=0.20)
    thin_spines(ax3)

    ax4 = fig.add_subplot(gs[1, 1])
    vals_anom = sst_anom.values[np.isfinite(sst_anom.values)]
    vals_resid = sst_resid.values[np.isfinite(sst_resid.values)]
    ax4.hist(vals_anom, bins=50, color=METHOD_COLORS["daily_anomaly"], alpha=0.52, edgecolor="black", lw=0.25, label="Daily anomaly")
    ax4.hist(vals_resid, bins=50, color=METHOD_COLORS["daily_residual"], alpha=0.48, edgecolor="black", lw=0.25, label="Daily residual")
    add_panel_label(ax4, "(d)", "Distribution of anomaly and residual", x=0.02, y=0.97)
    ax4.set_xlabel("SST anomaly / residual (°C)")
    ax4.set_ylabel("Frequency")
    ax4.legend(fontsize=6.8, frameon=True)
    apply_grid(ax4, alpha=0.16)
    thin_spines(ax4)

    save_figure(fig, fig_dir, "FigS02_basin_mean_preprocessing")
    if show:
        plt.show()
    plt.close(fig)


def plot_annual_metric_dual(ax, df_det, df_anom, ycol, ylabel, title, highlight_year=2023):
    ax.plot(df_det["year"], df_det[ycol], marker="o", ms=3.0, lw=0.95,
            color=METHOD_COLORS["daily_residual"], label="Detrended residual")
    ax.plot(df_anom["year"], df_anom[ycol], marker="s", ms=2.7, lw=0.90,
            color=METHOD_COLORS["daily_anomaly"], alpha=0.92, label="Deseasoned only")
    for df, marker, color, size in [
        (df_det, "o", METHOD_COLORS["daily_residual"], 36),
        (df_anom, "s", METHOD_COLORS["daily_anomaly"], 30),
    ]:
        if highlight_year in df["year"].values:
            v = float(df.loc[df["year"] == highlight_year, ycol].iloc[0])
            ax.scatter([highlight_year], [v], marker=marker, s=size, color=color,
                       edgecolor="white", linewidth=0.7, zorder=5)
    ax.axvline(highlight_year, color="0.35", lw=0.75, ls="--")
    ax.text(highlight_year + 0.25, ax.get_ylim()[0]+0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "2023", ha="left", va="top", fontsize=6.4, color="0.35")
    add_panel_label(ax, title.split()[0], " ".join(title.split()[1:]), boxed=True, x=0.02, y=0.97, fontsize=7.2)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.08 * (ymax - ymin))
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    apply_grid(ax, alpha=0.18)
    thin_spines(ax)


def plot_2023_context(data: dict, fig_dir: Path, show=False):
    year_focus = 2023
    y0 = pd.Timestamp(f"{year_focus}-01-01")
    y1 = pd.Timestamp(f"{year_focus}-12-31")

    sst_resid = data["sst_resid"]
    sst_anom = data["sst_anom"]
    thr_resid = data["thr_resid"]
    thr_anom = data["thr_anom"]
    event_df_det = data["event_df_det"]
    event_df_anom = data["event_df_anom"]
    annual_det = data["annual_det"]
    annual_anom = data["annual_anom"]

    sst_2023_resid = sst_resid.sel(time=slice(y0, y1))
    thr_2023_resid = thr_resid.sel(time=slice(y0, y1))
    sst_2023_anom = sst_anom.sel(time=slice(y0, y1))
    thr_2023_anom = thr_anom.sel(time=slice(y0, y1))

    events_2023_det = event_df_det[(event_df_det["time_end"] >= y0) & (event_df_det["time_start"] <= y1)].copy()
    events_2023_anom = event_df_anom[(event_df_anom["time_end"] >= y0) & (event_df_anom["time_start"] <= y1)].copy()

    fig = plt.figure(figsize=(FIG_W_FULL, 4.65))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1.0], hspace=0.31, wspace=0.33)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(sst_2023_resid["time"], sst_2023_resid, color=METHOD_COLORS["detrended"], lw=1.05,
            label="Daily SST residual (detrended)")
    ax.plot(thr_2023_resid["time"], thr_2023_resid, color=METHOD_COLORS["detrended_threshold"], lw=1.15,
            label="90th percentile threshold (detrended)")
    ax.plot(sst_2023_anom["time"], sst_2023_anom, color=METHOD_COLORS["deseasoned"], lw=0.85, alpha=0.62,
            label="Daily SST anomaly (deseasoned only)")
    ax.plot(thr_2023_anom["time"], thr_2023_anom, color=METHOD_COLORS["deseasoned_threshold"], lw=1.0, alpha=0.92,
            label="90th percentile threshold (deseasoned only)")
    ax.axhline(0, color="0.55", lw=0.65)

    for _, r in events_2023_det.iterrows():
        s = max(pd.Timestamp(r["time_start"]), y0)
        e = min(pd.Timestamp(r["time_end"]), y1)
        ax.axvspan(s, e, color=METHOD_COLORS["event_shade"], alpha=0.16, lw=0)
        ts = pd.Timestamp(r["time_start"])
        if y0 <= ts <= y1:
            ax.axvline(ts, color=METHOD_COLORS["event_shade"], lw=0.70, alpha=0.85)
    for _, r in events_2023_anom.iterrows():
        ts = pd.Timestamp(r["time_start"])
        if y0 <= ts <= y1:
            ax.axvline(ts, color=METHOD_COLORS["deseasoned"], lw=0.60, alpha=0.32, ls="--")

    add_panel_label(ax, "(a)", "2023 basin-mean MHW context", boxed=True, x=0.006, y=0.97, fontsize=7.6)
    ax.set_ylabel("SST anomaly / residual (°C)")
    ax.set_xlabel("")
    apply_grid(ax, alpha=0.18)
    ax.legend(loc="upper right", fontsize=6.3, frameon=True, ncol=2)
    thin_spines(ax)

    ax_b = fig.add_subplot(gs[1, 0])
    plot_annual_metric_dual(ax_b, annual_det, annual_anom, "event_count", "Events yr$^{-1}$", "(b) Annual event count")
    

    ax_c = fig.add_subplot(gs[1, 1])
    plot_annual_metric_dual(ax_c, annual_det, annual_anom, "cumulative_duration", "MHW days yr$^{-1}$", "(c) Annual cumulative duration")
    ax_c.legend(fontsize=6.4, frameon=True, loc="upper left", bbox_to_anchor=(0.02, 0.85))

    ax_d = fig.add_subplot(gs[1, 2])
    plot_annual_metric_dual(ax_d, annual_det, annual_anom, "max_intensity", "Max intensity (°C)", "(d) Annual maximum intensity")

    save_figure(fig, fig_dir, "FigS03_2023_context_detrended_vs_deseasoned")
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    set_mpl_style()
    fig_dir = args.fig_dir or (REVISED_DIR / "output" / "si")
    data = build_data(args.base, basin_file=args.basin_file, grid_file=args.grid_file)
    plot_preprocessing(data, fig_dir, show=args.show)
    plot_2023_context(data, fig_dir, show=args.show)


if __name__ == "__main__":
    main()
