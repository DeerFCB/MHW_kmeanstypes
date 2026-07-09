#!/usr/bin/env python
# coding: utf-8
"""
Final plotting script for the revised 3.2.2--3.2.3 atmospheric forcing,
moisture-transport, and surface-forcing decomposition figures.

This script intentionally does NOT regenerate or modify the 00_01/00_02 figures.
It relies on the shared style files already in Revised_0705/src.

Outputs
-------
Main/Fig03_large_domain_atmospheric_composites.{pdf,png}
Main/Fig04_gom_surface_forcing_composites.{pdf,png}
SI/FigS10b_large_domain_z500_q850_wind_composites.{pdf,png}
Main/Fig05_large_domain_q2m_moisture_transport_pathway.{pdf,png}
Main/Fig06_surface_forcing_decomposition_cascade.{pdf,png}
SI/FigS10_gom_mfc_transport.{pdf,png}
SI/FigS11_surface_flux_component_maps.{pdf,png}
SI/FigS12_LH_decomposition_maps.{pdf,png}
SI/FigS13_humidity_gradient_partition_maps.{pdf,png}
SI/FigS14_q2m_attribution_maps.{pdf,png}
SI/FigS15_peak_aligned_time_series.{pdf,png}
Tables/table_*.csv

Run from Revised_0705, for example:
    python scripts/00_03_make_atmos_surface_forcing_figures.py --base /Users/luty8/MHW_Project
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
import matplotlib.ticker as mticker
from matplotlib.path import Path as MplPath

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

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
    TYPE_COLORS,
    CMAP_SST,
    CMAP_ANOM,
    PANEL_LABELS,
)

# -----------------------------------------------------------------------------
# Type/order convention
# -----------------------------------------------------------------------------
DISPLAY_TYPES = [1, 2, 3, 4]
# Raw cluster order in the old composite files: raw 0,1,2,3.
# Manuscript display order used by the revised 00_02 script: raw [0,2,1,3].
RAW_CLUSTER_ORDER_FOR_DISPLAY = [0, 2, 1, 3]
RAW_LABEL_TO_DISPLAY = {0: 1, 1: 3, 2: 2, 3: 4}
RAW_PLUS_ONE_TO_DISPLAY = {1: 1, 2: 3, 3: 2, 4: 4}

# Whether the q2m attribution file stores type=1..4 already in manuscript order.
# Keep False unless you later confirm the file is raw+1 order.
Q2M_TYPE_DIM_IS_RAW_PLUS_ONE = False

# -----------------------------------------------------------------------------
# Constants for LH decomposition
# -----------------------------------------------------------------------------
CLIM_START, CLIM_END = 1982, 2011
TREND_START, TREND_END = 1982, 2024
AIR_DENSITY = 1.225
LATENT_HEAT_VAP = 2.5e6
BULK_TRANSFER_COEFF = 1.2e-3
# Downward-positive LH convention.
SCALE = -(AIR_DENSITY * LATENT_HEAT_VAP * BULK_TRANSFER_COEFF)
EPS = 0.622
ALPHA_SEAWATER = 0.98

# -----------------------------------------------------------------------------
# Argument parsing and paths
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make final 3.2.2--3.2.3 forcing figures.")
    parser.add_argument("--base", type=Path, default=Path("/Users/luty8/MHW_Project"), help="MHW project root")
    parser.add_argument("--main-dir", type=Path, default=None, help="Main figure output directory")
    parser.add_argument("--si-dir", type=Path, default=None, help="SI figure output directory")
    parser.add_argument("--table-dir", type=Path, default=None, help="Table output directory")
    parser.add_argument("--skip-heavy-decomp", action="store_true", help="Skip daily ERA5 LH-decomposition calculation")
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    return parser.parse_args()


def default_paths(base: Path) -> dict[str, Path]:
    return {
        "small_comp": base / "data/core/composite_result/small_onset_clusters_allvars.nc",
        "large_comp": base / "data/core/composite_result/large_onset_clusters_allvars.nc",
        "pl_comp": base / "data/core/composite_result/pl_onset_clusters_allvars.nc",
        "mfc_comp": base / "data/processed/era5_moisture/era5_GoM_moisture_onset_composite_K4_1982-2024.nc",
        "large_moisture_comp": base / "data/processed/era5_moisture/era5_large_moisture_onset_composite_K4_1982-2024.nc",
        "era5_daily": base / "data/base/era5/era5_atm_daily_0p25_GoM_1982-2024_v1.nc",
        "events": base / "data/core/oisst/mhw_basin_detrended_1982-2024_xmhw.nc",
        "labels_old": base / "data/core/eof_kmeans/detrend_labels_K4_1982-2024.nc",
        "labels_revision": base / "data/core/eof_kmeans/detrend_labels_clustermean_EOFKMeans_K4_revision.nc",
        "q2m_attr": base / "data/processed/era5_moisture/era5_GoM_q2m_thermo_rh_type_onset_composite_1982-2024.nc",
        "timelag_dir": base / "data/interim/time_composite_peakalign",
    }

# -----------------------------------------------------------------------------
# General data helpers
# -----------------------------------------------------------------------------
def standardize_latlon(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    rename = {}
    if "lat" in ds.dims or "lat" in ds.coords:
        rename["lat"] = "latitude"
    if "lon" in ds.dims or "lon" in ds.coords:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)
    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")
    if "longitude" in ds.coords:
        ds = ds.sortby("longitude")
    return ds


def standardize_time_lat_lon(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    rename = {}
    if "date" in ds.dims or "date" in ds.coords:
        rename["date"] = "time"
    if "latitude" in ds.dims or "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.dims or "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)
    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    if "lon" in ds.coords:
        ds = ds.sortby("lon")
    return ds


def wrap_lon_to_180(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    obj = standardize_latlon(obj)
    if "longitude" in obj.coords and float(obj["longitude"].max()) > 180:
        lon_new = ((obj["longitude"] + 180) % 360) - 180
        obj = obj.assign_coords(longitude=lon_new).sortby("longitude")
    return obj


def first_existing_var(ds: xr.Dataset, candidates: list[str], *, required: bool = True) -> str | None:
    for v in candidates:
        if v in ds.data_vars:
            return v
    if required:
        raise KeyError(f"None of {candidates} found. Available variables: {list(ds.data_vars)}")
    return None


def get_var_contains(ds: xr.Dataset, token: str) -> xr.DataArray:
    if token in ds.data_vars:
        return ds[token]
    matches = [v for v in ds.data_vars if token in v]
    if len(matches) == 1:
        return ds[matches[0]]
    if len(matches) > 1:
        print(f"[WARN] Multiple matches for {token}: {matches}; using {matches[0]}")
        return ds[matches[0]]
    raise KeyError(f"Cannot find variable containing {token}. Available: {list(ds.data_vars)}")


def get_sig_name(ds: xr.Dataset, var_name: str) -> str | None:
    if var_name.endswith("_mean"):
        cand = var_name.replace("_mean", "_sig05")
        if cand in ds.data_vars:
            return cand
    # Also support names already built without _mean.
    cand = f"{var_name}_sig05"
    return cand if cand in ds.data_vars else None


def convert_cluster_to_display(obj: xr.Dataset | xr.DataArray, dim: str = "cluster") -> xr.Dataset | xr.DataArray:
    if dim not in obj.dims:
        raise ValueError(f"Expected dimension {dim!r}, got dims {obj.dims}")
    if obj.sizes[dim] != 4:
        raise ValueError(f"Expected 4 clusters along {dim!r}, got {obj.sizes[dim]}")
    out = obj.isel({dim: RAW_CLUSTER_ORDER_FOR_DISPLAY})
    out = out.assign_coords({dim: DISPLAY_TYPES}).rename({dim: "display_type"})
    return out


def convert_type_to_display(obj: xr.Dataset | xr.DataArray, dim: str = "type", assume_raw_plus_one: bool = True) -> xr.Dataset | xr.DataArray:
    if "display_type" in obj.dims:
        return obj.sel(display_type=DISPLAY_TYPES)
    vals = [int(v) for v in obj[dim].values]
    if assume_raw_plus_one:
        order = [v for v in [1, 3, 2, 4] if v in vals]
        out = obj.sel({dim: order}).assign_coords({dim: DISPLAY_TYPES}).rename({dim: "display_type"})
    else:
        out = obj.rename({dim: "display_type"}).sel(display_type=DISPLAY_TYPES)
    return out


def load_display_composite(path: Path, dim: str = "cluster") -> xr.Dataset:
    ds = xr.open_dataset(path)
    ds = standardize_latlon(ds)
    if dim == "cluster":
        return convert_cluster_to_display(ds, dim="cluster")
    if dim == "type":
        return convert_type_to_display(ds, dim="type", assume_raw_plus_one=True)
    raise ValueError(dim)


def robust_symmetric_limit(arrays, q: float = 0.98, fallback: float = 1.0) -> float:
    vals = []
    for a in arrays if isinstance(arrays, (list, tuple)) else [arrays]:
        x = np.asarray(a).ravel()
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(np.abs(x))
    if not vals:
        return fallback
    all_abs = np.concatenate(vals)
    vmax = float(np.nanpercentile(all_abs, q * 100 if q <= 1 else q))
    if (not np.isfinite(vmax)) or vmax <= 0:
        vmax = float(np.nanmax(all_abs)) if all_abs.size else fallback
    return vmax


def build_gom_polygon_mask(lon: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
    lon_vertices = [-98, -89, -87, -84.22, -82.7, -80.5, -80.5, -83, -98, -98]
    lat_vertices = [17.5, 17.5, 21.3, 22, 22.8, 22.9, 25.0, 30.5, 30.5, 17]
    if float(lon.max()) > 180:
        lon_vertices = [(x + 360) if x < 0 else x for x in lon_vertices]
    polygon = np.column_stack([lon_vertices, lat_vertices])
    path = MplPath(polygon)
    lon2d, lat2d = np.meshgrid(lon.values, lat.values)
    inside = path.contains_points(np.column_stack([lon2d.ravel(), lat2d.ravel()])).reshape(lat.size, lon.size)
    return xr.DataArray(inside, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon})


def area_weighted_mean_latlon(da: xr.DataArray, mask: xr.DataArray | None = None) -> xr.DataArray:
    da = standardize_latlon(da.to_dataset(name="tmp"))["tmp"]
    w = np.cos(np.deg2rad(da["latitude"]))
    w2 = xr.broadcast(w, da)[0]
    if mask is not None:
        mask = standardize_latlon(mask.to_dataset(name="mask"))["mask"]
        da = da.where(mask)
        w2 = w2.where(mask)
    return (da * w2).sum(("latitude", "longitude"), skipna=True) / w2.sum(("latitude", "longitude"), skipna=True)


def area_weighted_event_values_latlon(da: xr.DataArray, mask: xr.DataArray | None = None) -> xr.DataArray:
    da = standardize_latlon(da.to_dataset(name="tmp"))["tmp"]
    w = np.cos(np.deg2rad(da["latitude"]))
    w2 = xr.broadcast(w, da)[0]
    if mask is not None:
        mask = standardize_latlon(mask.to_dataset(name="mask"))["mask"]
        da = da.where(mask)
        w2 = w2.where(mask)
    return (da * w2).sum(("latitude", "longitude"), skipna=True) / w2.sum(("latitude", "longitude"), skipna=True)


def grouped_mean_sem_from_events(ds_event: xr.Dataset, var_map: dict[str, str], mask=None, group_coord="display_type"):
    mean_dict = {"type": DISPLAY_TYPES}
    sem_dict = {"type": DISPLAY_TYPES}
    count_dict = {"type": DISPLAY_TYPES}
    for out_name, var_name in var_map.items():
        vals = area_weighted_event_values_latlon(ds_event[var_name], mask=mask)
        vals = vals.assign_coords({group_coord: ds_event[group_coord]})
        grouped = vals.groupby(group_coord)
        mean = grouped.mean("events", skipna=True).sel({group_coord: DISPLAY_TYPES})
        std = grouped.std("events", skipna=True).sel({group_coord: DISPLAY_TYPES})
        count = grouped.count("events").sel({group_coord: DISPLAY_TYPES})
        sem = std / np.sqrt(count)
        mean_dict[out_name] = mean.values
        sem_dict[out_name] = sem.values
        count_dict[out_name] = count.values
    return pd.DataFrame(mean_dict).set_index("type"), pd.DataFrame(sem_dict).set_index("type"), pd.DataFrame(count_dict).set_index("type")

# -----------------------------------------------------------------------------
# Map helpers
# -----------------------------------------------------------------------------
def add_map_base(ax, *, extent, land="#d9d9d9", coast_lw=0.42, proj=ccrs.PlateCarree(), resolution="110m"):
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.LAND, facecolor=land, zorder=3)
    ax.coastlines(resolution=resolution, linewidth=coast_lw, zorder=4)


def gridlines_gom(ax, row, col, nrow, ncol, label_size=7.0):
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.28,
        color="0.6", alpha=0.40, linestyle="--",
        xlocs=mticker.FixedLocator([-95, -90, -85, -80]),
        ylocs=mticker.FixedLocator([20, 25, 30]),
    )
    gl.top_labels = False; gl.right_labels = False
    gl.left_labels = (col == 0); gl.bottom_labels = (row == nrow - 1)
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": label_size}; gl.ylabel_style = {"size": label_size}
    gl.xpadding = 1.2; gl.ypadding = 1.2
    try: gl.rotate_labels = False
    except Exception: pass
    return gl


def gridlines_large(ax, row, col, nrow, ncol, label_size=6.8):
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.28,
        color="0.6", alpha=0.40, linestyle="--", x_inline=False, y_inline=False,
    )
    gl.top_labels = False; gl.right_labels = False
    gl.left_labels = (col == 0); gl.bottom_labels = (row == nrow - 1)
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": label_size}; gl.ylabel_style = {"size": label_size}
    gl.xpadding = 1.0; gl.ypadding = 1.0
    try: gl.rotate_labels = False
    except Exception: pass
    return gl


def col_colorbar(fig, axes_2d, mappable, col: int, label: str, pad=0.040, height=0.018):
    pos = [ax.get_position() for ax in axes_2d[:, col]]
    x0 = min(p.x0 for p in pos)
    x1 = max(p.x1 for p in pos)
    y0 = min(p.y0 for p in pos)
    cax = fig.add_axes([x0 + 0.04 * (x1 - x0), y0 - pad, 0.92 * (x1 - x0), height])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", extend="both")
    cb.set_label(label, fontsize=7.0, labelpad=1.0)
    cb.ax.tick_params(labelsize=6.7, width=0.5, length=2.0)
    cb.outline.set_linewidth(0.55)
    return cb


def right_colorbar(fig, axes, mappable, label: str, x=0.91, y=0.20, h=0.60, w=0.016):
    cax = fig.add_axes([x, y, w, h])
    cb = fig.colorbar(mappable, cax=cax, orientation="vertical", extend="both")
    cb.set_label(label, fontsize=7.2)
    cb.ax.tick_params(labelsize=6.8, width=0.5, length=2.0)
    cb.outline.set_linewidth(0.55)
    return cb


# -----------------------------------------------------------------------------
# Pressure-level large-domain composites: 500 hPa Z' and 850 hPa q' with wind
# -----------------------------------------------------------------------------
def _pressure_level_symmetric_limit(da: xr.DataArray, *, round_to: float, q: float = 0.995, fallback: float = 1.0) -> float:
    vmax = robust_symmetric_limit(da.values, q=q, fallback=fallback)
    if round_to > 0:
        vmax = np.ceil(vmax / round_to) * round_to
    return float(vmax)


def plot_pressure_level_z500_q850_wind(
    ds: xr.Dataset,
    si_dir: Path,
    *,
    z_var: str = "z_resid_mean",
    q_var: str = "q_resid_mean",
    u_var: str = "u_resid_mean",
    v_var: str = "v_resid_mean",
    p500: int = 500,
    p850: int = 850,
    stride: int = 8,
    show: bool = False,
):
    """Rows = MHW type; columns = 500-hPa Z' + wind and 850-hPa q' + wind."""
    ds = standardize_latlon(ds)
    ds = convert_cluster_to_display(ds, dim="cluster") if "cluster" in ds.dims else ds
    ds = wrap_lon_to_180(ds)

    for var in [z_var, q_var, u_var, v_var]:
        if var not in ds:
            raise KeyError(f"{var!r} not found in pressure-level composite. Available variables: {list(ds.data_vars)}")
    if "isobaricInhPa" not in ds[z_var].dims:
        raise ValueError(f"{z_var!r} must have an 'isobaricInhPa' dimension.")

    lev500 = float(ds[z_var].sel(isobaricInhPa=p500, method="nearest")["isobaricInhPa"])
    lev850 = float(ds[q_var].sel(isobaricInhPa=p850, method="nearest")["isobaricInhPa"])

    Z = ds[z_var].sel(isobaricInhPa=lev500)
    q = ds[q_var].sel(isobaricInhPa=lev850)
    U500 = ds[u_var].sel(isobaricInhPa=lev500)
    V500 = ds[v_var].sel(isobaricInhPa=lev500)
    U850 = ds[u_var].sel(isobaricInhPa=lev850)
    V850 = ds[v_var].sel(isobaricInhPa=lev850)
    sigZ_name = get_sig_name(ds, z_var)
    sigq_name = get_sig_name(ds, q_var)
    sigZ = ds[sigZ_name].sel(isobaricInhPa=lev500) if sigZ_name in ds.data_vars else None
    sigq = ds[sigq_name].sel(isobaricInhPa=lev850) if sigq_name in ds.data_vars else None

    lon = ds["longitude"]
    lat = ds["latitude"]
    lon2d, lat2d = np.meshgrid(lon.values, lat.values)
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    data_crs = ccrs.PlateCarree()

    vmax_z = _pressure_level_symmetric_limit(Z, round_to=100, q=0.995, fallback=800)
    vmax_q = _pressure_level_symmetric_limit(q, round_to=0.0005, q=0.995, fallback=0.002)

    fig, axes = plt.subplots(
        4, 2, figsize=(FIG_W_FULL, 5.15),
        subplot_kw={"projection": data_crs}, constrained_layout=False,
    )
    axes = np.atleast_2d(axes)
    pcm_z = pcm_q = None
    lon_q = lon.values[::stride]
    lat_q = lat.values[::stride]
    Lon_q, Lat_q = np.meshgrid(lon_q, lat_q)

    for i, typ in enumerate(DISPLAY_TYPES):
        # 500 hPa geopotential height/geopotential anomaly + wind
        ax = axes[i, 0]
        pcm_z = ax.pcolormesh(lon, lat, Z.sel(display_type=typ), cmap="RdBu_r", vmin=-vmax_z, vmax=vmax_z,
                              shading="auto", transform=data_crs)
        if sigZ is not None:
            mask = (sigZ.sel(display_type=typ) == 1).values
            if np.any(mask):
                sparse = np.zeros_like(mask, dtype=bool)
                sparse[::4, ::4] = mask[::4, ::4]
                ax.scatter(lon2d[sparse], lat2d[sparse], s=3.2, c="k", alpha=0.18, linewidths=0,
                           transform=data_crs, zorder=4)
        qv = ax.quiver(
            Lon_q, Lat_q,
            U500.sel(display_type=typ).values[::stride, ::stride],
            V500.sel(display_type=typ).values[::stride, ::stride],
            transform=data_crs, scale=100, width=0.0033, headwidth=3.6, headlength=5.0,
            color="0.05", alpha=0.78, zorder=5,
        )
        if i == 0:
            ax.set_title(r"500 hPa: $Z^{\prime}$ + wind", fontsize=8.2, pad=3.0, fontweight="semibold")
            ax.quiverkey(qv, X=0.76, Y=1.065, U=10, label=r"10 m s$^{-1}$", labelpos="E",
                         coordinates="axes", fontproperties={"size": 6.5})
        add_map_base(ax, extent=extent, proj=data_crs, resolution="110m")
        gridlines_large(ax, i, 0, 4, 2, label_size=6.6)
        add_panel_label(ax, PANEL_LABELS[i * 2], fontsize=6.8)

        # 850 hPa near-lower-tropospheric humidity anomaly + wind
        ax = axes[i, 1]
        pcm_q = ax.pcolormesh(lon, lat, q.sel(display_type=typ), cmap="BrBG", vmin=-vmax_q, vmax=vmax_q,
                              shading="auto", transform=data_crs)
        if sigq is not None:
            mask = (sigq.sel(display_type=typ) == 1).values
            if np.any(mask):
                sparse = np.zeros_like(mask, dtype=bool)
                sparse[::4, ::4] = mask[::4, ::4]
                ax.scatter(lon2d[sparse], lat2d[sparse], s=3.2, c="k", alpha=0.18, linewidths=0,
                           transform=data_crs, zorder=4)
        qv = ax.quiver(
            Lon_q, Lat_q,
            U850.sel(display_type=typ).values[::stride, ::stride],
            V850.sel(display_type=typ).values[::stride, ::stride],
            transform=data_crs, scale=50, width=0.0030, headwidth=3.6, headlength=5.0,
            color="0.05", alpha=0.78, zorder=5,
        )
        if i == 0:
            ax.set_title(r"850 hPa: $q^{\prime}$ + wind", fontsize=8.2, pad=3.0, fontweight="semibold")
            ax.quiverkey(qv, X=0.76, Y=1.065, U=5, label=r"5 m s$^{-1}$", labelpos="E",
                         coordinates="axes", fontproperties={"size": 6.5})
        add_map_base(ax, extent=extent, proj=data_crs, resolution="110m")
        gridlines_large(ax, i, 1, 4, 2, label_size=6.6)
        add_panel_label(ax, PANEL_LABELS[i * 2 + 1], fontsize=6.8)

    fig.subplots_adjust(left=0.090, right=0.985, top=0.940, bottom=0.135, wspace=0.050, hspace=0.065)
    for i, typ in enumerate(DISPLAY_TYPES):
        pos = axes[i, 0].get_position()
        fig.text(0.035, 0.5 * (pos.y0 + pos.y1), f"Type {typ}", rotation=90,
                 ha="center", va="center", fontsize=7.8, fontweight="semibold")

    # Two compact horizontal colorbars, one under each column.
    cax1 = fig.add_axes([0.130, 0.060, 0.330, 0.020])
    cb1 = fig.colorbar(pcm_z, cax=cax1, orientation="horizontal", extend="both")
    cb1.set_label(r"$Z^{\prime}$ anomaly", fontsize=7.0, labelpad=1.0)
    cb1.ax.tick_params(labelsize=6.5, width=0.5, length=2.0)
    cb1.outline.set_linewidth(0.55)

    cax2 = fig.add_axes([0.570, 0.060, 0.330, 0.020])
    cb2 = fig.colorbar(pcm_q, cax=cax2, orientation="horizontal", extend="both")
    cb2.set_label(r"$q^{\prime}$ anomaly", fontsize=7.0, labelpad=1.0)
    cb2.ax.tick_params(labelsize=6.5, width=0.5, length=2.0)
    cb2.outline.set_linewidth(0.55)

    save_figure(fig, si_dir, "FigS10b_large_domain_z500_q850_wind_composites")
    if show:
        plt.show()
    plt.close(fig)

# -----------------------------------------------------------------------------
# Figure 3/4: ERA5 onset composites, rows = type
# -----------------------------------------------------------------------------
def ensure_qnet_onset(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    if "net_flux_resid_mean" in ds:
        return ds
    sw = first_existing_var(ds, ["avg_snswrf_resid_mean", "avg_sdswrf_resid_mean"], required=False)
    lw = first_existing_var(ds, ["avg_snlwrf_resid_mean", "avg_sdlwrf_resid_mean"], required=False)
    lh = first_existing_var(ds, ["avg_slhtf_resid_mean"], required=False)
    sh = first_existing_var(ds, ["avg_ishf_resid_mean"], required=False)
    if all(v is not None for v in [sw, lw, lh, sh]):
        ds["net_flux_resid_mean"] = ds[sw] + ds[lw] + ds[lh] + ds[sh]
    return ds


def plot_type_rows_era5_maps(
    ds: xr.Dataset,
    variables: list[str],
    titles: list[str],
    cbar_labels: list[str],
    outdir: Path,
    fig_name: str,
    *,
    large_domain: bool = False,
    vector_col: int | None = None,
    u_var: str = "u10_resid_mean",
    v_var: str = "v10_resid_mean",
    vector_stride: int = 8,
    show: bool = False,
):
    ds = standardize_latlon(ds)
    nrow, ncol = 4, len(variables)
    proj = ccrs.PlateCarree(central_longitude=180) if large_domain and float(ds["longitude"].max()) > 180 else ccrs.PlateCarree()
    data_crs = ccrs.PlateCarree()
    fig_h = 1.05 * nrow + (0.10 if large_domain else 0.25)
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIG_W_FULL, fig_h), subplot_kw={"projection": proj}, constrained_layout=False)
    axes = np.atleast_2d(axes)
    lon = ds["longitude"]
    lat = ds["latitude"]
    lon2d, lat2d = np.meshgrid(lon.values, lat.values)
    if large_domain and float(lon.max()) > 180:
        extent = [120, 359.9, float(lat.min()), float(lat.max())]
    else:
        extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    mappables = []
    for j, var in enumerate(variables):
        vmax = robust_symmetric_limit(ds[var].values, q=0.98, fallback=1.0)
        cmap = CMAP_SST if var.startswith("sst") else CMAP_ANOM
        sig_name = get_sig_name(ds, var)
        sig = ds[sig_name] if sig_name is not None else None
        mappable_last = None
        for i, typ in enumerate(DISPLAY_TYPES):
            ax = axes[i, j]
            field = ds[var].sel(display_type=typ)
            mappable_last = ax.pcolormesh(lon, lat, field, vmin=-vmax, vmax=vmax, cmap=cmap, shading="auto", transform=data_crs)
            if sig is not None:
                mask = sig.sel(display_type=typ) == 1
                if bool(mask.any()):
                    skip = 4 if large_domain else 3
                    sparse = np.zeros_like(mask.values, dtype=bool)
                    sparse[::skip, ::skip] = mask.values[::skip, ::skip]
                    ax.scatter(lon2d[sparse], lat2d[sparse], s=3.0 if large_domain else 4.0, c="k", alpha=0.13, linewidths=0, transform=data_crs, zorder=4)
            if vector_col is not None and j == vector_col and u_var in ds and v_var in ds:
                u = ds[u_var].sel(display_type=typ)
                v = ds[v_var].sel(display_type=typ)
                lon_q = lon.values[::vector_stride]
                lat_q = lat.values[::vector_stride]
                U_q = u.values[::vector_stride, ::vector_stride]
                V_q = v.values[::vector_stride, ::vector_stride]
                Lon_q, Lat_q = np.meshgrid(lon_q, lat_q)
                qv = ax.quiver(Lon_q, Lat_q, U_q, V_q, transform=data_crs, scale=20, width=0.0065, headlength=4.2, headaxislength=4.0, headwidth=3.4, color="k", alpha=0.72, zorder=5)
                if i == 0:
                    ax.quiverkey(qv, X=0.78, Y=1.05, U=1, label=r"1 m s$^{-1}$", labelpos="E", coordinates="axes", fontproperties={"size": 6.5})
            add_map_base(ax, extent=extent, proj=data_crs, resolution="110m" if large_domain else "10m")
            (gridlines_large if large_domain else gridlines_gom)(ax, i, j, nrow, ncol)
            add_panel_label(ax, PANEL_LABELS[i * ncol + j], fontsize=6.8)
            if i == 0:
                ax.set_title(titles[j], fontsize=8.2, pad=3.0, fontweight="semibold")
            if j == 0:
                ax.text(-0.18 if large_domain else -0.21, 0.5, f"Type {typ}", transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=8.0, fontweight="semibold")
        mappables.append(mappable_last)

    fig.subplots_adjust(left=0.105 if large_domain else 0.125, right=0.985, top=0.945, bottom=0.150, wspace=0.035, hspace=0.055)
    for j, (mappable, label) in enumerate(zip(mappables, cbar_labels)):
        col_colorbar(fig, axes, mappable, j, label, pad=0.043, height=0.017)
    save_figure(fig, outdir, fig_name)
    if show:
        plt.show()
    plt.close(fig)

# -----------------------------------------------------------------------------
# Moisture transport figures
# -----------------------------------------------------------------------------
def plot_large_moisture_pathway(ds: xr.Dataset, main_dir: Path, show=False):
    ds = convert_type_to_display(ds, dim="type", assume_raw_plus_one=True) if "type" in ds.dims else ds
    ds = wrap_lon_to_180(ds)
    q2m = get_var_contains(ds, "q2m_resid_type_onset_composite")
    qu = get_var_contains(ds, "qu_resid_type_onset_composite")
    qv = get_var_contains(ds, "qv_resid_type_onset_composite")
    lon = q2m["longitude"]
    lat = q2m["latitude"]
    extent = [-115, -30, -5, 50]

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W_FULL, 4.95), subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=False)
    axes = np.atleast_2d(axes)
    im = None
    for i, typ in enumerate(DISPLAY_TYPES):
        r, c = divmod(i, 2)
        ax = axes[r, c]
        q_plot = q2m.sel(display_type=typ) * 1000.0
        im = ax.pcolormesh(lon, lat, q_plot, vmin=-3, vmax=3, cmap=CMAP_ANOM, shading="auto", transform=ccrs.PlateCarree())
        U = qu.sel(display_type=typ)
        V = qv.sel(display_type=typ)
        step = 5
        lon_q = lon.values[::step]
        lat_q = lat.values[::step]
        Lon_q, Lat_q = np.meshgrid(lon_q, lat_q)
        ax.quiver(Lon_q, Lat_q, U.values[::step, ::step], V.values[::step, ::step], transform=ccrs.PlateCarree(),
                  scale=0.35, width=0.0020, headwidth=3.0, headlength=4.0, color="0.15", alpha=0.85, zorder=4)
        add_map_base(ax, extent=extent, proj=ccrs.PlateCarree(), resolution="110m")
        gridlines_large(ax, r, c, 2, 2, label_size=6.8)
        add_panel_label(ax, PANEL_LABELS[i], f"Type {typ}", fontsize=7.0)

    fig.subplots_adjust(left=0.070, right=0.885, top=0.975, bottom=0.095, wspace=0.060, hspace=0.075)
    right_colorbar(fig, axes, im, r"Near-surface specific humidity anomaly, $q_{a}^{\prime}$ (g kg$^{-1}$)", x=0.905, y=0.205, h=0.58)
    save_figure(fig, main_dir, "Fig05_large_domain_q2m_moisture_transport_pathway")
    if show:
        plt.show()
    plt.close(fig)

def plot_gom_mfc_transport(ds: xr.Dataset, si_dir: Path, show=False):
    ds = convert_type_to_display(ds, dim="type", assume_raw_plus_one=True) if "type" in ds.dims else ds
    ds = standardize_latlon(ds)
    mfc = get_var_contains(ds, "mfc_resid_gkg_day_type_onset_composite")
    qu = get_var_contains(ds, "qu_resid_type_onset_composite")
    qv = get_var_contains(ds, "qv_resid_type_onset_composite")
    lon = mfc["longitude"]
    lat = mfc["latitude"]
    vmax = 5.0
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W_FULL, 4.95), subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=False)
    axes = np.atleast_2d(axes)
    im = None
    for i, typ in enumerate(DISPLAY_TYPES):
        r, c = divmod(i, 2)
        ax = axes[r, c]
        im = ax.pcolormesh(lon, lat, mfc.sel(display_type=typ), cmap=CMAP_ANOM, vmin=-vmax, vmax=vmax, shading="auto", transform=ccrs.PlateCarree())
        step = 4
        lon_q = lon.values[::step]
        lat_q = lat.values[::step]
        Lon_q, Lat_q = np.meshgrid(lon_q, lat_q)
        ax.quiver(Lon_q, Lat_q, qu.sel(display_type=typ).values[::step, ::step], qv.sel(display_type=typ).values[::step, ::step],
                  transform=ccrs.PlateCarree(), scale=0.45, width=0.0020, headwidth=3.0, color="0.15", alpha=0.85, zorder=4)
        add_map_base(ax, extent=[-99, -78, 17.5, 31], proj=ccrs.PlateCarree(), resolution="10m")
        gridlines_gom(ax, r, c, 2, 2, label_size=6.8)
        add_panel_label(ax, PANEL_LABELS[i], f"Type {typ}", fontsize=7.0)
    fig.subplots_adjust(left=0.070, right=0.885, top=0.975, bottom=0.095, wspace=0.060, hspace=0.075)
    right_colorbar(fig, axes, im, r"Moisture-flux convergence anomaly (g kg$^{-1}$ day$^{-1}$)", x=0.905, y=0.205, h=0.58)
    save_figure(fig, si_dir, "FigS10_gom_mfc_transport")
    if show:
        plt.show()
    plt.close(fig)

# -----------------------------------------------------------------------------
# LH/surface decomposition computation
# -----------------------------------------------------------------------------
def map_event_times_to_indices(ds_time: xr.DataArray, t_start: xr.DataArray, t_peak: xr.DataArray):
    t_all = pd.to_datetime(ds_time.values)
    def nearest_index(t):
        return int(np.argmin(np.abs(t_all - pd.Timestamp(t))))
    return np.array([nearest_index(t) for t in t_start.values], dtype=int), np.array([nearest_index(t) for t in t_peak.values], dtype=int)


def es_bolton(Tc):
    return 611.2 * np.exp((17.67 * Tc) / (Tc + 243.5))


def q_from_Td_p(TdK, pPa):
    TdC = TdK - 273.15
    e = es_bolton(TdC)
    w = EPS * e / (pPa - e)
    return w / (1.0 + w)


def qsat_from_T_p(TK, pPa):
    TC = TK - 273.15
    e = es_bolton(TC)
    w = EPS * e / (pPa - e)
    return w / (1.0 + w)


def deseason_and_detrend_simple(da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Deseason and detrend daily gridded data using 1982--2011 DOY mean and 1982--2024 annual trend."""
    da = da.astype("float32").where(np.isfinite(da))
    base = da.sel(time=slice(f"{CLIM_START}-01-01", f"{CLIM_END}-12-31"))
    clim = base.groupby("time.dayofyear").mean("time", skipna=True).rename({"dayofyear": "doy"})
    clim = clim.rolling(doy=31, center=True, min_periods=1).mean()
    clim = clim.reindex(doy=np.arange(1, 367), method="nearest")
    seas = clim.sel(doy=da["time"].dt.dayofyear).assign_coords(time=da["time"])
    if "doy" in seas.coords:
        seas = seas.drop_vars("doy")
    anom = (da - seas).astype("float32")
    ann = anom.sel(time=slice(f"{TREND_START}-01-01", f"{TREND_END}-12-31")).resample(time="YS").mean(skipna=True)
    ann_mid = ann.copy()
    ann_mid["time"] = ann_mid["time"] + np.timedelta64(182, "D")
    t0 = ann_mid["time"].isel(time=0)
    t_ann = ((ann_mid["time"] - t0) / np.timedelta64(1, "D")).astype("float")
    fit_in = ann_mid.assign_coords(time=t_ann)
    poly = fit_in.polyfit(dim="time", deg=1, skipna=True)
    t_daily = ((anom["time"] - t0) / np.timedelta64(1, "D")).astype("float")
    trend = xr.polyval(t_daily, poly["polyfit_coefficients"]).astype("float32")
    resid = (anom - trend).astype("float32")
    return anom, trend, resid


def get_label_file(paths: dict[str, Path]) -> Path:
    for p in [paths["labels_old"], paths["labels_revision"]]:
        if p.exists():
            return p
    raise FileNotFoundError(f"No label file found in {paths['labels_old']} or {paths['labels_revision']}")


def load_event_labels(paths: dict[str, Path]) -> xr.Dataset:
    ev = xr.open_dataset(paths["events"])
    label_path = get_label_file(paths)
    lb = xr.open_dataset(label_path)
    if "events" not in ev.dims:
        raise ValueError(f"Expected events dimension in {paths['events']}")
    n_ev = ev.sizes["events"]
    if "event" in lb.dims:
        raw = lb["cluster_label_K4"].values[:n_ev]
    elif "events" in lb.dims:
        raw = lb["cluster_label_K4"].values[:n_ev]
    else:
        raw = np.asarray(lb["cluster_label_K4"].values).ravel()[:n_ev]
    display = np.array([RAW_LABEL_TO_DISPLAY[int(x)] for x in raw], dtype=int)
    ev = ev.assign_coords(display_type=("events", display))
    return ev


def mean_over_index_window(da: xr.DataArray, s: int, p: int):
    return da.isel(time=slice(s, p + 1)).mean("time")


def compute_decomposition(paths: dict[str, Path], table_dir: Path) -> dict:
    ds_daily = xr.open_dataset(paths["era5_daily"])
    ds_daily = xr.decode_cf(ds_daily)
    ds_daily = standardize_time_lat_lon(ds_daily)

    ev = load_event_labels(paths)
    idx_start, idx_peak = map_event_times_to_indices(ds_daily["time"], ev["time_start"], ev["time_peak"])
    n_ev = ev.sizes["events"]
    tlen = ds_daily.sizes["time"]

    sst_name = first_existing_var(ds_daily, ["sst", "sst_skin", "analysed_sst"])
    d2m_name = first_existing_var(ds_daily, ["d2m"])
    sp_name = first_existing_var(ds_daily, ["sp"])
    u_name = first_existing_var(ds_daily, ["u10"])
    v_name = first_existing_var(ds_daily, ["v10"])
    sw_name = first_existing_var(ds_daily, ["avg_snswrf", "avg_sdswrf"])
    lw_name = first_existing_var(ds_daily, ["avg_snlwrf", "avg_sdlwrf"])
    lh_name = first_existing_var(ds_daily, ["avg_slhtf"])
    sh_name = first_existing_var(ds_daily, ["avg_ishf"])

    sst = ds_daily[sst_name].astype("float32")
    d2m = ds_daily[d2m_name].astype("float32")
    sp = ds_daily[sp_name].astype("float32")
    U = ds_daily["wind_speed"].astype("float32") if "wind_speed" in ds_daily else np.hypot(ds_daily[u_name], ds_daily[v_name]).astype("float32")
    SW_daily = ds_daily[sw_name].astype("float32")
    LW_daily = ds_daily[lw_name].astype("float32")
    LH_daily = ds_daily[lh_name].astype("float32")
    SH_daily = ds_daily[sh_name].astype("float32")

    qa = q_from_Td_p(d2m, sp).astype("float32")
    qs = (qsat_from_T_p(sst, sp) * ALPHA_SEAWATER).astype("float32")
    dq = (qs - qa).astype("float32")

    _, _, U_resid = deseason_and_detrend_simple(U)
    _, _, dq_resid = deseason_and_detrend_simple(dq)
    _, _, LH_resid = deseason_and_detrend_simple(LH_daily)
    _, _, qs_resid = deseason_and_detrend_simple(qs)
    _, _, qa_resid = deseason_and_detrend_simple(qa)
    _, _, SW_resid = deseason_and_detrend_simple(SW_daily)
    _, _, LW_resid = deseason_and_detrend_simple(LW_daily)
    _, _, SH_resid = deseason_and_detrend_simple(SH_daily)

    Qnet_resid = SW_resid + LW_resid + LH_resid + SH_resid
    U_bar = (U - U_resid).astype("float32")
    dq_bar = (dq - dq_resid).astype("float32")
    QL_U = SCALE * (dq_bar * U_resid)
    QL_dq = SCALE * (U_bar * dq_resid)
    QL_NL = SCALE * (U_resid * dq_resid)
    QL_sum = QL_U + QL_dq + QL_NL
    QL_qs = SCALE * (U_bar * qs_resid)
    QL_qa = SCALE * (-U_bar * qa_resid)
    QL_qpart_sum = QL_qs + QL_qa

    event_list = []
    template = Qnet_resid.isel(time=0) * np.nan
    for i in range(n_ev):
        s = int(idx_start[i])
        p = int(idx_peak[i])
        if (s < 0) or (p < 0) or (p < s) or (p >= tlen):
            event_list.append(xr.Dataset({
                "SW_resid": template, "LW_resid": template, "LH_resid": template, "SH_resid": template,
                "Qnet_resid": template, "QL_U": template, "QL_dq": template, "QL_NL": template,
                "QL_sum": template, "QL_qs": template, "QL_qa": template, "QL_qpart_sum": template,
            }))
        else:
            event_list.append(xr.Dataset({
                "SW_resid": mean_over_index_window(SW_resid, s, p),
                "LW_resid": mean_over_index_window(LW_resid, s, p),
                "LH_resid": mean_over_index_window(LH_resid, s, p),
                "SH_resid": mean_over_index_window(SH_resid, s, p),
                "Qnet_resid": mean_over_index_window(Qnet_resid, s, p),
                "QL_U": mean_over_index_window(QL_U, s, p),
                "QL_dq": mean_over_index_window(QL_dq, s, p),
                "QL_NL": mean_over_index_window(QL_NL, s, p),
                "QL_sum": mean_over_index_window(QL_sum, s, p),
                "QL_qs": mean_over_index_window(QL_qs, s, p),
                "QL_qa": mean_over_index_window(QL_qa, s, p),
                "QL_qpart_sum": mean_over_index_window(QL_qpart_sum, s, p),
            }))

    ds_event = xr.concat(event_list, dim="events").assign_coords(display_type=("events", ev["display_type"].values))
    if "lat" in ds_event.dims or "lon" in ds_event.dims:
        ds_event = ds_event.rename({d: {"lat": "latitude", "lon": "longitude"}[d] for d in ["lat", "lon"] if d in ds_event.dims})
    ds_cluster = ds_event.groupby("display_type").mean("events").sel(display_type=DISPLAY_TYPES)
    gom_mask = build_gom_polygon_mask(ds_cluster["longitude"], ds_cluster["latitude"])
    ocean_mask = np.isfinite(ds_cluster["LH_resid"].sel(display_type=1))
    gom_ocean_mask = gom_mask & ocean_mask

    surface_terms = {"SW": "SW_resid", "LW": "LW_resid", "LH": "LH_resid", "SH": "SH_resid", "Qnet": "Qnet_resid"}
    lh_terms = {"wind": "QL_U", "humidity_gradient": "QL_dq", "nonlinear": "QL_NL", "LH_total": "LH_resid"}
    qpart_terms = {"humidity_gradient_total": "QL_dq", "sst_controlled": "QL_qs", "air_humidity_controlled": "QL_qa"}
    surface_mean, surface_sem, _ = grouped_mean_sem_from_events(ds_event, surface_terms, mask=gom_ocean_mask)
    lh_mean, lh_sem, _ = grouped_mean_sem_from_events(ds_event, lh_terms, mask=gom_ocean_mask)
    qpart_mean, qpart_sem, _ = grouped_mean_sem_from_events(ds_event, qpart_terms, mask=gom_ocean_mask)

    table_dir.mkdir(parents=True, exist_ok=True)
    surface_mean.to_csv(table_dir / "table_surface_flux_components_basin_mean.csv")
    surface_sem.to_csv(table_dir / "table_surface_flux_components_basin_sem.csv")
    lh_mean.to_csv(table_dir / "table_lh_decomposition_basin_mean.csv")
    lh_sem.to_csv(table_dir / "table_lh_decomposition_basin_sem.csv")
    qpart_mean.to_csv(table_dir / "table_humidity_gradient_partition_basin_mean.csv")
    qpart_sem.to_csv(table_dir / "table_humidity_gradient_partition_basin_sem.csv")

    return {
        "ds_event": ds_event,
        "ds_cluster": ds_cluster,
        "mask": gom_ocean_mask,
        "surface_mean": surface_mean,
        "surface_sem": surface_sem,
        "lh_mean": lh_mean,
        "lh_sem": lh_sem,
        "qpart_mean": qpart_mean,
        "qpart_sem": qpart_sem,
    }

# -----------------------------------------------------------------------------
# Decomposition map and bar plots
# -----------------------------------------------------------------------------
def plot_type_rows_map_grid(
    col_items: list[tuple[str, xr.DataArray]],
    outdir: Path,
    fig_name: str,
    cbar_label: str,
    *,
    cmap: str = CMAP_ANOM,
    q: float = 0.99,
    extent=(-100, -77, 17, 31),
    show=False,
):
    """Rows = Type 1--4, columns = component/term."""
    items = []
    for title, da in col_items:
        da = standardize_latlon(da.to_dataset(name="tmp"))["tmp"]
        items.append((title, da))
    nrow, ncol = 4, len(items)
    vmax = robust_symmetric_limit([da.values for _, da in items], q=q, fallback=1.0)
    fig_h = 1.05 * nrow + 0.25
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIG_W_FULL, fig_h), subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=False)
    axes = np.atleast_2d(axes)
    im = None
    for i, typ in enumerate(DISPLAY_TYPES):
        for j, (title, da) in enumerate(items):
            ax = axes[i, j]
            im = ax.pcolormesh(da["longitude"], da["latitude"], da.sel(display_type=typ), cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto", transform=ccrs.PlateCarree())
            add_map_base(ax, extent=extent, proj=ccrs.PlateCarree(), resolution="110m")
            gridlines_gom(ax, i, j, nrow, ncol)
            add_panel_label(ax, PANEL_LABELS[i * ncol + j], fontsize=6.8)
            if i == 0:
                ax.set_title(title, fontsize=8.0, pad=2.8, fontweight="semibold")
    fig.subplots_adjust(left=0.115, right=0.895, top=0.945, bottom=0.090, wspace=0.035, hspace=0.055)
    for i, typ in enumerate(DISPLAY_TYPES):
        pos = axes[i, 0].get_position()
        fig.text(0.055, 0.5 * (pos.y0 + pos.y1), f"Type {typ}", rotation=90,
                 ha="center", va="center", fontsize=7.8, fontweight="semibold")
    right_colorbar(fig, axes, im, cbar_label, x=0.915, y=0.20, h=0.62)
    save_figure(fig, outdir, fig_name)
    if show:
        plt.show()
    plt.close(fig)


def _get_yerr(df_sem, comp):
    if df_sem is None or comp not in df_sem:
        return None
    arr = np.asarray(df_sem.reindex(DISPLAY_TYPES)[comp].values, dtype=float)
    return None if np.all(~np.isfinite(arr)) else arr


def annotate_bar_percentages(ax, containers, totals, fontsize=5.6, min_abs_pct=5.0):
    ax.relim(); ax.autoscale_view(); ax.margins(y=0.22)
    ymin, ymax = ax.get_ylim(); dy = 0.015 * (ymax - ymin)
    for cont in containers:
        for rect, total in zip(cont, totals):
            val = rect.get_height()
            if (not np.isfinite(val)) or (not np.isfinite(total)) or abs(total) < 1e-9:
                continue
            pct = 100.0 * val / total
            if min_abs_pct is not None and abs(pct) < min_abs_pct:
                continue
            ax.text(rect.get_x() + rect.get_width() / 2, val + (dy if val >= 0 else -dy), f"{pct:.0f}%", ha="center", va="bottom" if val >= 0 else "top", fontsize=fontsize, clip_on=False)


def plot_cascade_bar_figure(df_surface, df_lh, df_qpart, df_qa, main_dir: Path, *, sem_surface=None, sem_lh=None, sem_qpart=None, sem_qa=None, show=False):
    df_surface = df_surface.reindex(DISPLAY_TYPES)
    df_lh = df_lh.reindex(DISPLAY_TYPES)
    df_qpart = df_qpart.reindex(DISPLAY_TYPES)
    df_qa = df_qa.reindex(DISPLAY_TYPES)
    if sem_surface is not None: sem_surface = sem_surface.reindex(DISPLAY_TYPES)
    if sem_lh is not None: sem_lh = sem_lh.reindex(DISPLAY_TYPES)
    if sem_qpart is not None: sem_qpart = sem_qpart.reindex(DISPLAY_TYPES)
    if sem_qa is not None: sem_qa = sem_qa.reindex(DISPLAY_TYPES)

    fig, axes = plt.subplots(4, 1, figsize=(FIG_W_FULL, 6.8), constrained_layout=False)
    x = np.arange(4)
    err_kw = dict(ecolor="0.25", elinewidth=0.6, capsize=2.0, capthick=0.6)

    panels = [
        (axes[0], ["SW", "LW", "LH", "SH"], df_surface, sem_surface, "Qnet", r"$Q_{net}^{\prime}$ components: SW, LW, LH, SH", r"W m$^{-2}$", {"SW": "#f4a261", "LW": "#457b9d", "LH": "#2a9d8f", "SH": "#8ecae6"}),
        (axes[1], ["wind", "humidity_gradient", "nonlinear"], df_lh, sem_lh, "LH_total", r"LH$^{\prime}$ decomposition: wind, humidity-gradient, nonlinear", r"W m$^{-2}$", {"wind": "#4c78a8", "humidity_gradient": "#f58518", "nonlinear": "#8da0cb"}),
        (axes[2], ["sst_controlled", "air_humidity_controlled"], df_qpart, sem_qpart, "humidity_gradient_total", "Humidity-gradient partition: SST-controlled, air-humidity-controlled", r"W m$^{-2}$", {"sst_controlled": "#e76f51", "air_humidity_controlled": "#577590"}),
        (axes[3], ["thermodynamic", "rh_moisture", "residual"], df_qa, sem_qa, "qa_total", r"$q_a^{\prime}$ attribution: thermodynamic, RH/moisture, residual", r"g kg$^{-1}$", {"thermodynamic": "#f4a261", "rh_moisture": "#2a9d8f", "residual": "#8d99ae"}),
    ]
    pretty = {
        "humidity_gradient": "Humidity-gradient", "nonlinear": "Nonlinear", "wind": "Wind",
        "sst_controlled": "SST-controlled", "air_humidity_controlled": "Air-humidity-controlled",
        "thermodynamic": "Thermodynamic", "rh_moisture": "RH / moisture", "residual": "Residual",
    }
    totals_pretty = {"Qnet": "Qnet", "LH_total": r"LH$^{\prime}$", "humidity_gradient_total": "Total humidity-gradient", "qa_total": r"$q_a^{\prime}$ total"}

    for ip, (ax, comps, df, sem, total, title, ylabel, colors) in enumerate(panels):
        width = 0.76 / max(len(comps), 1)
        offset0 = -0.5 * width * (len(comps) - 1)
        containers = []
        for k, comp in enumerate(comps):
            yerr = _get_yerr(sem, comp)
            bc = ax.bar(x + offset0 + k * width, df[comp].values, width=width, yerr=yerr, error_kw=err_kw if yerr is not None else None, color=colors[comp], label=pretty.get(comp, comp))
            containers.append(bc)
        yerr_total = _get_yerr(sem, total)
        ax.errorbar(x, df[total].values, yerr=yerr_total, fmt="ko", ms=3.8, lw=0.8, capsize=2.0, label=totals_pretty.get(total, total))
        ax.axhline(0, color="k", lw=0.75)
        if ip != 2:
            annotate_bar_percentages(ax, containers, df[total].values)
        add_panel_label(ax, PANEL_LABELS[ip], title, boxed=False, x=0.0, y=1.10, fontsize=7.6)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Type {i}" for i in DISPLAY_TYPES])
        ax.legend(frameon=False, ncol=min(5, len(comps) + 1), fontsize=6.8, loc="upper right")
        apply_grid(ax, axis="y", alpha=0.20)
        thin_spines(ax)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.970, bottom=0.060, hspace=0.32)
    save_figure(fig, main_dir, "Fig06_surface_forcing_decomposition_cascade")
    if show:
        plt.show()
    plt.close(fig)


def load_q2m_attribution(paths: dict[str, Path], table_dir: Path) -> dict:
    ds = xr.open_dataset(paths["q2m_attr"])
    ds = standardize_latlon(ds)
    qa_total_raw = get_var_contains(ds, "q2m_resid_gkg")
    qa_thermo_raw = get_var_contains(ds, "q2m_thermo_attr_gkg")
    qa_rh_raw = get_var_contains(ds, "q2m_rh_attr_gkg")
    qa_resid_raw = get_var_contains(ds, "q2m_cross_resid_gkg")
    qa_total = convert_type_to_display(qa_total_raw, dim="type", assume_raw_plus_one=Q2M_TYPE_DIM_IS_RAW_PLUS_ONE)
    qa_thermo = convert_type_to_display(qa_thermo_raw, dim="type", assume_raw_plus_one=Q2M_TYPE_DIM_IS_RAW_PLUS_ONE)
    qa_rh = convert_type_to_display(qa_rh_raw, dim="type", assume_raw_plus_one=Q2M_TYPE_DIM_IS_RAW_PLUS_ONE)
    qa_resid = convert_type_to_display(qa_resid_raw, dim="type", assume_raw_plus_one=Q2M_TYPE_DIM_IS_RAW_PLUS_ONE)
    mask = build_gom_polygon_mask(qa_total["longitude"], qa_total["latitude"])
    qa_mean = pd.DataFrame({
        "type": qa_total["display_type"].values.astype(int),
        "qa_total": area_weighted_mean_latlon(qa_total, mask).values,
        "thermodynamic": area_weighted_mean_latlon(qa_thermo, mask).values,
        "rh_moisture": area_weighted_mean_latlon(qa_rh, mask).values,
        "residual": area_weighted_mean_latlon(qa_resid, mask).values,
    }).set_index("type").reindex(DISPLAY_TYPES)
    qa_sem = pd.DataFrame(np.nan, index=DISPLAY_TYPES, columns=["qa_total", "thermodynamic", "rh_moisture", "residual"])
    qa_sem.index.name = "type"
    qa_mean.to_csv(table_dir / "table_qa_attribution_basin_mean.csv")
    qa_sem.to_csv(table_dir / "table_qa_attribution_basin_sem.csv")
    return {"qa_total": qa_total.where(mask), "thermo": qa_thermo.where(mask), "rh": qa_rh.where(mask), "resid": qa_resid.where(mask), "mean": qa_mean, "sem": qa_sem}

# -----------------------------------------------------------------------------
# Time-lag line figure
# -----------------------------------------------------------------------------
def plot_peak_aligned_timelag(paths: dict[str, Path], si_dir: Path, show=False):
    out_dir = paths["timelag_dir"]
    tag = "peakalign_lag30_30_resid"
    selected = ["sst", "d2m", "wind_speed", "avg_slhtf"]
    titles = {
        "sst": "SST anomaly",
        "d2m": "2-m dewpoint temperature anomaly",
        "wind_speed": "10-m wind speed anomaly",
        "avg_slhtf": "Latent heat flux anomaly",
    }
    fig, axes = plt.subplots(1, 4, figsize=(FIG_W_FULL, 2.15), sharex=True, constrained_layout=False)
    for j, token in enumerate(selected):
        fn = out_dir / f"ERA5_{token}_{tag}_clusterLag_ts.nc"
        if not fn.exists():
            print(f"[WARN] Missing timelag file: {fn}; skipping FigS15")
            plt.close(fig)
            return
        ds = xr.open_dataset(fn)
        var = f"{token}_resid_ts"
        if var not in ds:
            var = list(ds.data_vars)[0]
        ts = ds[var]
        if "cluster" in ts.dims and ts.sizes["cluster"] == 4:
            ts = ts.isel(cluster=RAW_CLUSTER_ORDER_FOR_DISPLAY).assign_coords(cluster=DISPLAY_TYPES)
        for typ in DISPLAY_TYPES:
            ax = axes[j]
            idx = typ if typ in ts["cluster"].values else typ - 1
            ax.plot(ts["lag"], ts.sel(cluster=idx), lw=1.35, color=TYPE_COLORS[typ], label=f"Type {typ}")
        ax.axvline(0, color="k", ls="--", lw=0.75)
        ax.set_title(titles[token], fontsize=8.0, pad=2.5)
        ax.set_xlabel("Lag (days)")
        add_panel_label(ax, PANEL_LABELS[j], fontsize=7.0, boxed=False)
        apply_grid(ax, alpha=0.20)
        thin_spines(ax)
    axes[0].set_ylabel("Anomaly")
    axes[-1].legend(frameon=False, loc="upper right", fontsize=6.6)
    fig.subplots_adjust(left=0.070, right=0.995, bottom=0.205, top=0.855, wspace=0.24)
    save_figure(fig, si_dir, "FigS15_peak_aligned_time_series")
    if show:
        plt.show()
    plt.close(fig)

# -----------------------------------------------------------------------------
# Main control
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    set_mpl_style()
    paths = default_paths(args.base)
    main_dir = args.main_dir or (REVISED_DIR / "output" / "main")
    si_dir = args.si_dir or (REVISED_DIR / "output" / "si")
    table_dir = args.table_dir or (REVISED_DIR / "output" / "tables")
    main_dir.mkdir(parents=True, exist_ok=True)
    si_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # Large and GoM ERA5 onset composite maps.
    ds_large = load_display_composite(paths["large_comp"], dim="cluster")
    plot_type_rows_era5_maps(
        ds_large,
        variables=["sst_resid_mean", "t2m_resid_mean", "wind_speed_resid_mean"],
        titles=[r"SST$^{\prime}$", r"T2m$^{\prime}$", r"WS$^{\prime}$"],
        cbar_labels=[r"SST$^{\prime}$ (°C)", r"T2m$^{\prime}$ (°C)", r"WS$^{\prime}$ (m s$^{-1}$)"],
        outdir=main_dir,
        fig_name="Fig03_large_domain_atmospheric_composites",
        large_domain=True,
        show=args.show,
    )

    if paths["pl_comp"].exists():
        plot_pressure_level_z500_q850_wind(xr.open_dataset(paths["pl_comp"]), si_dir, stride=8, show=args.show)
    else:
        print(f"[WARN] Missing pressure-level composite: {paths['pl_comp']}")

    ds_small = load_display_composite(paths["small_comp"], dim="cluster")
    ds_small = ensure_qnet_onset(ds_small)
    plot_type_rows_era5_maps(
        ds_small,
        variables=["d2m_resid_mean", "wind_speed_resid_mean", "net_flux_resid_mean", "avg_slhtf_resid_mean"],
        titles=[r"d2m$^{\prime}$", r"WS$^{\prime}$ + UV10$^{\prime}$", r"Qnet$^{\prime}$", r"LH$^{\prime}$"],
        cbar_labels=[r"d2m$^{\prime}$ (°C)", r"WS$^{\prime}$ (m s$^{-1}$)", r"Qnet$^{\prime}$ (W m$^{-2}$)", r"LH$^{\prime}$ (W m$^{-2}$)"],
        outdir=main_dir,
        fig_name="Fig04_gom_surface_forcing_composites",
        large_domain=False,
        vector_col=1,
        u_var="u10_resid_mean",
        v_var="v10_resid_mean",
        vector_stride=8,
        show=args.show,
    )

    # Moisture pathway and MFC maps.
    if paths["large_moisture_comp"].exists():
        plot_large_moisture_pathway(xr.open_dataset(paths["large_moisture_comp"]), main_dir, show=args.show)
    else:
        print(f"[WARN] Missing large moisture composite: {paths['large_moisture_comp']}")

    if paths["mfc_comp"].exists():
        plot_gom_mfc_transport(xr.open_dataset(paths["mfc_comp"]), si_dir, show=args.show)
    else:
        print(f"[WARN] Missing GoM MFC composite: {paths['mfc_comp']}")

    # Quick surface-component maps from onset composite file.
    sw_var = first_existing_var(ds_small, ["avg_snswrf_resid_mean", "avg_sdswrf_resid_mean"], required=False)
    lw_var = first_existing_var(ds_small, ["avg_snlwrf_resid_mean", "avg_sdlwrf_resid_mean"], required=False)
    lh_var = first_existing_var(ds_small, ["avg_slhtf_resid_mean"], required=False)
    sh_var = first_existing_var(ds_small, ["avg_ishf_resid_mean"], required=False)
    if all(v is not None for v in [sw_var, lw_var, lh_var, sh_var]):
        surface_items = [
            ("SW", ds_small[sw_var]), ("LW", ds_small[lw_var]), ("LH", ds_small[lh_var]), ("SH", ds_small[sh_var]), ("Sum", ds_small["net_flux_resid_mean"]),
        ]
        plot_type_rows_map_grid(surface_items, si_dir, "FigS11_surface_flux_component_maps", r"W m$^{-2}$ (downward positive)", cmap=CMAP_ANOM, q=0.99, show=args.show)

    if args.skip_heavy_decomp:
        print("Skipping daily ERA5 LH-decomposition calculation (--skip-heavy-decomp).")
        plot_peak_aligned_timelag(paths, si_dir, show=args.show)
        return

    decomp = compute_decomposition(paths, table_dir)
    ds_cluster = decomp["ds_cluster"]
    mask = decomp["mask"]
    qa = load_q2m_attribution(paths, table_dir)

    plot_type_rows_map_grid(
        [(r"$K\,\overline{\Delta q}\,U^{\prime}$", ds_cluster["QL_U"].where(mask)),
         (r"$K\,\overline{U}\,\Delta q^{\prime}$", ds_cluster["QL_dq"].where(mask)),
         (r"$K\,U^{\prime}\Delta q^{\prime}$", ds_cluster["QL_NL"].where(mask)),
         ("Sum", ds_cluster["QL_sum"].where(mask))],
        si_dir, "FigS12_LH_decomposition_maps", r"W m$^{-2}$ (downward positive)", cmap=CMAP_ANOM, q=0.99, show=args.show,
    )
    plot_type_rows_map_grid(
        [(r"$K\,\overline{U}\,\Delta q^{\prime}$", ds_cluster["QL_dq"].where(mask)),
         (r"$K\,\overline{U}\,q_s^{\prime}$", ds_cluster["QL_qs"].where(mask)),
         (r"$-K\,\overline{U}\,q_a^{\prime}$", ds_cluster["QL_qa"].where(mask)),
         ("Sum", ds_cluster["QL_qpart_sum"].where(mask))],
        si_dir, "FigS13_humidity_gradient_partition_maps", r"W m$^{-2}$ (downward positive)", cmap=CMAP_ANOM, q=0.99, show=args.show,
    )
    plot_type_rows_map_grid(
        [(r"Total $q_a^{\prime}$", qa["qa_total"]), ("Thermodynamic", qa["thermo"]), ("RH / moisture", qa["rh"]), ("Residual", qa["resid"])],
        si_dir, "FigS14_q2m_attribution_maps", r"g kg$^{-1}$", cmap=CMAP_ANOM, q=0.99, show=args.show,
    )
    plot_cascade_bar_figure(
        decomp["surface_mean"], decomp["lh_mean"], decomp["qpart_mean"], qa["mean"],
        main_dir,
        sem_surface=decomp["surface_sem"], sem_lh=decomp["lh_sem"], sem_qpart=decomp["qpart_sem"], sem_qa=qa["sem"],
        show=args.show,
    )

    plot_peak_aligned_timelag(paths, si_dir, show=args.show)


if __name__ == "__main__":
    main()
