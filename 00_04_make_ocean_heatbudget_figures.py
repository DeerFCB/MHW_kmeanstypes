#!/usr/bin/env python3
"""
Create revised ocean-state, GLORYS vertical-structure, and mixed-layer heat-budget figures.

This script is intentionally separate from 00_01, 00_02, and 00_03.
It only writes new outputs under Revised_0705/output/main, output/si, and output/tables.

Expected local inputs, when available:
  data/core/pre_onset/glorys_ocean_state_preonset_type_composite_minus20to1_GoM_1993-2024.nc
  data/core/pre_onset/local_regression_lead_windows_peak_pm3_SSTA_against_preonset_ocean_state.nc
  data/core/pre_onset/local_regression_lead_window_summary_peak_pm3.csv
  data/processed/glorys_event_profiles_shelf_deep_ssh_onset_minus_pre.nc
  data/processed/glorys_mlhb_event_budget/event_mlhb_budget_whole_core_noncore_polygon_oisstcore_sst0p8_1993-2024.csv
  data/processed/glorys_mlhb_event_budget/mlhb_type_mean_contribution_maps_polygon_oisstcore_sst0p8_1993-2024.nc
  data/core/composite_result/raw_onset_resid_allvars.nc

Figures created if inputs exist:
  main/Fig07_preonset_ocean_state_type_composite
  main/Fig08_preonset_ocean_state_correlation_summary
  main/Fig09_glorys_vertical_warming_profiles
  main/Fig10_mixed_layer_heat_budget_whole_gom
  si/FigS16_seasonal_type_mean_vertical_structure
  si/FigS17_zonal_sections_potential_temperature_anomaly_onset
  si/FigS18_shelf_deep_onset_temperature_profiles
  si/FigS19_vertical_warming_quantitative_summary
  si/FigS20_mixed_layer_heat_budget_contribution_maps
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False
    ccrs = None
    cfeature = None
    LONGITUDE_FORMATTER = None
    LATITUDE_FORMATTER = None

# -----------------------------------------------------------------------------
# Shared style
# -----------------------------------------------------------------------------
FIG_W_FULL = 7.2
FIG_W_HALF = 3.5
DPI = 600

TYPE_LIST = [1, 2, 3, 4]
TYPE_COLORS = {
    1: "#0072B2",   # blue
    2: "#56B4E9",   # sky blue
    3: "#E69F00",   # orange
    4: "#D55E00",   # vermillion
}
COLOR_SHELF = "#4D4D4D"
COLOR_DEEP = "#E69F00"
COLOR_SSH_MINUS = "#0072B2"
COLOR_SSH_NEUTRAL = "0.45"
COLOR_SSH_PLUS = "#D55E00"

CMAP_ANOM = "RdBu_r"
CMAP_TEMP = "RdYlBu_r"
CMAP_MLD = "RdBu_r"
LAND_COLOR = "0.86"
PANEL_LABELS = [f"({chr(97+i)})" for i in range(80)]

# GoM section mask used by the original notebook.  Keep this here so the
# section figure does not show non-GoM longitudes or spurious blank columns.
GOM_LON_VERTICES = np.array([-98, -89, -87, -84.22, -82.7, -80.5, -80.5, -83, -98, -98], dtype=float)
GOM_LAT_VERTICES = np.array([17.5, 17.5, 21.3, 22.0, 22.8, 22.9, 25.0, 30.5, 30.5, 17.0], dtype=float)
SECTION_LON_RANGE = (-98.0, -80.0)
SECTION_DEPTH_RANGE = (0.0, 150.0)


def format_lon_w(x, pos=None):
    if not np.isfinite(x):
        return ""
    return f"{abs(int(round(x)))}°W" if x < 0 else f"{int(round(x))}°E"


def gom_lon_mask_for_lat(lon, lat0):
    """Return a 1-D boolean mask for lon points inside the GoM polygon at one latitude.

    This reproduces the regionmask step used in the notebook, but avoids adding a
    hard dependency here.  Points outside the polygon are removed before plotting
    longitude-depth sections.
    """
    from matplotlib.path import Path as MplPath
    pts = np.column_stack([np.asarray(lon, dtype=float), np.full(len(lon), float(lat0))])
    poly = np.column_stack([GOM_LON_VERTICES, GOM_LAT_VERTICES])
    return MplPath(poly).contains_points(pts, radius=1e-9)


def infer_section_bottom_depth(section_da):
    """Infer a bathymetric envelope from NaNs in a lon-depth section.

    GLORYS values below the seafloor are normally missing.  The deepest finite
    value at each longitude therefore provides a clean bottom mask for the section
    panels without requiring a separate bathymetry file.
    """
    if "type_display" in section_da.dims:
        arr = section_da.transpose("type_display", "depth", "longitude").values
        finite = np.isfinite(arr).any(axis=0)
    elif "cluster" in section_da.dims:
        arr = section_da.transpose("cluster", "depth", "longitude").values
        finite = np.isfinite(arr).any(axis=0)
    else:
        finite = np.isfinite(section_da.transpose("depth", "longitude").values)
    depths = np.asarray(section_da["depth"].values, dtype=float)
    bottom = np.full(finite.shape[1], np.nan)
    for j in range(finite.shape[1]):
        ok = finite[:, j]
        if ok.any():
            bottom[j] = np.nanmax(depths[ok])
    return bottom


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": DPI,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".png"]:
        path = outdir / f"{name}{suffix}"
        if suffix == ".png":
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        print("Saved:", path)


def add_panel_label(ax, label: str, title: str | None = None, x=0.02, y=0.98, fontsize=7.2, boxed=True):
    text = label if title is None else f"{label} {title}"
    bbox = dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.78) if boxed else None
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=fontsize, fontweight="bold", bbox=bbox, zorder=20)


def thin_spines(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(0.65)


def apply_grid(ax, axis="both", alpha=0.22):
    ax.grid(True, axis=axis, lw=0.35, alpha=alpha)


def standardize_latlon(ds: xr.Dataset | xr.DataArray):
    rename = {}
    for old in ["lon", "nav_lon"]:
        if old in ds.coords and "longitude" not in ds.coords:
            rename[old] = "longitude"
    for old in ["lat", "nav_lat"]:
        if old in ds.coords and "latitude" not in ds.coords:
            rename[old] = "latitude"
    if rename:
        ds = ds.rename(rename)
    if "longitude" in ds.coords and float(ds["longitude"].max()) > 180:
        ds = ds.assign_coords(longitude=((ds["longitude"] + 180) % 360) - 180).sortby("longitude")
    return ds


def convert_type_coord(ds: xr.Dataset | xr.DataArray):
    if "type_display" in ds.coords or "type_display" in ds.dims:
        return ds
    for cand in ["display_type", "type", "cluster"]:
        if cand in ds.coords or cand in ds.dims:
            vals = ds[cand].values
            vals = np.asarray(vals)
            if np.nanmin(vals) == 0:
                vals = vals + 1
            return ds.assign_coords(type_display=(cand, vals.astype(int))).swap_dims({cand: "type_display"}) if cand in ds.dims else ds.assign_coords(type_display=(cand, vals.astype(int)))
    return ds


def add_gom_map(ax, extent=(-98, -80, 17, 31), row=0, col=0, nrow=1, ncol=1, label_size=6.7):
    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        ax.set_extent(extent, crs=proj)
        ax.coastlines(resolution="10m", linewidth=0.45, zorder=5)
        ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor="none", zorder=3)
        gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.28, color="0.60", alpha=0.35, linestyle="--",
                          xlocs=mticker.FixedLocator([-95, -90, -85, -80]),
                          ylocs=mticker.FixedLocator([20, 25, 30]))
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = (col == 0)
        gl.bottom_labels = (row == nrow - 1)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {"size": label_size}
        gl.ylabel_style = {"size": label_size}
        gl.xpadding = 1.2
        gl.ypadding = 1.2
        try:
            gl.rotate_labels = False
        except Exception:
            pass
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        if col == 0:
            ax.set_ylabel("Latitude")
        if row == nrow - 1:
            ax.set_xlabel("Longitude")
    thin_spines(ax)


def pmesh_map(ax, da, *, vmin=None, vmax=None, cmap=CMAP_ANOM, levels=None, extent=(-98, -80, 17, 31), row=0, col=0, nrow=1, ncol=1):
    da = standardize_latlon(da)
    lon = da["longitude"]
    lat = da["latitude"]
    if HAS_CARTOPY:
        tr = ccrs.PlateCarree()
        if levels is not None:
            im = ax.contourf(lon, lat, da, levels=levels, cmap=cmap, extend="both", transform=tr)
        else:
            im = ax.pcolormesh(lon, lat, da, vmin=vmin, vmax=vmax, cmap=cmap, shading="auto", transform=tr)
    else:
        if levels is not None:
            im = ax.contourf(lon, lat, da, levels=levels, cmap=cmap, extend="both")
        else:
            im = ax.pcolormesh(lon, lat, da, vmin=vmin, vmax=vmax, cmap=cmap, shading="auto")
    add_gom_map(ax, extent=extent, row=row, col=col, nrow=nrow, ncol=ncol)
    return im


def right_colorbar(fig, axes, im, label, x=0.905, y=0.20, h=0.58, w=0.016):
    cax = fig.add_axes([x, y, w, h])
    cb = fig.colorbar(im, cax=cax, orientation="vertical", extend="both")
    cb.set_label(label, fontsize=7.6)
    cb.ax.tick_params(labelsize=6.8, width=0.55, length=2.2)
    cb.outline.set_linewidth(0.6)
    return cb


def robust_symmetric_limits(da, q=0.98, fallback=1.0):
    vals = np.asarray(da.values)
    vmax = float(np.nanquantile(np.abs(vals), q)) if np.isfinite(vals).any() else fallback
    if (not np.isfinite(vmax)) or vmax == 0:
        vmax = fallback
    return -vmax, vmax


def open_optional(path: Path, strict=False):
    if not path.exists():
        msg = f"Missing input, skipping related figure: {path}"
        if strict:
            raise FileNotFoundError(msg)
        print("[SKIP]", msg)
        return None
    return xr.open_dataset(path)

# -----------------------------------------------------------------------------
# Fig07: pre-onset ocean state type composite
# -----------------------------------------------------------------------------

def plot_fig07_preonset_ocean_state(ds_type: xr.Dataset, main_dir: Path):
    """Pre-onset ocean state maps.

    Rows are MHW types and columns are ocean-state variables.  This version uses
    a more compact figure width/height ratio because a 4 x 3 Cartopy layout
    otherwise leaves large horizontal gaps when forced into the full manuscript
    width.
    """
    ds_type = convert_type_coord(standardize_latlon(ds_type))
    specs = [
        ("MLD_resid_pre", r"Pre-onset MLD$'$", "m", (-20, 20), CMAP_MLD, np.arange(-20, 20.1, 5)),
        ("SSH_resid_pre", r"Pre-onset SSH$'$", "m", (-0.15, 0.15), CMAP_ANOM, np.arange(-0.15, 0.151, 0.05)),
        ("T100_resid_pre", r"Pre-onset $T_{0-100}'$", "°C", (-0.8, 0.8), CMAP_TEMP, np.arange(-0.8, 0.801, 0.2)),
    ]
    nrow, ncol = 4, len(specs)

    # The compact width avoids the very large empty spaces that appear when only
    # three Cartopy map columns are stretched across FIG_W_FULL.
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(6.35, 6.25),
        subplot_kw={"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {},
        constrained_layout=False,
    )
    axes = np.atleast_2d(axes)
    im_by_col = [None] * ncol

    for i, typ in enumerate(TYPE_LIST):
        for j, (var, col_title, unit, (vmin, vmax), cmap, levels) in enumerate(specs):
            if var not in ds_type:
                raise KeyError(f"{var} not in pre-onset type dataset")
            ax = axes[i, j]
            da = ds_type[var].sel(type_display=typ)
            im = pmesh_map(
                ax, da, vmin=vmin, vmax=vmax, cmap=cmap,
                extent=(-98, -80, 17, 31), row=i, col=j, nrow=nrow, ncol=ncol,
            )
            im_by_col[j] = im
            if i == 0:
                ax.set_title(col_title, fontsize=8.0, pad=3.0, fontweight="semibold")
            add_panel_label(ax, PANEL_LABELS[i*ncol + j], fontsize=6.6)

    # Compact spacing: maps are close together, with enough left margin for Type labels.
    fig.subplots_adjust(left=0.118, right=0.985, top=0.940, bottom=0.138, wspace=0.030, hspace=0.050)

    # Keep the row labels close to the first-column maps.  The gap is controlled by
    # left - x_text; here it is ~0.07 in figure coordinates.
    for i, typ in enumerate(TYPE_LIST):
        pos = axes[i, 0].get_position()
        fig.text(0.052, 0.5 * (pos.y0 + pos.y1), f"Type {typ}", rotation=90,
                 ha="center", va="center", fontsize=7.5, fontweight="semibold")

    # One compact horizontal colorbar per variable column.
    for j, (_, _col_title, unit, _vrange, _cmap, _levels) in enumerate(specs):
        pos = axes[-1, j].get_position()
        cax = fig.add_axes([pos.x0 + 0.05 * pos.width, 0.070, 0.90 * pos.width, 0.012])
        cb = fig.colorbar(im_by_col[j], cax=cax, orientation="horizontal", extend="both")
        cb.set_label(unit, fontsize=6.8, labelpad=1.2)
        cb.ax.tick_params(labelsize=6.2, width=0.5, length=2)
        cb.outline.set_linewidth(0.6)

    save_figure(fig, main_dir, "Fig07_preonset_ocean_state_type_composite")
    plt.close(fig)

# -----------------------------------------------------------------------------
# Fig08 and FigS11: pre-onset lead regression
# -----------------------------------------------------------------------------

def window_suffix(a: int, b: int) -> str:
    def s(v):
        return f"minus{abs(v)}" if v < 0 else f"plus{v}"
    return f"{s(a)}to{abs(b) if b < 0 else b}"


def parse_window_suffix(s):
    s = str(s)
    m = re.match(r"minus(\d+)to(\d+)", s)
    if m:
        return -int(m.group(1)), -int(m.group(2))
    nums = re.findall(r"-?\d+", s)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    raise ValueError(f"Cannot parse window from {s}")

PREDICTORS = [("SSH", r"Pre-onset SSH$'$"), ("MLD", r"Pre-onset MLD$'$"), ("T100", r"Pre-onset $T_{0-100}'$")]


def plot_corr_map_panel(ax, ds_reg, suffix, pred, label, title=None, rlim=0.6, row=0, col=0, nrow=1, ncol=1):
    rname = f"{suffix}_{pred}_r_pooled"
    pname = f"{suffix}_{pred}_p_pooled"
    if rname not in ds_reg:
        # tolerate old variable naming
        rname = f"{suffix}_{pred}_r"
    if rname not in ds_reg:
        raise KeyError(f"{rname} not found in regression dataset")
    rmap = standardize_latlon(ds_reg[rname])
    pmap = standardize_latlon(ds_reg[pname]) if pname in ds_reg else xr.full_like(rmap, np.nan)
    im = pmesh_map(ax, rmap, vmin=-rlim, vmax=rlim, cmap=CMAP_ANOM, extent=(-98, -80, 17, 31), row=row, col=col, nrow=nrow, ncol=ncol)
    sig = (pmap < 0.05).values
    lon = rmap["longitude"].values
    lat = rmap["latitude"].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    sparse = np.zeros_like(sig, dtype=bool)
    sparse[::2, ::2] = sig[::2, ::2]
    if HAS_CARTOPY:
        ax.scatter(lon2d[sparse], lat2d[sparse], s=2.4, c="k", alpha=0.18, linewidths=0, transform=ccrs.PlateCarree(), zorder=6)
    else:
        ax.scatter(lon2d[sparse], lat2d[sparse], s=2.4, c="k", alpha=0.18, linewidths=0, zorder=6)
    add_panel_label(ax, label, title=title, fontsize=6.7)
    return im


def read_lead_summary(path: Path):
    df = pd.read_csv(path)
    if "lead_start" not in df.columns or "lead_end" not in df.columns:
        win_col = "window_suffix" if "window_suffix" in df.columns else "window"
        parsed = df[win_col].apply(parse_window_suffix)
        df["lead_start"] = [p[0] for p in parsed]
        df["lead_end"] = [p[1] for p in parsed]
    if "lead_center" not in df.columns:
        df["lead_center"] = 0.5 * (df["lead_start"].astype(float) + df["lead_end"].astype(float))
    df["predictor"] = df["predictor"].replace({
        "SSH_resid_pre": "SSH", "MLD_resid_pre": "MLD", "T100_resid_pre": "T100",
        "SSH_resid": "SSH", "MLD_resid": "MLD", "T100_resid": "T100",
    })
    if "plot_kind" in df.columns:
        df = df[df["plot_kind"] == "pooled"].copy()
    return df.sort_values(["predictor", "lead_center"])


def plot_fig08_correlation_summary(ds_reg: xr.Dataset, summary_csv: Path, main_dir: Path):
    df = read_lead_summary(summary_csv)
    pred_label = {"SSH": "SSH′", "MLD": "MLD′", "T$_{0-100}'$": "T$_{0-100}'$", "T100": "T$_{0-100}'$"}
    latest = df.sort_values("lead_center")["lead_center"].max()
    row_latest = df.loc[df["lead_center"] == latest].iloc[0]
    a, b = int(row_latest["lead_start"]), int(row_latest["lead_end"])
    suffix = window_suffix(a, b)

    fig = plt.figure(figsize=(FIG_W_FULL, 4.65))
    gs_outer = fig.add_gridspec(3, 1, height_ratios=[1.02, 0.12, 1.10], hspace=0.36)
    gs_top = gs_outer[0].subgridspec(1, 3, wspace=0.10)
    gs_bottom = gs_outer[2].subgridspec(1, 2, wspace=0.34)

    map_axes = []
    im = None
    for j, (pred, title) in enumerate(PREDICTORS):
        ax = fig.add_subplot(gs_top[0, j], projection=ccrs.PlateCarree() if HAS_CARTOPY else None)
        map_axes.append(ax)
        im = plot_corr_map_panel(ax, ds_reg, suffix, pred, PANEL_LABELS[j], title=title, rlim=0.6, row=0, col=j, nrow=1, ncol=3)

    cax = fig.add_subplot(gs_outer[1])
    # Shorten the colorbar so it does not dominate the gap between maps and line plots.
    pos = cax.get_position()
    cax.set_position([pos.x0 + 0.18 * pos.width, pos.y0 + 0.20 * pos.height, 0.64 * pos.width, 0.60 * pos.height])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.set_label(f"Local event-level correlation with peak-stage SSTA ({a} to {b} d)", fontsize=7.2, labelpad=1.0)
    cb.ax.tick_params(labelsize=6.6, width=0.5, length=2)
    cb.outline.set_linewidth(0.6)

    ax1 = fig.add_subplot(gs_bottom[0, 0])
    ax2 = fig.add_subplot(gs_bottom[0, 1])
    for pred, _ in PREDICTORS:
        dd = df[df["predictor"] == pred].sort_values("lead_center")
        if dd.empty:
            continue
        ax1.plot(dd["lead_center"], dd["mean_r"], marker="o", ms=3.2, lw=1.0, label=pred_label.get(pred, pred))
        frac_col = "fraction_sig_positive_p05" if "fraction_sig_positive_p05" in dd.columns else "positive_sig_fraction"
        if frac_col in dd.columns:
            ax2.plot(dd["lead_center"], 100*dd[frac_col], marker="o", ms=3.2, lw=1.0, label=pred_label.get(pred, pred))
    for ax, lab, title, ylabel in [
        (ax1, "(d)", "Mean local correlation", "Mean local r"),
        (ax2, "(e)", "Positive significant area", "Positive significant area (%)"),
    ]:
        ax.axhline(0, color="0.35", lw=0.7)
        ax.set_xlabel("Lead-window center relative to event start (days)")
        ax.set_ylabel(ylabel, labelpad=6)
        add_panel_label(ax, lab, title=title, fontsize=7.0, boxed=True, y=0.98)
        apply_grid(ax, alpha=0.22)
        thin_spines(ax)
    ax1.legend(frameon=False, loc="center left", fontsize=6.8)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.965, bottom=0.100)
    save_figure(fig, main_dir, "Fig08_preonset_ocean_state_correlation_summary")
    plt.close(fig)


def plot_figS11_lead_maps(ds_reg: xr.Dataset, si_dir: Path):
    windows = [(-60, -51), (-50, -41), (-40, -31), (-30, -21), (-20, -11), (-10, -1)]
    nrow, ncol = len(windows), 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIG_W_FULL, 6.75),
                             subplot_kw={"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}, constrained_layout=False)
    axes = np.atleast_2d(axes)
    im = None
    for i, (a, b) in enumerate(windows):
        suffix = window_suffix(a, b)
        for j, (pred, title) in enumerate(PREDICTORS):
            im = plot_corr_map_panel(axes[i, j], ds_reg, suffix, pred, PANEL_LABELS[i*ncol+j], title=None, rlim=0.6, row=i, col=j, nrow=nrow, ncol=ncol)
            if i == 0:
                axes[i, j].set_title(title, fontsize=8.0, pad=2.5, fontweight="semibold")
            if j == 0:
                axes[i, j].text(-0.25, 0.5, f"{a} to {b} d", transform=axes[i, j].transAxes, rotation=90,
                                ha="center", va="center", fontsize=7.1, fontweight="semibold")
    fig.subplots_adjust(left=0.125, right=0.900, top=0.955, bottom=0.090, wspace=0.035, hspace=0.075)
    cax = fig.add_axes([0.20, 0.045, 0.60, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("Local event-level correlation with peak-stage SSTA", fontsize=7.3)
    cb.ax.tick_params(labelsize=6.6, width=0.5, length=2)
    save_figure(fig, si_dir, "FigSXX_preonset_lead_window_correlation_maps")
    plt.close(fig)

# -----------------------------------------------------------------------------
# Fig09 and vertical-profile SI
# -----------------------------------------------------------------------------

SWAP_MAP_MAIN = {0: 0, 1: 2, 2: 1, 3: 3}


def prepare_profile_ds(ds: xr.Dataset):
    if "raw_cluster0" in ds and "event" in ds["raw_cluster0"].dims:
        raw = ds["raw_cluster0"].values.astype(int)
        display1 = np.array([SWAP_MAP_MAIN[int(x)] + 1 for x in raw])
        if "display_type" in ds:
            ds = ds.rename({"display_type": "display_type_original"})
        ds = ds.assign(display_type=("event", display1.astype(np.int16)))
    return ds


def event_select_by_type(ds, typ):
    return ds.where(ds["display_type"] == typ, drop=True)


def mean_sem_event(da):
    mean = da.mean("event", skipna=True)
    cnt = da.count("event")
    std = da.std("event", skipna=True)
    sem = std / np.sqrt(cnt)
    try:
        n = int(np.nanmax(cnt.values))
    except Exception:
        n = int(da.sizes.get("event", 0))
    return mean, sem, n


def plot_profile(ax, mean, sem, depth, color, label, lw=1.35, alpha=0.16):
    x = np.asarray(mean)
    s = np.asarray(sem)
    z = np.asarray(depth)
    ax.plot(x, z, color=color, lw=lw, label=label)
    ax.fill_betweenx(z, x-s, x+s, color=color, alpha=alpha, lw=0)


def format_profile_ax(ax, depth_max=270, xlim=None, ylabel=False):
    ax.invert_yaxis()
    ax.set_ylim(depth_max, 0)
    ax.axvline(0, color="0.35", lw=0.7)
    ax.set_xlabel(r"$\Delta T'$ (°C)", fontsize=7.3)
    if ylabel:
        ax.set_ylabel("Depth (m)")
    if xlim is not None:
        ax.set_xlim(*xlim)
    apply_grid(ax, alpha=0.18)
    thin_spines(ax)


def regime_label(reg):
    s = str(reg)
    if s in ["SSH-", "SSH−", "negative", "ssh_minus"]:
        return "SSH−"
    if s in ["SSH+", "positive", "ssh_plus"]:
        return "SSH+"
    return "Neutral" if s.lower() == "neutral" else s


def regime_color(reg):
    lab = regime_label(reg)
    return {"SSH−": COLOR_SSH_MINUS, "Neutral": COLOR_SSH_NEUTRAL, "SSH+": COLOR_SSH_PLUS}.get(lab, "0.45")


def plot_fig09_vertical_profiles(ds: xr.Dataset, main_dir: Path):
    ds = prepare_profile_ds(ds)
    regions = list(ds["region"].values) if "region" in ds.coords else ["Shelf", "Deep"]
    ssh_regs = list(ds["ssh_regime"].values) if "ssh_regime" in ds.coords else []
    fig, axes = plt.subplots(2, 4, figsize=(FIG_W_FULL, 4.65), sharey=True, constrained_layout=False)
    for j, typ in enumerate(TYPE_LIST):
        sub = event_select_by_type(ds, typ)
        ax = axes[0, j]
        for reg, color in [("Shelf", COLOR_SHELF), ("Deep", COLOR_DEEP)]:
            if reg not in regions or "theta_delta_region" not in sub:
                continue
            mean, sem, n = mean_sem_event(sub["theta_delta_region"].sel(region=reg))
            plot_profile(ax, mean, sem, ds["depth"].values, color, reg)
        ax.set_title(f"Type {typ}", fontsize=8.0, pad=2.5, fontweight="semibold")
        format_profile_ax(ax, depth_max=270, xlim=(-0.25, 0.75), ylabel=(j == 0))
        ax.set_xlabel("")
        add_panel_label(ax, PANEL_LABELS[j], fontsize=6.8)
        if j == 3:
            ax.legend(frameon=False, loc="lower right", fontsize=6.7)
        ax = axes[1, j]
        for reg in ssh_regs:
            if "theta_delta_ssh" not in sub:
                continue
            mean, sem, n = mean_sem_event(sub["theta_delta_ssh"].sel(ssh_regime=reg))
            plot_profile(ax, mean, sem, ds["depth"].values, regime_color(reg), regime_label(reg))
        format_profile_ax(ax, depth_max=270, xlim=(-0.50, 0.50), ylabel=(j == 0))
        add_panel_label(ax, PANEL_LABELS[4+j], fontsize=6.8)
        if j == 3:
            ax.legend(frameon=False, loc="lower right", fontsize=6.7)
    fig.subplots_adjust(left=0.125, right=0.985, top=0.940, bottom=0.095, wspace=0.070, hspace=0.150)
    for row, label in enumerate(["Shelf vs deep", "Deep SSH regimes"]):
        pos = axes[row, 0].get_position()
        fig.text(0.045, 0.5 * (pos.y0 + pos.y1), label, rotation=90,
                 ha="center", va="center", fontsize=7.2, fontweight="semibold")
    save_figure(fig, main_dir, "Fig09_glorys_vertical_warming_profiles")
    plt.close(fig)


def plot_figS18_onset_profiles(ds: xr.Dataset, si_dir: Path):
    ds = prepare_profile_ds(ds)
    if "theta_onset_region" not in ds:
        print("[SKIP] theta_onset_region not in profile dataset")
        return
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W_FULL, 5.05), sharey=True, constrained_layout=False)
    axes = axes.ravel()
    for k, (ax, typ) in enumerate(zip(axes, TYPE_LIST)):
        sub = event_select_by_type(ds, typ)
        for reg, color in [("Shelf", COLOR_SHELF), ("Deep", COLOR_DEEP)]:
            if reg in ds["region"].values:
                mean, sem, n = mean_sem_event(sub["theta_onset_region"].sel(region=reg))
                plot_profile(ax, mean, sem, ds["depth"].values, color, reg)
        format_profile_ax(ax, depth_max=270, xlim=None, ylabel=(k % 2 == 0))
        ax.set_xlabel("Onset-mean $T'$ (°C)")
        add_panel_label(ax, PANEL_LABELS[k], title=f"Type {typ}", fontsize=7.0)
        ax.legend(frameon=False, loc="lower right", fontsize=6.8)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.970, bottom=0.085, wspace=0.085, hspace=0.170)
    save_figure(fig, si_dir, "FigS18_shelf_deep_onset_temperature_profiles")
    plt.close(fig)


def layer_overlap_weights(depth, z0, z1):
    z = np.asarray(depth, dtype=float)
    edges = np.zeros(len(z)+1, dtype=float)
    edges[1:-1] = 0.5*(z[:-1] + z[1:])
    edges[0] = max(0.0, z[0] - 0.5*(z[1]-z[0])) if len(z) > 1 else 0.0
    edges[-1] = z[-1] + 0.5*(z[-1]-z[-2]) if len(z) > 1 else z[-1]
    w = np.maximum(0.0, np.minimum(edges[1:], z1) - np.maximum(edges[:-1], z0))
    return xr.DataArray(w, coords={"depth": depth}, dims="depth")


def layer_mean(da, z0, z1):
    w = layer_overlap_weights(da["depth"].values, z0, z1)
    return da.weighted(w).mean("depth", skipna=True)


def build_layer_summaries(ds):
    ds = prepare_profile_ds(ds)
    rows_region = []
    rows_ssh = []
    for typ in TYPE_LIST:
        sub = event_select_by_type(ds, typ)
        if "theta_delta_region" in sub:
            for reg in sub["region"].values:
                for z0, z1 in [(0, 50), (50, 100), (100, 200)]:
                    vals = layer_mean(sub["theta_delta_region"].sel(region=reg), z0, z1).values
                    vals = vals[np.isfinite(vals)]
                    rows_region.append({"type": typ, "category": str(reg), "layer": f"{z0}-{z1} m", "mean": np.nanmean(vals), "sem": np.nanstd(vals, ddof=1)/np.sqrt(len(vals)) if len(vals) > 1 else np.nan, "n": len(vals)})
        if "theta_delta_ssh" in sub:
            for reg in sub["ssh_regime"].values:
                for z0, z1 in [(0, 50), (100, 250)]:
                    vals = layer_mean(sub["theta_delta_ssh"].sel(ssh_regime=reg), z0, z1).values
                    vals = vals[np.isfinite(vals)]
                    rows_ssh.append({"type": typ, "ssh_regime": regime_label(reg), "layer": f"{z0}-{z1} m", "mean": np.nanmean(vals), "sem": np.nanstd(vals, ddof=1)/np.sqrt(len(vals)) if len(vals) > 1 else np.nan, "n": len(vals)})
    return pd.DataFrame(rows_region), pd.DataFrame(rows_ssh)


def build_area_fraction_df(ds):
    ds = prepare_profile_ds(ds)
    if "ssh_regime_area_fraction" not in ds:
        return pd.DataFrame()
    rows = []
    for typ in TYPE_LIST:
        sub = ds["ssh_regime_area_fraction"].where(ds["display_type"] == typ, drop=True)
        for reg in sub["ssh_regime"].values:
            vals = sub.sel(ssh_regime=reg).values
            vals = vals[np.isfinite(vals)]
            rows.append({"type": typ, "ssh_regime": regime_label(reg), "mean": np.nanmean(vals), "sem": np.nanstd(vals, ddof=1)/np.sqrt(len(vals)) if len(vals) > 1 else np.nan, "n": len(vals)})
    return pd.DataFrame(rows)


def plot_figS19_vertical_summary(ds: xr.Dataset, si_dir: Path, table_dir: Path):
    reg_df, ssh_df = build_layer_summaries(ds)
    area_df = build_area_fraction_df(ds)
    if reg_df.empty and ssh_df.empty:
        print("[SKIP] no vertical summary variables found")
        return
    table_dir.mkdir(parents=True, exist_ok=True)
    reg_df.to_csv(table_dir / "glorys_layer_mean_warming_shelf_deep.csv", index=False)
    ssh_df.to_csv(table_dir / "glorys_layer_mean_warming_deep_ssh_regimes.csv", index=False)
    if not area_df.empty:
        area_df.to_csv(table_dir / "glorys_deep_ssh_regime_area_fraction.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(FIG_W_FULL, 4.65), constrained_layout=False)
    x = np.arange(4)
    width = 0.32
    layers = ["0-50 m", "50-100 m", "100-200 m"]
    for i, layer in enumerate(layers):
        ax = axes[0, i]
        for off, cat, color in [(-width/2, "Shelf", COLOR_SHELF), (width/2, "Deep", "#CC79A7")]:
            sub = reg_df[(reg_df["layer"] == layer) & (reg_df["category"] == cat)]
            means = [sub.loc[sub["type"] == t, "mean"].iloc[0] if np.any(sub["type"] == t) else np.nan for t in TYPE_LIST]
            sems = [sub.loc[sub["type"] == t, "sem"].iloc[0] if np.any(sub["type"] == t) else np.nan for t in TYPE_LIST]
            ax.bar(x+off, means, width=width, yerr=sems, capsize=2, color=color, alpha=0.90, label=cat)
        ax.set_title(layer, fontsize=7.5)
        ax.axhline(0, color="0.35", lw=0.7)
        ax.set_xticks(x); ax.set_xticklabels([f"T{t}" for t in TYPE_LIST])
        apply_grid(ax, axis="y", alpha=0.22); thin_spines(ax)
        add_panel_label(ax, PANEL_LABELS[i], fontsize=6.6, boxed=False, y=1.08)
    axes[0, 0].set_ylabel(r"Layer-mean $\Delta T'$ (°C)")
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="upper right")

    # Deep SSH regime layer means and area fraction.
    for k, (ax, layer) in enumerate(zip(axes[1, :2], ["0-50 m", "100-250 m"])):
        width2 = 0.22
        for m, (reg, color) in enumerate([("SSH−", COLOR_SSH_MINUS), ("Neutral", COLOR_SSH_NEUTRAL), ("SSH+", COLOR_SSH_PLUS)]):
            sub = ssh_df[(ssh_df["layer"] == layer) & (ssh_df["ssh_regime"] == reg)]
            means = [sub.loc[sub["type"] == t, "mean"].iloc[0] if np.any(sub["type"] == t) else np.nan for t in TYPE_LIST]
            sems = [sub.loc[sub["type"] == t, "sem"].iloc[0] if np.any(sub["type"] == t) else np.nan for t in TYPE_LIST]
            ax.bar(x+(m-1)*width2, means, width=width2, yerr=sems, capsize=2, color=color, alpha=0.90, label=reg)
        ax.set_title(f"Deep, {layer}", fontsize=7.5)
        ax.axhline(0, color="0.35", lw=0.7)
        ax.set_xticks(x); ax.set_xticklabels([f"T{t}" for t in TYPE_LIST])
        apply_grid(ax, axis="y", alpha=0.22); thin_spines(ax)
        add_panel_label(ax, PANEL_LABELS[3+k], fontsize=6.6, boxed=False, y=1.08)
    axes[1, 0].set_ylabel(r"Layer-mean $\Delta T'$ (°C)")
    axes[1, 0].legend(frameon=False, fontsize=6.1, loc="upper right")

    ax = axes[1, 2]
    if not area_df.empty:
        width2 = 0.22
        for m, (reg, color) in enumerate([("SSH−", COLOR_SSH_MINUS), ("Neutral", COLOR_SSH_NEUTRAL), ("SSH+", COLOR_SSH_PLUS)]):
            sub = area_df[area_df["ssh_regime"] == reg]
            means = [sub.loc[sub["type"] == t, "mean"].iloc[0] if np.any(sub["type"] == t) else np.nan for t in TYPE_LIST]
            sems = [sub.loc[sub["type"] == t, "sem"].iloc[0] if np.any(sub["type"] == t) else np.nan for t in TYPE_LIST]
            ax.bar(x+(m-1)*width2, means, width=width2, yerr=sems, capsize=2, color=color, alpha=0.90, label=reg)
    ax.set_title("Deep SSH-regime area fraction", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([f"T{t}" for t in TYPE_LIST])
    ax.set_ylabel("Fraction of deep-region area")
    apply_grid(ax, axis="y", alpha=0.22); thin_spines(ax)
    add_panel_label(ax, PANEL_LABELS[5], fontsize=6.6, boxed=False, y=1.08)
    fig.subplots_adjust(left=0.070, right=0.990, top=0.935, bottom=0.080, wspace=0.23, hspace=0.42)
    save_figure(fig, si_dir, "FigS19_vertical_warming_quantitative_summary")
    plt.close(fig)

# -----------------------------------------------------------------------------
# Fig10 and heat-budget SI
# -----------------------------------------------------------------------------

BUDGET_LABELS = {
    "dT_obs": r"$\Delta T_{ML}$",
    "dT_Qnet": r"$\Delta T_{Qnet}$",
    "dT_hadv": r"$\Delta T_{hadv}$",
    "residual": "Residual",
}
BUDGET_COLORS = {"dT_obs": "#0072B2", "dT_Qnet": "#E69F00", "dT_hadv": "#009E73", "residual": "#D55E00"}


def plot_fig10_heat_budget_bar(df: pd.DataFrame, main_dir: Path, table_dir: Path):
    df = df.copy()
    df["type_display"] = df["type_display"].astype(int)
    budget_cols = ["dT_obs", "dT_Qnet", "dT_hadv", "residual"]
    region = "whole_gom"
    summary = df.groupby(["region_kind", "type_display"])[budget_cols].agg(["mean", "sem", "count"])
    table_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_dir / "mixed_layer_heat_budget_region_type_summary.csv")
    types = sorted(df["type_display"].dropna().unique())
    fig, ax = plt.subplots(figsize=(FIG_W_HALF, 2.70), constrained_layout=False)
    width = 0.18
    x = np.arange(len(types))
    for i, var in enumerate(budget_cols):
        means = [summary.loc[(region, t), (var, "mean")] for t in types]
        sems = [summary.loc[(region, t), (var, "sem")] for t in types]
        ax.bar(x+(i-1.5)*width, means, width=width, yerr=sems, capsize=2.2, label=BUDGET_LABELS[var],
               color=BUDGET_COLORS[var], linewidth=0.4, edgecolor="0.2")
    counts = [summary.loc[(region, t), ("dT_obs", "count")] for t in types]
    ax.axhline(0, color="0.25", lw=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Type {int(t)}\n(n={int(c)})" for t, c in zip(types, counts)])
    ax.set_ylabel("Onset-stage contribution (°C)")
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=6.6)
    apply_grid(ax, axis="y", alpha=0.22)
    thin_spines(ax)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.96, bottom=0.19)
    save_figure(fig, main_dir, "Fig10_mixed_layer_heat_budget_whole_gom")
    plt.close(fig)


def plot_figS20_heat_budget_maps(ds_contrib: xr.Dataset, si_dir: Path):
    ds_contrib = convert_type_coord(standardize_latlon(ds_contrib))
    contrib_vars = ["dTML", "dTQnet", "dThadv", "residual"]
    titles = {"dTML": r"$\Delta T_{ML}$", "dTQnet": r"$\Delta T_{Qnet}$", "dThadv": r"$\Delta T_{hadv}$", "residual": "Residual"}
    nrow, ncol = len(contrib_vars), 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIG_W_FULL, 5.40),
                             subplot_kw={"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}, constrained_layout=False)
    axes = np.atleast_2d(axes)
    for i, var in enumerate(contrib_vars):
        if var not in ds_contrib:
            continue
        da = ds_contrib[var]
        vmin, vmax = robust_symmetric_limits(da, q=0.98, fallback=0.5)
        im = None
        for j, typ in enumerate(TYPE_LIST):
            ax = axes[i, j]
            im = pmesh_map(ax, da.sel(type_display=typ), vmin=vmin, vmax=vmax, cmap=CMAP_ANOM, extent=(-99, -78, 17.5, 31), row=i, col=j, nrow=nrow, ncol=ncol)
            if i == 0:
                ax.set_title(f"Type {typ}", fontsize=8.0, pad=2.5, fontweight="semibold")
            if j == 0:
                ax.text(-0.22, 0.5, titles[var], transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=7.4, fontweight="semibold")
            add_panel_label(ax, PANEL_LABELS[i*ncol+j], fontsize=6.5)
        cax = fig.add_axes([0.903, 0.76 - i*0.215, 0.014, 0.150])
        cb = fig.colorbar(im, cax=cax, orientation="vertical", extend="both")
        cb.set_label("°C", fontsize=6.8)
        cb.ax.tick_params(labelsize=6.2, width=0.5, length=2)
    fig.subplots_adjust(left=0.110, right=0.885, top=0.945, bottom=0.075, wspace=0.050, hspace=0.085)
    save_figure(fig, si_dir, "FigS20_mixed_layer_heat_budget_contribution_maps")
    plt.close(fig)

# -----------------------------------------------------------------------------
# Optional: GLORYS longitude-depth temperature sections
# -----------------------------------------------------------------------------

def plot_figS17_temperature_sections(ds_raw: xr.Dataset, si_dir: Path):
    ds_raw = convert_type_coord(standardize_latlon(ds_raw))
    var = "thetao_onset_resid"
    if var not in ds_raw:
        print("[SKIP] thetao_onset_resid not found for section plot")
        return
    da = ds_raw[var]
    lat_targets = [29, 26, 24, 21]
    lon_min, lon_max = SECTION_LON_RANGE
    dep_min, dep_max = SECTION_DEPTH_RANGE
    glo_slice = slice(min(lon_min, lon_max), max(lon_min, lon_max))
    nrow, ncol = 4, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIG_W_FULL, 5.35), sharex=True, sharey=True, constrained_layout=False)
    im = None
    for i, lat0 in enumerate(lat_targets):
        sec_all = da.sel(latitude=float(lat0), method="nearest").sel(longitude=glo_slice).sel(depth=slice(dep_min, dep_max))
        lon = np.asarray(sec_all["longitude"].values, dtype=float)
        inside = gom_lon_mask_for_lat(lon, lat0)
        sec_all = sec_all.where(xr.DataArray(inside, dims=["longitude"], coords={"longitude": sec_all["longitude"]}))

        for j, typ in enumerate(TYPE_LIST):
            ax = axes[i, j]
            x = sec_all.sel(type_display=typ) if "type_display" in sec_all.coords else sec_all.isel(cluster=typ-1)
            depth_plot = np.asarray(x["depth"].values, dtype=float)
            order = np.argsort(depth_plot)
            depth_plot = depth_plot[order]
            x_plot = x.isel(depth=order)
            im = ax.pcolormesh(x_plot["longitude"], depth_plot, x_plot, cmap=CMAP_TEMP, vmin=-1.8, vmax=1.8, shading="auto")

            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(dep_max, dep_min)
            ax.set_xticks(np.arange(-95, -79, 5))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon_w))
            if i == 0:
                ax.set_title(f"Type {typ}", fontsize=8.0, pad=2.5, fontweight="semibold")
            if j == 0:
                ax.set_ylabel("Depth (m)")
            else:
                ax.set_ylabel("")
            if i == nrow-1:
                ax.set_xlabel("Longitude")
            else:
                ax.set_xticklabels([])
            add_panel_label(ax, PANEL_LABELS[i*ncol+j], fontsize=6.3)
            apply_grid(ax, alpha=0.13)
            thin_spines(ax)
    fig.subplots_adjust(left=0.090, right=0.900, top=0.945, bottom=0.085, wspace=0.055, hspace=0.100)
    for i, lat0 in enumerate(lat_targets):
        pos = axes[i, 0].get_position()
        fig.text(0.045, 0.5 * (pos.y0 + pos.y1), f"{lat0}°N", rotation=90,
                 ha="center", va="center", fontsize=7.2, fontweight="semibold")
    right_colorbar(fig, axes, im, r"$T'$ (°C)", x=0.918, y=0.20, h=0.62)
    save_figure(fig, si_dir, "FigS17_zonal_sections_potential_temperature_anomaly_onset")
    plt.close(fig)


def _as_dataarray_from_nc(ds: xr.Dataset, preferred: list[str] | None = None) -> xr.DataArray:
    preferred = preferred or []
    for name in preferred:
        if name in ds:
            return ds[name]
    return ds[list(ds.data_vars)[0]]


def plot_figS16_seasonal_type_mean_vertical_structure(monthly_path: Path, cluster_path: Path, si_dir: Path):
    if not monthly_path.exists() or not cluster_path.exists():
        print("[SKIP] Missing S16 vertical-structure cache:", monthly_path, cluster_path)
        return
    theta_month = _as_dataarray_from_nc(xr.open_dataset(monthly_path), ["thetao", "theta"])
    theta_cluster = _as_dataarray_from_nc(xr.open_dataset(cluster_path), ["thetao", "theta"])
    theta_month = theta_month.sel(depth=slice(0, 200))
    theta_cluster = theta_cluster.sel(depth=slice(0, 200))
    if "cluster" in theta_cluster.dims and theta_cluster.sizes["cluster"] >= 4:
        theta_cluster = theta_cluster.isel(cluster=[0, 2, 1, 3]).assign_coords(type_display=("cluster", TYPE_LIST)).swap_dims({"cluster": "type_display"})
    elif "type_display" not in theta_cluster.coords:
        theta_cluster = convert_type_coord(theta_cluster)

    depth = theta_month["depth"]
    prof_dec_may = theta_month.sel(month=[12, 1, 2, 3, 4, 5]).mean("month")
    prof_jun_nov = theta_month.sel(month=[6, 7, 8, 9, 10, 11]).mean("month")

    fig, ax = plt.subplots(figsize=(FIG_W_HALF, 4.35), constrained_layout=False)
    ax.plot(prof_dec_may, depth, color="0.40", lw=1.5, ls="--", label="Dec–May climatology")
    ax.plot(prof_jun_nov, depth, color="0.10", lw=1.5, ls="-", label="Jun–Nov climatology")
    for typ in TYPE_LIST:
        if "type_display" in theta_cluster.coords:
            prof = theta_cluster.sel(type_display=typ)
        else:
            prof = theta_cluster.isel(cluster=typ-1)
        ax.plot(prof, prof["depth"], color=TYPE_COLORS[typ], lw=1.7, label=f"Type {typ}")
    ax.set_ylim(200, 0)
    ax.set_xlabel("Potential temperature (°C)")
    ax.set_ylabel("Depth (m)")
    apply_grid(ax, alpha=0.22)
    thin_spines(ax)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left", handlelength=2.6)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.12)
    save_figure(fig, si_dir, "FigS16_seasonal_type_mean_vertical_structure")
    plt.close(fig)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("/Users/luty8/MHW_Project"))
    parser.add_argument("--strict", action="store_true", help="Raise on missing optional input instead of skipping.")
    parser.add_argument("--skip-sections", action="store_true", help="Skip optional longitude-depth section plot.")
    args = parser.parse_args()

    set_style()

    rev_dir = args.base / "code/projects/Kmeans_eof_2baseline/Revised_0705"
    main_dir = rev_dir / "output/main"
    si_dir = rev_dir / "output/si"
    table_dir = rev_dir / "output/tables"
    main_dir.mkdir(parents=True, exist_ok=True)
    si_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    pre_dir = args.base / "data/core/pre_onset"
    proc_dir = args.base / "data/processed"
    budget_dir = proc_dir / "glorys_mlhb_event_budget"
    comp_dir = args.base / "data/core/composite_result"
    interim_dir = args.base / "data/interim"

    # FigS16
    plot_figS16_seasonal_type_mean_vertical_structure(
        interim_dir / "thetao_monthly_profiles_climatology_GoM_1993_2024.nc",
        interim_dir / "thetao_cluster_profiles_K4_start_to_peak_GoM_1993_2024.nc",
        si_dir,
    )

    # Fig07
    ds_type_path = pre_dir / "glorys_ocean_state_preonset_type_composite_minus20to1_GoM_1993-2024.nc"
    ds_type = open_optional(ds_type_path, strict=args.strict)
    if ds_type is not None:
        plot_fig07_preonset_ocean_state(ds_type, main_dir)

    # Fig08 and S11
    reg_path = pre_dir / "local_regression_lead_windows_peak_pm3_SSTA_against_preonset_ocean_state.nc"
    sum_path = pre_dir / "local_regression_lead_window_summary_peak_pm3.csv"
    ds_reg = open_optional(reg_path, strict=args.strict)
    if ds_reg is not None and sum_path.exists():
        plot_fig08_correlation_summary(ds_reg, sum_path, main_dir)
    elif ds_reg is not None:
        print("[SKIP] Missing lead-window summary CSV:", sum_path)

    # Fig09, S12, S13
    profile_path = proc_dir / "glorys_event_profiles_shelf_deep_ssh_onset_minus_pre.nc"
    prof_ds = open_optional(profile_path, strict=args.strict)
    if prof_ds is not None:
        plot_fig09_vertical_profiles(prof_ds, main_dir)
        plot_figS18_onset_profiles(prof_ds, si_dir)
        plot_figS19_vertical_summary(prof_ds, si_dir, table_dir)

    # Fig10 and heat budget maps
    thresh = "0p8"
    event_csv = budget_dir / f"event_mlhb_budget_whole_core_noncore_polygon_oisstcore_sst{thresh}_1993-2024.csv"
    contrib_nc = budget_dir / f"mlhb_type_mean_contribution_maps_polygon_oisstcore_sst{thresh}_1993-2024.nc"
    if event_csv.exists():
        df_budget = pd.read_csv(event_csv)
        plot_fig10_heat_budget_bar(df_budget, main_dir, table_dir)
    elif args.strict:
        raise FileNotFoundError(event_csv)
    else:
        print("[SKIP] Missing heat-budget event CSV:", event_csv)
    ds_contrib = open_optional(contrib_nc, strict=args.strict)
    if ds_contrib is not None:
        plot_figS20_heat_budget_maps(ds_contrib, si_dir)

    # Optional GLORYS zonal sections
    if not args.skip_sections:
        ds_raw = open_optional(comp_dir / "raw_onset_resid_allvars.nc", strict=False)
        if ds_raw is not None:
            plot_figS17_temperature_sections(ds_raw, si_dir)


if __name__ == "__main__":
    main()
