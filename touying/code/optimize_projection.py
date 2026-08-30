from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from osgeo import gdal
from scipy.optimize import minimize


TOUYING_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = TOUYING_DIR.parent
REPO_ROOT = PROJECT_DIR.parent

sys.path.insert(0, str(REPO_ROOT / "geocoding" / "code"))
sys.path.insert(0, str(REPO_ROOT.parent / "src"))

from io_paths import BUILDINGS_SHP, DSM_TIF, FULL_AREA_RASTER_DIR, RSLC_DIR, TIF_DIR  # noqa: E402
from raster_height import RasterHeightSampler  # noqa: E402
from geocode_gamma_rslc_with_buildings import make_orbit, parse_gamma_par, read_rslc_amplitude  # noqa: E402
from geocode_tongji_all_buildings_compare_gamma import load_area_buildings  # noqa: E402
from reproduce_thesis_tongji_tsx import rasterize_building, refine_mask  # noqa: E402
from chinese_matplotlib import install_chinese_labels  # noqa: E402

install_chinese_labels()


def geotiff_bounds(path: Path) -> tuple[float, float, float, float]:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(path)
    gt = ds.GetGeoTransform()
    xs = [gt[0], gt[0] + ds.RasterXSize * gt[1]]
    ys = [gt[3], gt[3] + ds.RasterYSize * gt[5]]
    ds = None
    return min(xs), min(ys), max(xs), max(ys)


def local_en(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    east = (lon - lon0) * math.pi / 180.0 * 6378137.0 * math.cos(math.radians(lat0))
    north = (lat - lat0) * math.pi / 180.0 * 6378137.0
    return east, north


def en_to_lonlat(east: np.ndarray, north: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    lon = lon0 + east / (6378137.0 * math.cos(math.radians(lat0))) * 180.0 / math.pi
    lat = lat0 + north / 6378137.0 * 180.0 / math.pi
    return lon, lat


def shifted_building(building: dict, east_m: float, north_m: float, height_shift_m: float) -> dict:
    item = dict(building)
    ring = np.asarray(building["ring_lonlat"], dtype=np.float64)
    lon0 = float(np.mean(ring[:, 0]))
    lat0 = float(np.mean(ring[:, 1]))
    east, north = local_en(ring[:, 0], ring[:, 1], lon0, lat0)
    lon, lat = en_to_lonlat(east + east_m, north + north_m, lon0, lat0)
    item["ring_lonlat"] = np.column_stack([lon, lat])
    if "base_height_m" in item:
        item["base_height_m"] = max(0.0, float(item["base_height_m"]) + height_shift_m)
    if "top_height_m" in item:
        item["top_height_m"] = max(float(item.get("base_height_m", 0.0)), float(item["top_height_m"]) + height_shift_m)
    return item


def apply_dsm_heights(buildings: list[dict], dsm: RasterHeightSampler) -> list[dict]:
    out = []
    for building in buildings:
        try:
            top_h = dsm.building_surface_height(building["ring_lonlat"])
        except Exception:
            continue
        item = dict(building)
        item["top_height_m"] = float(top_h)
        item["base_height_m"] = max(0.0, float(top_h) - float(item["height_m"]))
        out.append(item)
    return out


def projection_score(model: dict, amp: np.ndarray, min_pixels: int) -> dict:
    mask0 = model["mask0"]
    n0 = int(mask0.sum())
    if n0 < min_pixels:
        return {"ok": False, "score": -1e6, "mask0_pixels": n0, "mask_pixels": 0, "mean_amp": 0.0, "p90_amp": 0.0}
    vals = amp[mask0].astype(np.float64)
    refined = refine_mask(mask0, amp)
    n_refined = int(refined.sum())
    if vals.size == 0:
        return {"ok": False, "score": -1e6, "mask0_pixels": n0, "mask_pixels": n_refined, "mean_amp": 0.0, "p90_amp": 0.0}
    p75 = float(np.percentile(vals, 75))
    p90 = float(np.percentile(vals, 90))
    top = vals[vals >= p75]
    mean_top = float(np.mean(top)) if top.size else float(np.mean(vals))
    return {
        "ok": True,
        "score": 0.55 * mean_top + 0.45 * p90 + 0.02 * math.log1p(max(n_refined, 0)),
        "mask0_pixels": n0,
        "mask_pixels": n_refined,
        "mean_amp": float(np.mean(vals)),
        "p90_amp": p90,
    }


def optimize_one(
    building: dict,
    par: dict,
    orbit,
    amp: np.ndarray,
    max_offset_m: float,
    max_height_shift_m: float,
    maxiter: int,
    min_pixels: int,
) -> tuple[dict, dict, dict, dict]:
    baseline_model = rasterize_building(building, par, orbit, amp.shape)
    baseline = projection_score(baseline_model, amp, min_pixels)

    def objective(x: np.ndarray) -> float:
        east_m, north_m, height_shift_m = map(float, x)
        candidate = shifted_building(building, east_m, north_m, height_shift_m)
        try:
            model = rasterize_building(candidate, par, orbit, amp.shape)
            score = projection_score(model, amp, min_pixels)["score"]
        except Exception:
            return 1e6
        offset_penalty = 8.0 * ((east_m * east_m + north_m * north_m) / max(max_offset_m * max_offset_m, 1e-6))
        height_penalty = 2.0 * ((height_shift_m * height_shift_m) / max(max_height_shift_m * max_height_shift_m, 1e-6))
        return -score + offset_penalty + height_penalty

    res = minimize(
        objective,
        np.zeros(3, dtype=np.float64),
        method="Powell",
        bounds=[(-max_offset_m, max_offset_m), (-max_offset_m, max_offset_m), (-max_height_shift_m, max_height_shift_m)],
        options={"maxiter": maxiter, "xtol": 0.05, "ftol": 0.05, "disp": False},
    )
    east_m, north_m, height_shift_m = map(float, res.x)
    optimized = shifted_building(building, east_m, north_m, height_shift_m)
    optimized_model = rasterize_building(optimized, par, orbit, amp.shape)
    optimized_score = projection_score(optimized_model, amp, min_pixels)
    offsets = {
        "east_offset_m": east_m,
        "north_offset_m": north_m,
        "horizontal_offset_m": math.hypot(east_m, north_m),
        "height_shift_m": height_shift_m,
        "optimizer_success": bool(res.success),
        "optimizer_fun": float(res.fun),
    }
    return optimized, baseline, optimized_score, offsets


def write_geojson(path: Path, buildings: list[dict]) -> None:
    features = []
    for item in buildings:
        ring = np.asarray(item["ring_lonlat"], dtype=float)
        features.append(
            {
                "type": "Feature",
                "properties": {k: v for k, v in item.items() if k != "ring_lonlat"},
                "geometry": {"type": "Polygon", "coordinates": [ring.tolist() + [ring[0].tolist()]]},
            }
        )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_overlay(path: Path, amp: np.ndarray, rows: list[dict], par: dict, orbit) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9.0, 7.0), dpi=220)
    ax.imshow(amp, cmap="gray", vmin=0, vmax=255)
    colors = ["#00d4ff", "#ffb000"]
    labels = ["初始", "优化后"]
    all_rc = []
    for item in rows:
        for mode, color, label in zip(["initial_building", "optimized_building"], colors, labels):
            model = rasterize_building(item[mode], par, orbit, amp.shape)
            rc = model["projected_rc"]
            all_rc.append(rc)
            for tri in model["triangles"]:
                pts = np.column_stack([rc[tri, 1], rc[tri, 0]])
                ax.add_patch(MplPolygon(pts, closed=True, fill=False, edgecolor=color, linewidth=0.28, alpha=0.58))
            rr, cc = np.nonzero(model["mask0"])
            if rr.size:
                step = max(1, rr.size // 900)
                ax.scatter(cc[::step], rr[::step], s=0.8, c=color, alpha=0.32, linewidths=0, label=label)
    rc_all = np.vstack(all_rc)
    xmin = max(0, float(np.nanmin(rc_all[:, 1])) - 80)
    xmax = min(amp.shape[1], float(np.nanmax(rc_all[:, 1])) + 80)
    ymin = max(0, float(np.nanmin(rc_all[:, 0])) - 80)
    ymax = min(amp.shape[0], float(np.nanmax(rc_all[:, 0])) + 80)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_xlabel("距离向列号")
    ax.set_ylabel("方位向行号")
    ax.set_title("合成孔径雷达坐标中的建筑矢量投影优化")
    handles, labels_seen = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels_seen):
        unique.setdefault(label, handle)
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gamma_tif = Path(args.gamma_tif) if args.gamma_tif else TIF_DIR / f"{args.date}_gamma_dem_geocoded_wgs84.tif"
    if not gamma_tif.exists():
        fallback = FULL_AREA_RASTER_DIR / f"{args.date}_building_aligned_gamma_dsm_geocoded_wgs84.tif"
        gamma_tif = fallback if fallback.exists() else gamma_tif

    par = parse_gamma_par(RSLC_DIR / f"{args.date}.rslc.par")
    orbit = make_orbit(par)
    amp = read_rslc_amplitude(RSLC_DIR / f"{args.date}.rslc", int(par["azimuth_lines"]), int(par["range_samples"]))
    dsm = RasterHeightSampler(Path(args.dsm))
    bounds = geotiff_bounds(gamma_tif)
    buildings = apply_dsm_heights(load_area_buildings(Path(args.buildings_shp), bounds), dsm)

    if args.fids:
        wanted = {int(x) for x in args.fids.split(",") if x.strip()}
        buildings = [b for b in buildings if int(b["fid"]) in wanted]
    if args.max_buildings > 0:
        buildings = buildings[: args.max_buildings]
    if not buildings:
        raise RuntimeError("No buildings selected for projection optimization")

    rows = []
    optimized_buildings = []
    plot_rows = []
    for i, building in enumerate(buildings, start=1):
        optimized, before, after, offsets = optimize_one(
            building,
            par,
            orbit,
            amp,
            max_offset_m=args.max_offset_m,
            max_height_shift_m=args.max_height_shift_m,
            maxiter=args.maxiter,
            min_pixels=args.min_pixels,
        )
        metric = {
            "date": args.date,
            "fid": int(building["fid"]),
            "height_m": float(building["height_m"]),
            "base_height_m": float(building.get("base_height_m", 0.0)),
            "top_height_m": float(building.get("top_height_m", 0.0)),
            "before_score": before["score"],
            "after_score": after["score"],
            "score_gain": after["score"] - before["score"],
            "before_mask0_pixels": before["mask0_pixels"],
            "after_mask0_pixels": after["mask0_pixels"],
            "before_mask_pixels": before["mask_pixels"],
            "after_mask_pixels": after["mask_pixels"],
            "before_mean_amp": before["mean_amp"],
            "after_mean_amp": after["mean_amp"],
            "before_p90_amp": before["p90_amp"],
            "after_p90_amp": after["p90_amp"],
            **offsets,
        }
        rows.append(metric)
        opt_item = dict(optimized)
        opt_item.update(metric)
        optimized_buildings.append(opt_item)
        if len(plot_rows) < args.plot_buildings:
            plot_rows.append({"initial_building": building, "optimized_building": optimized})
        print(
            f"fid={building['fid']} score {before['score']:.2f}->{after['score']:.2f} "
            f"offset=({offsets['east_offset_m']:.2f},{offsets['north_offset_m']:.2f})m "
            f"height_shift={offsets['height_shift_m']:.2f}m",
            flush=True,
        )

    csv_path = out_dir / f"{args.date}_projection_optimization_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_geojson(out_dir / f"{args.date}_optimized_buildings.geojson", optimized_buildings)
    plot_overlay(out_dir / f"{args.date}_projection_optimization_overlay.png", amp, plot_rows, par, orbit)

    summary = {
        "date": args.date,
        "buildings": len(rows),
        "mean_score_gain": float(np.mean([r["score_gain"] for r in rows])),
        "median_horizontal_offset_m": float(np.median([r["horizontal_offset_m"] for r in rows])),
        "metrics_csv": str(csv_path),
        "optimized_geojson": str(out_dir / f"{args.date}_optimized_buildings.geojson"),
        "note": "Small EN/height correction optimized against RSLC amplitude using the GAMMA orbit/range/Doppler projection functions. This is an experiment for vector projection accuracy, not a replacement for the main GAMMA/DSM geocoded backdrop.",
    }
    (out_dir / f"{args.date}_projection_optimization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize Tongji building-vector projection in SAR coordinates.")
    parser.add_argument("--date", default="20200708")
    parser.add_argument("--buildings-shp", default=str(BUILDINGS_SHP))
    parser.add_argument("--dsm", default=str(DSM_TIF))
    parser.add_argument("--gamma-tif", default="")
    parser.add_argument("--out-dir", default=str(TOUYING_DIR / "results"))
    parser.add_argument("--fids", default="", help="Comma-separated FIDs. Empty means use sorted buildings in scene bounds.")
    parser.add_argument("--max-buildings", type=int, default=10)
    parser.add_argument("--max-offset-m", type=float, default=8.0)
    parser.add_argument("--max-height-shift-m", type=float, default=12.0)
    parser.add_argument("--maxiter", type=int, default=22)
    parser.add_argument("--min-pixels", type=int, default=4)
    parser.add_argument("--plot-buildings", type=int, default=6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
