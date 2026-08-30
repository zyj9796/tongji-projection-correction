from __future__ import annotations

import argparse
import csv
import json
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


TOUYING_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = TOUYING_DIR.parent
REPO_ROOT = PROJECT_DIR.parent

sys.path.insert(0, str(TOUYING_DIR / "code"))
sys.path.insert(0, str(REPO_ROOT / "geocoding" / "code"))
sys.path.insert(0, str(REPO_ROOT.parent / "src"))

from optimize_projection import apply_dsm_heights, geotiff_bounds, projection_score  # noqa: E402
from io_paths import BUILDINGS_SHP, DSM_TIF, FULL_AREA_RASTER_DIR, RSLC_DIR, TIF_DIR  # noqa: E402
from raster_height import RasterHeightSampler  # noqa: E402
from geocode_gamma_rslc_with_buildings import make_orbit, parse_gamma_par, read_rslc_amplitude  # noqa: E402
from geocode_tongji_all_buildings_compare_gamma import load_area_buildings  # noqa: E402
from reproduce_thesis_tongji_tsx import rasterize_building  # noqa: E402
from chinese_matplotlib import install_chinese_labels  # noqa: E402

install_chinese_labels()


def write_wgs84_buildings(path: Path, buildings: list[dict]) -> None:
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


def write_sar_projection_geojson(path: Path, projection_items: list[dict]) -> None:
    features = []
    for item in projection_items:
        for surface_name, rc in [("bottom", item["bottom_rc"]), ("roof", item["roof_rc"])]:
            xy = np.column_stack([rc[:, 1], rc[:, 0]])
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "fid": item["fid"],
                        "surface": surface_name,
                        "height_m": item["height_m"],
                        "base_height_m": item["base_height_m"],
                        "top_height_m": item["top_height_m"],
                        "mask0_pixels": item["mask0_pixels"],
                        "mask_pixels": item["mask_pixels"],
                        "projection_score": item["projection_score"],
                    },
                    "geometry": {"type": "Polygon", "coordinates": [xy.tolist() + [xy[0].tolist()]]},
                }
            )
    payload = {
        "type": "FeatureCollection",
        "name": "tongji_full_area_building_projection_sar_col_row",
        "coordinate_system": "SAR image coordinates: x=range column, y=azimuth row",
        "features": features,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_triangles_npz(path: Path, projection_items: list[dict]) -> None:
    arrays = {}
    meta = []
    for i, item in enumerate(projection_items):
        key = f"fid_{item['fid']}_{i}"
        arrays[f"{key}_projected_rc"] = item["projected_rc"]
        arrays[f"{key}_triangles"] = item["triangles"]
        meta.append({"key": key, "fid": item["fid"]})
    arrays["meta_json"] = np.asarray(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(path, **arrays)


def plot_full_area(path: Path, amp: np.ndarray, projection_items: list[dict], max_draw: int) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 8.0), dpi=260)
    ax.imshow(amp, cmap="gray", vmin=0, vmax=255)
    draw_items = projection_items if max_draw <= 0 else projection_items[:max_draw]
    all_rc = []
    for item in draw_items:
        roof_xy = np.column_stack([item["roof_rc"][:, 1], item["roof_rc"][:, 0]])
        bottom_xy = np.column_stack([item["bottom_rc"][:, 1], item["bottom_rc"][:, 0]])
        all_rc.append(item["projected_rc"])
        ax.add_patch(MplPolygon(bottom_xy, closed=True, fill=False, edgecolor="#00d4ff", linewidth=0.22, alpha=0.38))
        ax.add_patch(MplPolygon(roof_xy, closed=True, fill=False, edgecolor="#ffb000", linewidth=0.24, alpha=0.62))
    if all_rc:
        rc_all = np.vstack(all_rc)
        xmin = max(0, float(np.nanmin(rc_all[:, 1])) - 80)
        xmax = min(amp.shape[1], float(np.nanmax(rc_all[:, 1])) + 80)
        ymin = max(0, float(np.nanmin(rc_all[:, 0])) - 80)
        ymax = min(amp.shape[0], float(np.nanmax(rc_all[:, 0])) + 80)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)
    ax.set_xlabel("距离向列号")
    ax.set_ylabel("方位向行号")
    ax.set_title(f"同济校区全区建筑矢量投影（{len(projection_items)}栋建筑）")
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
    if not gamma_tif.exists():
        raise FileNotFoundError(gamma_tif)

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
        raise RuntimeError("No buildings selected")

    rows = []
    projection_items = []
    skipped = []
    for i, building in enumerate(buildings, start=1):
        try:
            model = rasterize_building(building, par, orbit, amp.shape)
            score = projection_score(model, amp, args.min_pixels)
            rc = np.asarray(model["projected_rc"], dtype=np.float64)
            n = int(np.asarray(building["ring_lonlat"]).shape[0])
            finite = np.all(np.isfinite(rc), axis=1)
            row = {
                "date": args.date,
                "fid": int(building["fid"]),
                "height_m": float(building["height_m"]),
                "base_height_m": float(building.get("base_height_m", 0.0)),
                "top_height_m": float(building.get("top_height_m", 0.0)),
                "vertices": n,
                "projected_vertices": int(np.sum(finite)),
                "row_min": float(np.nanmin(rc[:, 0])),
                "row_max": float(np.nanmax(rc[:, 0])),
                "col_min": float(np.nanmin(rc[:, 1])),
                "col_max": float(np.nanmax(rc[:, 1])),
                "center_row": float(np.nanmean(rc[:, 0])),
                "center_col": float(np.nanmean(rc[:, 1])),
                "mask0_pixels": int(score["mask0_pixels"]),
                "mask_pixels": int(score["mask_pixels"]),
                "projection_score": float(score["score"]),
                "mean_amp": float(score["mean_amp"]),
                "p90_amp": float(score["p90_amp"]),
                "ok": int(bool(score["ok"])),
            }
            rows.append(row)
            projection_items.append(
                {
                    **row,
                    "bottom_rc": rc[:n],
                    "roof_rc": rc[n : 2 * n],
                    "projected_rc": rc,
                    "triangles": np.asarray(model["triangles"], dtype=np.int32),
                }
            )
        except Exception as exc:
            skipped.append({"fid": int(building.get("fid", -1)), "reason": type(exc).__name__, "message": str(exc)})
        if i % 100 == 0:
            print(f"projected {i}/{len(buildings)} valid={len(rows)} skipped={len(skipped)}", flush=True)

    if not rows:
        raise RuntimeError("No projected buildings were generated")

    metrics_csv = out_dir / f"{args.date}_full_area_projection_metrics.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    skipped_csv = out_dir / f"{args.date}_full_area_projection_skipped.csv"
    with skipped_csv.open("w", encoding="utf-8", newline="") as f:
        fields = sorted({k for row in skipped for k in row}) if skipped else ["fid", "reason"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(skipped)

    write_wgs84_buildings(out_dir / f"{args.date}_full_area_projection_buildings_wgs84.geojson", buildings)
    write_sar_projection_geojson(out_dir / f"{args.date}_full_area_projection_sar_col_row.geojson", projection_items)
    write_triangles_npz(out_dir / f"{args.date}_full_area_projection_triangles.npz", projection_items)
    plot_full_area(out_dir / f"{args.date}_full_area_projection_overlay.png", amp, projection_items, args.max_plot_buildings)

    summary = {
        "date": args.date,
        "input_buildings": len(buildings),
        "projected_buildings": len(rows),
        "skipped_buildings": len(skipped),
        "ok_buildings": int(sum(r["ok"] for r in rows)),
        "mean_mask0_pixels": float(np.mean([r["mask0_pixels"] for r in rows])),
        "median_projection_score": float(np.median([r["projection_score"] for r in rows])),
        "metrics_csv": str(metrics_csv),
        "sar_projection_geojson": str(out_dir / f"{args.date}_full_area_projection_sar_col_row.geojson"),
        "overlay_png": str(out_dir / f"{args.date}_full_area_projection_overlay.png"),
    }
    (out_dir / f"{args.date}_full_area_projection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Tongji full-area building projection in SAR coordinates.")
    parser.add_argument("--date", default="20200708")
    parser.add_argument("--buildings-shp", default=str(BUILDINGS_SHP))
    parser.add_argument("--dsm", default=str(DSM_TIF))
    parser.add_argument("--gamma-tif", default="")
    parser.add_argument("--out-dir", default=str(TOUYING_DIR / "results" / "full_area_projection"))
    parser.add_argument("--fids", default="")
    parser.add_argument("--max-buildings", type=int, default=0, help="0 means all buildings in scene bounds")
    parser.add_argument("--min-pixels", type=int, default=4)
    parser.add_argument("--max-plot-buildings", type=int, default=0, help="0 means draw all projected buildings")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
