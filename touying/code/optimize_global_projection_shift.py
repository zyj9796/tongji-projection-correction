from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import binary_dilation


TOUYING_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = TOUYING_DIR.parent
WORK_ROOT = PROJECT_DIR / "results" / "outputs" / "work" / "gamma_dsm_geocode"


def par_value(path: Path, key: str, cast=float):
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(rf"^{re.escape(key)}:\s+([^\s]+)", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing {key} in {path}")
    return cast(float(match.group(1))) if cast is int else cast(match.group(1))


def read_real4(path: Path, width: int, nlines: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=">f4")
    if arr.size != width * nlines:
        raise ValueError(f"Unexpected size for {path}: {arr.size}, expected {width * nlines}")
    return arr.reshape(nlines, width).astype(np.float32)


def stretch_amp(arr: np.ndarray) -> np.ndarray:
    amp = np.sqrt(np.maximum(arr.astype(np.float32), 0.0))
    valid = np.isfinite(amp) & (amp > 0)
    if not np.any(valid):
        return np.zeros_like(amp, dtype=np.float32)
    p2, p98 = np.percentile(amp[valid], [2, 98])
    return np.clip((amp - p2) / max(float(p98 - p2), 1e-6), 0.0, 1.0)


def edge_map(img: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(img.astype(np.float32))
    edge = np.hypot(gx, gy)
    valid = np.isfinite(edge)
    p98 = np.percentile(edge[valid], 98) if np.any(valid) else 1.0
    return np.clip(edge / max(float(p98), 1e-6), 0.0, 1.0)


def load_surface_polygons(path: Path, surface: str, min_mask_pixels: int) -> list[np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    polys: list[np.ndarray] = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        if props.get("surface") != surface:
            continue
        if int(float(props.get("mask0_pixels", 0))) < min_mask_pixels:
            continue
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        if geom.get("type") != "Polygon" or not coords:
            continue
        xy = np.asarray(coords[0], dtype=np.float64)
        if xy.shape[0] > 1 and np.allclose(xy[0], xy[-1]):
            xy = xy[:-1]
        if xy.shape[0] >= 3 and np.all(np.isfinite(xy)):
            polys.append(xy)
    return polys


def rasterize_polygons(polys: list[np.ndarray], rows: int, cols: int) -> np.ndarray:
    mask = np.zeros((rows, cols), dtype=bool)
    for xy in polys:
        xmin = max(0, int(np.floor(np.min(xy[:, 0]))) - 1)
        xmax = min(cols - 1, int(np.ceil(np.max(xy[:, 0]))) + 1)
        ymin = max(0, int(np.floor(np.min(xy[:, 1]))) - 1)
        ymax = min(rows - 1, int(np.ceil(np.max(xy[:, 1]))) + 1)
        if xmax < xmin or ymax < ymin:
            continue
        yy, xx = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
        inside = MplPath(xy).contains_points(np.column_stack([xx.ravel(), yy.ravel()])).reshape(yy.shape)
        mask[ymin : ymax + 1, xmin : xmax + 1] |= inside
    return mask


def shift_mask(mask: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    rows, cols = mask.shape
    src_r0 = max(0, -row_shift)
    src_r1 = min(rows, rows - row_shift)
    src_c0 = max(0, -col_shift)
    src_c1 = min(cols, cols - col_shift)
    dst_r0 = max(0, row_shift)
    dst_r1 = min(rows, rows + row_shift)
    dst_c0 = max(0, col_shift)
    dst_c1 = min(cols, cols + col_shift)
    if src_r1 > src_r0 and src_c1 > src_c0:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
    return out


def score_shift(mask: np.ndarray, amp: np.ndarray, edges: np.ndarray, row_shift: int, col_shift: int) -> dict:
    shifted = shift_mask(mask, row_shift, col_shift)
    n = int(shifted.sum())
    if n < 100:
        return {"score": -1e9, "inside_amp": 0.0, "ring_amp": 0.0, "inside_edge": 0.0, "pixels": n}
    ring = binary_dilation(shifted, iterations=5) & ~binary_dilation(shifted, iterations=1)
    inside_amp = float(np.mean(amp[shifted]))
    ring_amp = float(np.mean(amp[ring])) if np.any(ring) else float(np.mean(amp))
    inside_edge = float(np.mean(edges[shifted]))
    ring_edge = float(np.mean(edges[ring])) if np.any(ring) else float(np.mean(edges))
    contrast = inside_amp - ring_amp
    edge_contrast = inside_edge - ring_edge
    penalty = 0.0008 * (row_shift * row_shift + col_shift * col_shift)
    return {
        "score": 100.0 * contrast + 45.0 * edge_contrast - penalty,
        "inside_amp": inside_amp,
        "ring_amp": ring_amp,
        "inside_edge": inside_edge,
        "ring_edge": ring_edge,
        "pixels": n,
    }


def search(mask: np.ndarray, amp: np.ndarray, edges: np.ndarray, max_shift: int, coarse_step: int) -> tuple[int, int, dict, list[dict]]:
    rows = []
    best = (-10**9, 0, 0, {})
    for dr in range(-max_shift, max_shift + 1, coarse_step):
        for dc in range(-max_shift, max_shift + 1, coarse_step):
            s = score_shift(mask, amp, edges, dr, dc)
            rows.append({"stage": "coarse", "row_shift": dr, "col_shift": dc, **s})
            if s["score"] > best[0]:
                best = (s["score"], dr, dc, s)
    _, br, bc, _ = best
    for dr in range(br - coarse_step, br + coarse_step + 1):
        for dc in range(bc - coarse_step, bc + coarse_step + 1):
            s = score_shift(mask, amp, edges, dr, dc)
            rows.append({"stage": "fine", "row_shift": dr, "col_shift": dc, **s})
            if s["score"] > best[0]:
                best = (s["score"], dr, dc, s)
    _, br, bc, bs = best
    return br, bc, bs, rows


def shift_geojson(in_path: Path, out_path: Path, row_shift: int, col_shift: int) -> int:
    data = json.loads(in_path.read_text(encoding="utf-8"))
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue
        shifted_rings = []
        for ring in geom.get("coordinates", []):
            shifted_rings.append([[float(x) + col_shift, float(y) + row_shift] for x, y, *rest in ring])
        geom["coordinates"] = shifted_rings
        props = feat.setdefault("properties", {})
        props["sar_brightness_opt_row_shift"] = row_shift
        props["sar_brightness_opt_col_shift"] = col_shift
    data["sar_brightness_optimization"] = {"row_shift": row_shift, "col_shift": col_shift}
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(data.get("features", []))


def shift_metrics(in_path: Path, out_path: Path, row_shift: int, col_shift: int) -> int:
    rows = list(csv.DictReader(in_path.open("r", encoding="utf-8", newline="")))
    for row in rows:
        source_row_keys = ["corrected_row_min", "corrected_row_max", "corrected_center_row"]
        source_col_keys = ["corrected_col_min", "corrected_col_max", "corrected_center_col"]
        for key in source_row_keys:
            row[f"brightness_opt_{key}"] = float(row[key]) + row_shift
        for key in source_col_keys:
            row[f"brightness_opt_{key}"] = float(row[key]) + col_shift
        row["sar_brightness_opt_row_shift"] = row_shift
        row["sar_brightness_opt_col_shift"] = col_shift
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = WORK_ROOT / args.date
    mli_par = work / f"{args.date}.mli.par"
    mli = work / f"{args.date}.mli"
    cols = par_value(mli_par, "range_samples", int)
    rows = par_value(mli_par, "azimuth_lines", int)
    amp = stretch_amp(read_real4(mli, cols, rows))
    edges = edge_map(amp)

    corrected_geojson = Path(args.corrected_geojson)
    corrected_metrics = Path(args.corrected_metrics)
    polys = load_surface_polygons(corrected_geojson, args.surface, args.min_mask_pixels)
    if not polys:
        raise RuntimeError(f"No {args.surface} polygons selected for optimization")
    mask = rasterize_polygons(polys, rows, cols)
    base = score_shift(mask, amp, edges, 0, 0)
    best_dr, best_dc, best, table = search(mask, amp, edges, args.max_shift, args.coarse_step)

    search_csv = out_dir / f"{args.date}_sar_brightness_shift_search.csv"
    with search_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)

    out_geojson = out_dir / f"{args.date}_full_area_projection_sar_col_row_brightness_optimized.geojson"
    out_metrics = out_dir / f"{args.date}_full_area_projection_metrics_brightness_optimized.csv"
    n_features = shift_geojson(corrected_geojson, out_geojson, best_dr, best_dc)
    n_rows = shift_metrics(corrected_metrics, out_metrics, best_dr, best_dc)
    summary = {
        "date": args.date,
        "input_corrected_geojson": str(corrected_geojson),
        "optimized_surface": args.surface,
        "selected_polygons": len(polys),
        "base_score": base["score"],
        "best_score": best["score"],
        "score_gain": best["score"] - base["score"],
        "additional_row_shift": best_dr,
        "additional_col_shift": best_dc,
        "best_inside_amp": best["inside_amp"],
        "best_ring_amp": best["ring_amp"],
        "best_inside_edge": best["inside_edge"],
        "optimized_features": n_features,
        "optimized_metric_rows": n_rows,
        "optimized_geojson": str(out_geojson),
        "optimized_metrics_csv": str(out_metrics),
        "search_csv": str(search_csv),
        "note": "Additional integer SAR-coordinate shift optimized directly against real MLI amplitude. Positive row is down; positive col is right.",
    }
    summary_path = out_dir / f"{args.date}_sar_brightness_projection_optimization_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)

    if args.plot:
        cmd = [
            "/usr/bin/python3",
            str(TOUYING_DIR / "code" / "plot_corrected_projection.py"),
            "--date",
            args.date,
            "--corrected-geojson",
            str(out_geojson),
            "--out-dir",
            str(out_dir),
            "--surfaces",
            args.surface,
        ]
        subprocess.run(cmd, check=True)
        default_png = out_dir / f"{args.date}_full_area_projection_corrected_overlay.png"
        renamed = out_dir / f"{args.date}_full_area_projection_brightness_optimized_overlay.png"
        if default_png.exists():
            default_png.replace(renamed)
            print(str(renamed), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize corrected projection by direct SAR amplitude contrast.")
    parser.add_argument("--date", default="20200708")
    parser.add_argument(
        "--corrected-geojson",
        default=str(TOUYING_DIR / "results" / "full_area_projection_corrected" / "20200708_full_area_projection_sar_col_row_corrected.geojson"),
    )
    parser.add_argument(
        "--corrected-metrics",
        default=str(TOUYING_DIR / "results" / "full_area_projection_corrected" / "20200708_full_area_projection_metrics_corrected.csv"),
    )
    parser.add_argument("--out-dir", default=str(TOUYING_DIR / "results" / "full_area_projection_brightness_optimized"))
    parser.add_argument("--max-shift", type=int, default=60)
    parser.add_argument("--coarse-step", type=int, default=3)
    parser.add_argument("--min-mask-pixels", type=int, default=4)
    parser.add_argument("--surface", default="roof", choices=["roof", "bottom"])
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
