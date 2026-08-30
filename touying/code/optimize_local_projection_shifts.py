from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from collections import defaultdict
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


def polygon_array(feat: dict) -> np.ndarray | None:
    geom = feat.get("geometry", {})
    coords = geom.get("coordinates", [])
    if geom.get("type") != "Polygon" or not coords:
        return None
    xy = np.asarray(coords[0], dtype=np.float64)
    if xy.shape[0] > 1 and np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    if xy.shape[0] < 3 or not np.all(np.isfinite(xy)):
        return None
    return xy


def rasterize_local_polygon(xy: np.ndarray, image_rows: int, image_cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xmin = max(0, int(np.floor(np.min(xy[:, 0]))) - 8)
    xmax = min(image_cols - 1, int(np.ceil(np.max(xy[:, 0]))) + 8)
    ymin = max(0, int(np.floor(np.min(xy[:, 1]))) - 8)
    ymax = min(image_rows - 1, int(np.ceil(np.max(xy[:, 1]))) + 8)
    if xmax < xmin or ymax < ymin:
        z = np.zeros((0,), dtype=np.int64)
        return z, z, z, z
    yy, xx = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
    inside = MplPath(xy).contains_points(np.column_stack([xx.ravel(), yy.ravel()])).reshape(yy.shape)
    if not np.any(inside):
        z = np.zeros((0,), dtype=np.int64)
        return z, z, z, z
    ring = binary_dilation(inside, iterations=5) & ~binary_dilation(inside, iterations=1)
    rr_in, cc_in = np.nonzero(inside)
    rr_ring, cc_ring = np.nonzero(ring)
    return rr_in + ymin, cc_in + xmin, rr_ring + ymin, cc_ring + xmin


def shifted_values(img: np.ndarray, rr: np.ndarray, cc: np.ndarray, dr: int, dc: int) -> np.ndarray:
    r = rr + dr
    c = cc + dc
    ok = (r >= 0) & (c >= 0) & (r < img.shape[0]) & (c < img.shape[1])
    if not np.any(ok):
        return np.zeros((0,), dtype=np.float32)
    return img[r[ok], c[ok]]


def score_shift(
    amp: np.ndarray,
    edges: np.ndarray,
    rr_in: np.ndarray,
    cc_in: np.ndarray,
    rr_ring: np.ndarray,
    cc_ring: np.ndarray,
    dr: int,
    dc: int,
    max_shift: int,
) -> dict:
    inside_amp_vals = shifted_values(amp, rr_in, cc_in, dr, dc)
    if inside_amp_vals.size < 4:
        return {"score": -1e9, "inside_amp": 0.0, "ring_amp": 0.0, "inside_edge": 0.0, "ring_edge": 0.0}
    ring_amp_vals = shifted_values(amp, rr_ring, cc_ring, dr, dc)
    inside_edge_vals = shifted_values(edges, rr_in, cc_in, dr, dc)
    ring_edge_vals = shifted_values(edges, rr_ring, cc_ring, dr, dc)
    inside_amp = float(np.mean(inside_amp_vals))
    ring_amp = float(np.mean(ring_amp_vals)) if ring_amp_vals.size else float(np.mean(amp))
    inside_edge = float(np.mean(inside_edge_vals)) if inside_edge_vals.size else 0.0
    ring_edge = float(np.mean(ring_edge_vals)) if ring_edge_vals.size else float(np.mean(edges))
    offset_penalty = 0.0015 * (dr * dr + dc * dc) / max(max_shift, 1)
    score = 100.0 * (inside_amp - ring_amp) + 35.0 * (inside_edge - ring_edge) - offset_penalty
    return {
        "score": score,
        "inside_amp": inside_amp,
        "ring_amp": ring_amp,
        "inside_edge": inside_edge,
        "ring_edge": ring_edge,
    }


def optimize_one(
    amp: np.ndarray,
    edges: np.ndarray,
    xy: np.ndarray,
    max_shift: int,
    coarse_step: int,
) -> dict:
    rr_in, cc_in, rr_ring, cc_ring = rasterize_local_polygon(xy, amp.shape[0], amp.shape[1])
    if rr_in.size < 4:
        return {"ok": 0, "row_shift": 0, "col_shift": 0, "pixels": int(rr_in.size), "base_score": -1e9, "best_score": -1e9}
    best = {"score": -1e9, "row_shift": 0, "col_shift": 0}
    base = score_shift(amp, edges, rr_in, cc_in, rr_ring, cc_ring, 0, 0, max_shift)
    for dr in range(-max_shift, max_shift + 1, coarse_step):
        for dc in range(-max_shift, max_shift + 1, coarse_step):
            s = score_shift(amp, edges, rr_in, cc_in, rr_ring, cc_ring, dr, dc, max_shift)
            if s["score"] > best["score"]:
                best = {**s, "row_shift": dr, "col_shift": dc}
    br = int(best["row_shift"])
    bc = int(best["col_shift"])
    for dr in range(br - coarse_step, br + coarse_step + 1):
        for dc in range(bc - coarse_step, bc + coarse_step + 1):
            if abs(dr) > max_shift or abs(dc) > max_shift:
                continue
            s = score_shift(amp, edges, rr_in, cc_in, rr_ring, cc_ring, dr, dc, max_shift)
            if s["score"] > best["score"]:
                best = {**s, "row_shift": dr, "col_shift": dc}
    return {
        "ok": 1,
        "pixels": int(rr_in.size),
        "base_score": float(base["score"]),
        "best_score": float(best["score"]),
        "score_gain": float(best["score"] - base["score"]),
        "row_shift": int(best["row_shift"]),
        "col_shift": int(best["col_shift"]),
        "inside_amp": float(best["inside_amp"]),
        "ring_amp": float(best["ring_amp"]),
        "inside_edge": float(best["inside_edge"]),
        "ring_edge": float(best["ring_edge"]),
    }


def shift_feature(feat: dict, row_shift: int, col_shift: int) -> None:
    geom = feat.get("geometry", {})
    if geom.get("type") != "Polygon":
        return
    rings = []
    for ring in geom.get("coordinates", []):
        rings.append([[float(x) + col_shift, float(y) + row_shift] for x, y, *rest in ring])
    geom["coordinates"] = rings
    props = feat.setdefault("properties", {})
    props["local_opt_row_shift"] = row_shift
    props["local_opt_col_shift"] = col_shift


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

    data = json.loads(Path(args.input_geojson).read_text(encoding="utf-8"))
    by_fid: dict[int, list[dict]] = defaultdict(list)
    roof_by_fid: dict[int, dict] = {}
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        fid = int(props.get("fid", -1))
        if fid < 0:
            continue
        by_fid[fid].append(feat)
        if props.get("surface") == "roof" and int(float(props.get("mask0_pixels", 0))) >= args.min_mask_pixels:
            roof_by_fid[fid] = feat

    rows_out = []
    for i, (fid, roof) in enumerate(sorted(roof_by_fid.items()), start=1):
        xy = polygon_array(roof)
        if xy is None:
            result = {"ok": 0, "row_shift": 0, "col_shift": 0, "pixels": 0, "base_score": -1e9, "best_score": -1e9}
        else:
            result = optimize_one(amp, edges, xy, args.max_shift, args.coarse_step)
        for feat in by_fid.get(fid, []):
            shift_feature(feat, int(result["row_shift"]), int(result["col_shift"]))
        props = roof.get("properties", {})
        rows_out.append(
            {
                "fid": fid,
                "height_m": float(props.get("height_m", 0.0)),
                "mask0_pixels": int(float(props.get("mask0_pixels", 0))),
                "mask_pixels": int(float(props.get("mask_pixels", 0))),
                **result,
            }
        )
        if i % 100 == 0:
            print(f"local optimized {i}/{len(roof_by_fid)}", flush=True)

    data["local_projection_optimization"] = {
        "max_shift": args.max_shift,
        "coarse_step": args.coarse_step,
        "optimized_fids": len(rows_out),
        "coordinate_system": "SAR image coordinates: x=range column, y=azimuth row",
    }
    out_geojson = out_dir / f"{args.date}_full_area_projection_sar_col_row_local_optimized.geojson"
    out_geojson.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_csv = out_dir / f"{args.date}_local_projection_shift_metrics.csv"
    fieldnames = []
    seen = set()
    for row in rows_out:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    ok_rows = [r for r in rows_out if int(r["ok"]) == 1]
    summary = {
        "date": args.date,
        "optimized_buildings": len(rows_out),
        "ok_buildings": len(ok_rows),
        "median_row_shift": float(np.median([r["row_shift"] for r in ok_rows])) if ok_rows else 0.0,
        "median_col_shift": float(np.median([r["col_shift"] for r in ok_rows])) if ok_rows else 0.0,
        "median_score_gain": float(np.median([r["score_gain"] for r in ok_rows])) if ok_rows else 0.0,
        "mean_score_gain": float(np.mean([r["score_gain"] for r in ok_rows])) if ok_rows else 0.0,
        "local_optimized_geojson": str(out_geojson),
        "metrics_csv": str(metrics_csv),
        "note": "Per-building integer SAR-coordinate shifts optimized against real MLI amplitude. Shift is applied to both roof and bottom polygons for the same FID.",
    }
    summary_path = out_dir / f"{args.date}_local_projection_optimization_summary.json"
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
        ]
        subprocess.run(cmd, check=True)
        default_png = out_dir / f"{args.date}_full_area_projection_corrected_overlay.png"
        renamed = out_dir / f"{args.date}_full_area_projection_local_optimized_overlay.png"
        if default_png.exists():
            default_png.replace(renamed)
            print(str(renamed), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize local per-building projection shifts in SAR coordinates.")
    parser.add_argument("--date", default="20200708")
    parser.add_argument(
        "--input-geojson",
        default=str(
            TOUYING_DIR
            / "results"
            / "full_area_projection_brightness_optimized"
            / "20200708_full_area_projection_sar_col_row_brightness_optimized.geojson"
        ),
    )
    parser.add_argument("--out-dir", default=str(TOUYING_DIR / "results" / "full_area_projection_local_optimized"))
    parser.add_argument("--max-shift", type=int, default=18)
    parser.add_argument("--coarse-step", type=int, default=3)
    parser.add_argument("--min-mask-pixels", type=int, default=4)
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
