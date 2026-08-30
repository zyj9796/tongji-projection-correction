from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


TOUYING_DIR = Path(__file__).resolve().parents[1]


def shift_geometry(geom: dict, col_shift: float, row_shift: float) -> dict:
    if geom.get("type") != "Polygon":
        return geom
    rings = []
    for ring in geom.get("coordinates", []):
        rings.append([[float(x) + col_shift, float(y) + row_shift] for x, y, *rest in ring])
    return {"type": "Polygon", "coordinates": rings}


def apply_geojson(in_path: Path, out_path: Path, row_shift: float, col_shift: float) -> int:
    data = json.loads(in_path.read_text(encoding="utf-8"))
    for feat in data.get("features", []):
        feat["geometry"] = shift_geometry(feat.get("geometry", {}), col_shift, row_shift)
        props = feat.setdefault("properties", {})
        props["correction_row_shift"] = row_shift
        props["correction_col_shift"] = col_shift
        props["correction_applied"] = 1
    data["correction"] = {
        "row_shift_applied": row_shift,
        "col_shift_applied": col_shift,
        "coordinate_system": "SAR image coordinates: x=range column, y=azimuth row",
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(data.get("features", []))


def apply_metrics(in_path: Path, out_path: Path, row_shift: float, col_shift: float) -> int:
    with in_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ["row_min", "row_max", "center_row"]:
            row[f"corrected_{key}"] = float(row[key]) + row_shift
        for key in ["col_min", "col_max", "center_col"]:
            row[f"corrected_{key}"] = float(row[key]) + col_shift
        row["correction_row_shift"] = row_shift
        row["correction_col_shift"] = col_shift
    if rows:
        fields = list(rows[0].keys())
    else:
        fields = []
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def apply_npz(in_path: Path, out_path: Path, row_shift: float, col_shift: float) -> None:
    data = np.load(in_path, allow_pickle=False)
    arrays = {}
    for key in data.files:
        arr = data[key]
        if key.endswith("_projected_rc"):
            shifted = np.asarray(arr, dtype=np.float64).copy()
            shifted[:, 0] += row_shift
            shifted[:, 1] += col_shift
            arrays[key] = shifted
        else:
            arrays[key] = arr
    arrays["correction_json"] = np.asarray(
        json.dumps({"row_shift_applied": row_shift, "col_shift_applied": col_shift}, ensure_ascii=False)
    )
    np.savez_compressed(out_path, **arrays)


def run(args: argparse.Namespace) -> None:
    correction = json.loads(Path(args.correction_json).read_text(encoding="utf-8"))
    row_shift = float(correction["row_shift_to_apply_to_simulated"])
    col_shift = float(correction["col_shift_to_apply_to_simulated"])
    in_dir = Path(args.projection_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geojson_in = in_dir / f"{args.date}_full_area_projection_sar_col_row.geojson"
    metrics_in = in_dir / f"{args.date}_full_area_projection_metrics.csv"
    npz_in = in_dir / f"{args.date}_full_area_projection_triangles.npz"

    geojson_out = out_dir / f"{args.date}_full_area_projection_sar_col_row_corrected.geojson"
    metrics_out = out_dir / f"{args.date}_full_area_projection_metrics_corrected.csv"
    npz_out = out_dir / f"{args.date}_full_area_projection_triangles_corrected.npz"

    n_features = apply_geojson(geojson_in, geojson_out, row_shift, col_shift)
    n_rows = apply_metrics(metrics_in, metrics_out, row_shift, col_shift)
    apply_npz(npz_in, npz_out, row_shift, col_shift)

    summary = {
        "date": args.date,
        "row_shift_applied": row_shift,
        "col_shift_applied": col_shift,
        "source_correction_json": str(Path(args.correction_json)),
        "source_projection_dir": str(in_dir),
        "corrected_features": n_features,
        "corrected_metric_rows": n_rows,
        "corrected_geojson": str(geojson_out),
        "corrected_metrics_csv": str(metrics_out),
        "corrected_triangles_npz": str(npz_out),
        "note": "Correction is applied in SAR radar coordinates only: x=range column, y=azimuth row.",
    }
    (out_dir / f"{args.date}_full_area_projection_correction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply DSM simulated SAR correction to full-area SAR projection outputs.")
    parser.add_argument("--date", default="20200708")
    parser.add_argument("--projection-dir", default=str(TOUYING_DIR / "results" / "full_area_projection"))
    parser.add_argument("--correction-json", default=str(TOUYING_DIR / "results" / "dsm_simulated_sar" / "20200708_dsm_simulated_sar_correction.json"))
    parser.add_argument("--out-dir", default=str(TOUYING_DIR / "results" / "full_area_projection_corrected"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
