from __future__ import annotations

import argparse
import json
import os
import re
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
WORKSPACE = PROJECT_DIR.parent
WORK_ROOT = WORKSPACE / "geocoding" / "results" / "outputs" / "work" / "gamma_dsm_geocode"
sys.path.insert(0, str(WORKSPACE / "geocoding" / "code"))
from chinese_matplotlib import install_chinese_labels  # noqa: E402
install_chinese_labels()


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


def load_polygons(path: Path) -> list[tuple[str, np.ndarray, dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        xy = np.asarray(coords[0], dtype=np.float64)
        if xy.shape[0] > 1 and np.allclose(xy[0], xy[-1]):
            xy = xy[:-1]
        props = feat.get("properties", {})
        out.append((str(props.get("surface", "")), xy, props))
    return out


def plot(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = WORK_ROOT / args.date
    mli_par = work / f"{args.date}.mli.par"
    mli = work / f"{args.date}.mli"
    cols = par_value(mli_par, "range_samples", int)
    rows = par_value(mli_par, "azimuth_lines", int)
    bg = stretch_amp(read_real4(mli, cols, rows))

    surface_filter = set(args.surfaces.split(",")) if args.surfaces else None
    corrected = [
        item
        for item in load_polygons(Path(args.corrected_geojson))
        if surface_filter is None or item[0] in surface_filter
    ]
    if not corrected:
        raise RuntimeError(f"No polygon features in {args.corrected_geojson}")

    fig, ax = plt.subplots(figsize=(11.0, 8.0), dpi=300)
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1)
    all_xy = []
    for surface, xy, props in corrected:
        all_xy.append(xy)
        if surface == "roof":
            color = "#ffb000"
            lw = 0.24
            alpha = 0.64
        else:
            color = "#00d4ff"
            lw = 0.34 if surface_filter == {"bottom"} else 0.20
            alpha = 0.62 if surface_filter == {"bottom"} else 0.32
        ax.add_patch(MplPolygon(xy, closed=True, fill=False, edgecolor=color, linewidth=lw, alpha=alpha))

    xy_all = np.vstack(all_xy)
    finite = np.all(np.isfinite(xy_all), axis=1)
    xy_all = xy_all[finite]
    xmin = max(0, float(np.nanmin(xy_all[:, 0])) - 80)
    xmax = min(cols, float(np.nanmax(xy_all[:, 0])) + 80)
    ymin = max(0, float(np.nanmin(xy_all[:, 1])) - 80)
    ymax = min(rows, float(np.nanmax(xy_all[:, 1])) + 80)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_xlabel("距离向列号")
    ax.set_ylabel("方位向行号")
    ax.set_title(f"同济校区全区建筑校正投影（{args.date}）")
    ax.text(
        0.012,
        0.988,
        f"要素数：{len(corrected)}\n{args.surfaces or '青色：底面，琥珀色：屋顶'}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3},
    )
    fig.tight_layout()
    out_png = out_dir / f"{args.date}_full_area_projection_corrected_overlay.png"
    fig.savefig(out_png)
    plt.close(fig)
    print(out_png)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot corrected Tongji full-area SAR-coordinate building projection.")
    parser.add_argument("--date", default="20200708")
    parser.add_argument(
        "--corrected-geojson",
        default=str(TOUYING_DIR / "results" / "full_area_projection_corrected" / "20200708_full_area_projection_sar_col_row_corrected.geojson"),
    )
    parser.add_argument("--out-dir", default=str(TOUYING_DIR / "results" / "full_area_projection_corrected"))
    parser.add_argument("--surfaces", default="", help="Comma-separated surface filter, e.g. bottom or roof,bottom.")
    plot(parser.parse_args())


if __name__ == "__main__":
    main()
