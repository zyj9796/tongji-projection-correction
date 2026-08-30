from __future__ import annotations

import argparse
import json
import math
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
from osgeo import gdal


TOUYING_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = TOUYING_DIR.parent
WORKSPACE = PROJECT_DIR.parent
WORK_ROOT = WORKSPACE / "geocoding" / "results" / "outputs" / "work" / "gamma_dsm_geocode"
RASTER_ROOT = WORKSPACE / "geocoding" / "results" / "outputs" / "rasters" / "main"
sys.path.insert(0, str(WORKSPACE / "geocoding" / "code"))
from chinese_matplotlib import install_chinese_labels  # noqa: E402
install_chinese_labels()


def par_value(path: Path, key: str, cast=float):
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(rf"^{re.escape(key)}:\s+([^\s]+)", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing {key} in {path}")
    return cast(float(match.group(1))) if cast is int else cast(match.group(1))


def read_gamma_real4(path: Path, width: int, nlines: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=">f4")
    if arr.size != width * nlines:
        raise ValueError(f"Unexpected size for {path}: {arr.size}, expected {width * nlines}")
    return arr.reshape(nlines, width).astype(np.float32)


def read_lut(path: Path, width: int, nlines: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.fromfile(path, dtype=">f4")
    if arr.size != width * nlines * 2:
        raise ValueError(f"Unexpected LUT size for {path}: {arr.size}, expected {width * nlines * 2}")
    lut = arr.reshape(nlines, width, 2).astype(np.float32)
    return lut[:, :, 0], lut[:, :, 1]


def robust01(arr: np.ndarray, mask: np.ndarray | None = None, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    data = arr[np.isfinite(arr)] if mask is None else arr[mask & np.isfinite(arr)]
    data = data[data > 0]
    if data.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    p_lo, p_hi = np.percentile(data, [lo, hi])
    out = np.clip((arr.astype(np.float32) - p_lo) / max(float(p_hi - p_lo), 1e-6), 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def dsm_backscatter_proxy(dem: np.ndarray, incidence_deg: float) -> np.ndarray:
    dem = dem.astype(np.float32)
    valid = np.isfinite(dem) & (dem > -1000)
    filled = dem.copy()
    fill_value = float(np.nanmedian(filled[valid])) if np.any(valid) else 0.0
    filled[~valid] = fill_value
    gy, gx = np.gradient(filled)
    slope = np.hypot(gx, gy)
    relief = np.abs(filled - np.nanmedian(filled[valid])) if np.any(valid) else np.zeros_like(filled)
    inc = math.radians(incidence_deg)
    slope_term = np.clip(slope / max(np.percentile(slope[valid], 98), 1e-6), 0.0, 1.0) if np.any(valid) else slope
    relief_term = np.clip(relief / max(np.percentile(relief[valid], 98), 1e-6), 0.0, 1.0) if np.any(valid) else relief
    proxy = 0.65 * slope_term + 0.25 * relief_term + 0.10 * np.sin(inc)
    proxy[~valid] = 0.0
    return robust01(proxy, valid)


def splat_map_to_radar(
    map_img: np.ndarray,
    range_col: np.ndarray,
    az_row: np.ndarray,
    radar_rows: int,
    radar_cols: int,
) -> np.ndarray:
    col = range_col.ravel().astype(np.float64)
    row = az_row.ravel().astype(np.float64)
    val = map_img.ravel().astype(np.float64)
    ok = np.isfinite(col) & np.isfinite(row) & np.isfinite(val) & (val > 0)
    ok &= (col >= 0) & (row >= 0) & (col < radar_cols - 1) & (row < radar_rows - 1)
    col = col[ok]
    row = row[ok]
    val = val[ok]
    c0 = np.floor(col).astype(np.int64)
    r0 = np.floor(row).astype(np.int64)
    dc = col - c0
    dr = row - r0
    out = np.zeros((radar_rows, radar_cols), dtype=np.float64)
    wsum = np.zeros((radar_rows, radar_cols), dtype=np.float64)
    for rr, cc, ww in [
        (r0, c0, (1.0 - dr) * (1.0 - dc)),
        (r0, c0 + 1, (1.0 - dr) * dc),
        (r0 + 1, c0, dr * (1.0 - dc)),
        (r0 + 1, c0 + 1, dr * dc),
    ]:
        np.add.at(out, (rr, cc), val * ww)
        np.add.at(wsum, (rr, cc), ww)
    ok_out = wsum > 0
    out[ok_out] /= wsum[ok_out]
    return robust01(out.astype(np.float32), ok_out)


def highpass_edges(img: np.ndarray) -> np.ndarray:
    img = robust01(img)
    gy, gx = np.gradient(img.astype(np.float32))
    edge = np.hypot(gx, gy)
    return robust01(edge)


def phase_correlation_shift(reference: np.ndarray, moving: np.ndarray) -> dict:
    ref = highpass_edges(reference)
    mov = highpass_edges(moving)
    mask = (ref > 0) & (mov > 0)
    if int(mask.sum()) < 100:
        mask = np.ones_like(ref, dtype=bool)
    ref = (ref - float(ref[mask].mean())) * mask
    mov = (mov - float(mov[mask].mean())) * mask
    win_y = np.hanning(ref.shape[0]).astype(np.float32)
    win_x = np.hanning(ref.shape[1]).astype(np.float32)
    win = win_y[:, None] * win_x[None, :]
    f_ref = np.fft.fft2(ref * win)
    f_mov = np.fft.fft2(mov * win)
    cps = f_ref * np.conj(f_mov)
    cps /= np.maximum(np.abs(cps), 1e-12)
    corr = np.fft.ifft2(cps).real
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    row_shift = float(peak[0])
    col_shift = float(peak[1])
    if row_shift > ref.shape[0] / 2:
        row_shift -= ref.shape[0]
    if col_shift > ref.shape[1] / 2:
        col_shift -= ref.shape[1]

    # Sub-pixel parabolic refinement around the integer peak.
    pr, pc = peak
    def refine(axis_vals: tuple[float, float, float]) -> float:
        a, b, c = axis_vals
        denom = a - 2.0 * b + c
        if abs(denom) < 1e-12:
            return 0.0
        return float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))

    row_delta = refine((corr[(pr - 1) % corr.shape[0], pc], corr[pr, pc], corr[(pr + 1) % corr.shape[0], pc]))
    col_delta = refine((corr[pr, (pc - 1) % corr.shape[1]], corr[pr, pc], corr[pr, (pc + 1) % corr.shape[1]]))
    peak_value = float(corr[peak])
    noise = np.delete(corr.ravel(), int(np.argmax(corr)))
    snr = float((peak_value - float(np.mean(noise))) / max(float(np.std(noise)), 1e-12))
    return {
        "row_shift_to_apply_to_simulated": row_shift + row_delta,
        "col_shift_to_apply_to_simulated": col_shift + col_delta,
        "integer_peak_row": int(peak[0]),
        "integer_peak_col": int(peak[1]),
        "peak_value": peak_value,
        "phase_corr_snr": snr,
    }


def write_tif(path: Path, arr: np.ndarray) -> None:
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32, options=["COMPRESS=LZW", "TILED=YES"])
    ds.GetRasterBand(1).WriteArray(arr.astype(np.float32))
    ds.FlushCache()
    ds = None


def make_png(path: Path, real: np.ndarray, sim: np.ndarray) -> None:
    real_show = robust01(np.sqrt(np.maximum(real, 0.0)))
    sim_show = robust01(sim)
    diff = np.clip(real_show - sim_show + 0.5, 0.0, 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0), dpi=220)
    for ax, img, title, cmap in [
        (axes[0], real_show, "实测雷达复数影像／多视影像幅度", "gray"),
        (axes[1], sim_show, "数字表面模型模拟雷达代理影像", "gray"),
        (axes[2], diff, "差异代理影像", "coolwarm"),
    ]:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = WORK_ROOT / args.date
    dem_par = work / f"{args.date}.dem_seg.par"
    mli_par = work / f"{args.date}.mli.par"
    dem_path = work / f"{args.date}.dem_seg"
    lut_path = work / f"{args.date}.lt"
    mli_path = work / f"{args.date}.mli"
    for path in [dem_par, mli_par, dem_path, lut_path, mli_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing GAMMA work file: {path}. Run bash run.sh or code/gamma_dsm_geocode.py first.")

    map_width = par_value(dem_par, "width", int)
    map_nlines = par_value(dem_par, "nlines", int)
    radar_cols = par_value(mli_par, "range_samples", int)
    radar_rows = par_value(mli_par, "azimuth_lines", int)
    incidence = par_value(mli_par, "incidence_angle", float)

    dem = read_gamma_real4(dem_path, map_width, map_nlines)
    range_col, az_row = read_lut(lut_path, map_width, map_nlines)
    map_proxy = dsm_backscatter_proxy(dem, incidence)
    sim_radar = splat_map_to_radar(map_proxy, range_col, az_row, radar_rows, radar_cols)
    real_mli = read_gamma_real4(mli_path, radar_cols, radar_rows)
    shift = phase_correlation_shift(np.sqrt(np.maximum(real_mli, 0.0)), sim_radar)

    sim_tif = out_dir / f"{args.date}_dsm_simulated_sar_radar.tif"
    map_tif = out_dir / f"{args.date}_dsm_backscatter_proxy_map.tif"
    png = out_dir / f"{args.date}_dsm_simulated_sar_vs_real.png"
    summary_path = out_dir / f"{args.date}_dsm_simulated_sar_correction.json"
    write_tif(sim_tif, sim_radar)
    write_tif(map_tif, map_proxy)
    make_png(png, real_mli, sim_radar)

    summary = {
        "date": args.date,
        "method": "DSM slope/relief proxy projected to radar coordinates with GAMMA gc_map2 lookup table, then phase-correlated with real MLI amplitude.",
        "radar_rows": radar_rows,
        "radar_cols": radar_cols,
        "map_width": map_width,
        "map_nlines": map_nlines,
        **shift,
        "interpretation": "Apply row_shift/col_shift to the DSM simulated image or to projection overlays to align them to the real SAR amplitude. Positive row means down; positive col means right.",
        "simulated_radar_tif": str(sim_tif),
        "map_proxy_tif": str(map_tif),
        "comparison_png": str(png),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DSM simulated SAR proxy and estimate radar-coordinate correction.")
    parser.add_argument("--date", default="20200708")
    parser.add_argument("--out-dir", default=str(TOUYING_DIR / "results" / "dsm_simulated_sar"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
