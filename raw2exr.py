#!/usr/bin/env python3
"""
raw2exr — camera raw → scene-linear, camera-native RGB OpenEXR (batch).

What the numbers in the EXR mean, per pixel and channel c:

    value_c = (raw_c − black_c) / (white − min(black)) · scale

  raw_c   the ADC count of the sensel behind colour filter c (LibRaw decode only:
          container unpacking, vendor decompression/linearisation curves, masked-
          area crop — nothing "creative").
  black_c the per-channel black level ("floor"). This is the only thing subtracted.
  white   the sensor saturation level. One common denominator for all channels,
          so every channel gets the *same* gain — no accidental white balance.
          1.0 (× scale) = sensor saturation.
  scale   optional global exposure factor (see --scale). Default 1.

  Demosaic only fills in the two missing channels at each site; the original
  samples are preserved bit-exactly (verified on every file with --verify,
  which is on by default). --binned skips demosaic entirely (2×2 / 3×3 bin).

  Camera-native data is strongly GREEN by nature (the green channel collects the
  most light); that is the correct, unbalanced signal. --wb can bake in a white
  balance (as-shot / daylight / calibration / custom); the gains and the matrices
  re-expressed for balanced data are then written to the header.

  NOT applied by default: white balance, colour matrix, gamma/tone curve, auto-brightening,
  highlight recovery, noise reduction, chromatic-aberration correction.
  Sub-floor noise (values < 0 after black subtraction) is clipped to 0 by LibRaw
  in demosaic mode; --binned mode can keep it (--keep-negatives).

Colour: the data is in the camera's own RGB space. The XYZ→camera matrix that
ships with the raw (Adobe/LibRaw table, or the DNG ColorMatrix tags) is stored
in the EXR header as-is, together with its inverse (camera→XYZ) and
camera→Rec.709-linear, the camera's response to the calibration white
(divide by it to white-balance to that illuminant), as-shot and daylight WB
multipliers, and everything else needed to reproduce or invert the pipeline.
No `chromaticities` attribute is written — the data is not in a colorimetric
RGB space, so none would be correct.

Dependencies:  pip install rawpy OpenEXR numpy      (required)
               pip install exifread tifffile        (optional: EXIF / DNG tags)
               exiftool on PATH                     (optional, best EXIF source)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

__version__ = "1.1.0"

RAW_EXTS = {
    ".3fr", ".ari", ".arw", ".bay", ".cr2", ".cr3", ".crw", ".dcr", ".dng", ".erf",
    ".fff", ".iiq", ".kdc", ".mef", ".mos", ".mrw", ".nef", ".nrw", ".orf", ".pef",
    ".raf", ".raw", ".rw2", ".rwl", ".sr2", ".srf", ".srw", ".x3f",
}

# CIE 1931 2° XYZ → linear Rec.709 / sRGB primaries, D65 white (Lindbloom).
XYZ_TO_REC709 = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
])

# EXIF/DNG LightSource codes → (name, XYZ white) for the ones with a defined white.
ILLUMINANTS = {
    17: ("Standard light A", (1.09850, 1.0, 0.35585)),
    20: ("D55", (0.95682, 1.0, 0.92149)),
    21: ("D65", (0.95047, 1.0, 1.08883)),
    22: ("D75", (0.94972, 1.0, 1.22638)),
    23: ("D50", (0.96422, 1.0, 0.82521)),
}
ILLUMINANT_NAMES = {
    1: "Daylight", 2: "Fluorescent", 3: "Tungsten", 4: "Flash", 9: "Fine weather",
    10: "Cloudy", 11: "Shade", 12: "Daylight fluorescent", 13: "Day-white fluorescent",
    14: "Cool-white fluorescent", 15: "White fluorescent", 17: "Standard light A",
    18: "Standard light B", 19: "Standard light C", 20: "D55", 21: "D65", 22: "D75",
    23: "D50", 24: "ISO studio tungsten", 255: "Other",
}

DNG_TAGS = {  # TIFF tag codes read directly from DNG files
    50708: "UniqueCameraModel", 50714: "BlackLevel", 50717: "WhiteLevel",
    50721: "ColorMatrix1", 50722: "ColorMatrix2", 50728: "AsShotNeutral",
    50730: "BaselineExposure", 50778: "CalibrationIlluminant1",
    50779: "CalibrationIlluminant2",
}


# ----------------------------------------------------------------------------- helpers
def apply_flip(img: np.ndarray, flip: int) -> np.ndarray:
    """Apply a LibRaw/dcraw flip code (bit 1: mirror x, bit 2: mirror y, bit 4: transpose).
    Verified identical to LibRaw's own user_flip output for 3 (180°), 5 (90° CCW), 6 (90° CW)."""
    if flip & 2:
        img = img[::-1]
    if flip & 1:
        img = img[:, ::-1]
    if flip & 4:
        img = img.transpose(1, 0, 2)
    return np.ascontiguousarray(img)


def _rational(v):
    """tifffile returns RATIONAL/SRATIONAL as (num, den) or flat (n0,d0,n1,d1,...)."""
    if isinstance(v, (int, float)):
        return float(v)
    v = list(v)
    if len(v) == 2 and all(isinstance(x, (int, np.integer)) for x in v):
        return v[0] / v[1] if v[1] else float("nan")
    if v and isinstance(v[0], (tuple, list)):
        return [_rational(x) for x in v]
    if len(v) % 2 == 0 and all(isinstance(x, (int, np.integer)) for x in v):
        return [v[i] / v[i + 1] if v[i + 1] else float("nan") for i in range(0, len(v), 2)]
    return [float(x) for x in v]


def _iter_boxes(buf: bytes, start: int, end: int):
    """Iterate ISO-BMFF boxes in buf[start:end] → (type, payload_start, box_end)."""
    pos = start
    end = min(end, len(buf))
    while pos + 8 <= end:
        size = int.from_bytes(buf[pos:pos + 4], "big")
        typ = buf[pos + 4:pos + 8]
        hdr = 8
        if size == 1:
            size = int.from_bytes(buf[pos + 8:pos + 16], "big")
            hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr:
            break
        yield typ, pos + hdr, min(pos + size, end)
        pos += size


def _exif_streams(path: Path) -> list[bytes]:
    """Return byte blobs that exifread can parse (TIFF or JPEG) for containers it does not
    understand itself: Canon CR3 (ISO-BMFF: moov/uuid/CMT1+CMT2), Fujifilm RAF (embedded JPEG),
    Minolta MRW (TTW block). Everything else: the file itself."""
    with open(path, "rb") as f:
        head = f.read(4 * 1024 * 1024)
    if head[4:8] == b"ftyp" and head[8:12] in (b"crx ", b"heic", b"heix", b"mif1"):
        blobs = []
        for typ, a, b in _iter_boxes(head, 0, len(head)):
            if typ != b"moov":
                continue
            for t2, c, d in _iter_boxes(head, a, b):
                if t2 != b"uuid":
                    continue
                for t3, e, g in _iter_boxes(head, c + 16, d):
                    if t3 in (b"CMT1", b"CMT2"):
                        blobs.append(head[e:g])
        return blobs
    if head.startswith(b"FUJIFILMCCD-RAW"):
        jpg_off = int.from_bytes(head[84:88], "big")
        jpg_len = int.from_bytes(head[88:92], "big")
        with open(path, "rb") as f:
            f.seek(jpg_off)
            jpg = f.read(jpg_len)
        model = head[28:60].split(b"\0")[0].decode("latin1", "replace").strip()
        return [jpg, b"RAFMODEL:" + model.encode("latin1", "replace")]
    if head[:4] == b"\0MRM":
        pos, blobs = 8, []
        while pos + 8 <= len(head):
            tag = head[pos:pos + 4]
            ln = int.from_bytes(head[pos + 4:pos + 8], "big")
            if tag == b"\0TTW":
                blobs.append(head[pos + 8:pos + 8 + ln])
                break
            if tag == b"\0PRD" or tag[0:1] == b"\0":
                pos += 8 + ln
            else:
                break
        return blobs
    return [None]   # sentinel: parse the file itself


def read_exif(path: Path) -> dict:
    """Best-effort EXIF/DNG metadata: exiftool → exifread (+container unwrapping) + tifffile. Never raises."""
    out: dict = {}
    if shutil.which("exiftool"):
        try:
            j = subprocess.run(["exiftool", "-j", "-n", str(path)], capture_output=True,
                               text=True, timeout=60).stdout
            d = json.loads(j)[0]
            for k in ("Make", "Model", "ISO", "ExposureTime", "FNumber", "ExposureCompensation",
                      "DateTimeOriginal", "LensModel", "Orientation", "BaselineExposure",
                      "UniqueCameraModel", "CalibrationIlluminant1", "CalibrationIlluminant2"):
                if k in d:
                    out[k] = d[k]
            for k in ("ColorMatrix1", "ColorMatrix2", "AsShotNeutral"):
                if k in d:
                    v = d[k]
                    out[k] = [float(x) for x in v.split()] if isinstance(v, str) else v
            out["_source"] = "exiftool"
            return out
        except Exception:
            pass
    try:
        import io
        import logging
        import exifread
        logging.getLogger("exifread").setLevel(logging.CRITICAL)   # silence "File format not recognized."
        tags: dict = {}
        for blob in _exif_streams(path):
            if isinstance(blob, bytes) and blob.startswith(b"RAFMODEL:"):
                out.setdefault("Model", blob[9:].decode("latin1"))
                out.setdefault("Make", "FUJIFILM")
                continue
            try:
                if blob is None:
                    with open(path, "rb") as f:
                        tags.update(exifread.process_file(f, details=False))
                else:
                    tags.update(exifread.process_file(io.BytesIO(blob), details=False))
            except Exception:
                continue

        def tag(name):
            for pre in ("EXIF ", "Image ", "MakerNote "):
                if pre + name in tags:
                    return tags[pre + name]
            return None

        def num(name):
            try:
                v = tag(name).values
                v = v[0] if isinstance(v, (list, tuple)) else v
                return float(v)
            except Exception:
                return None

        def txt(name):
            t = tag(name)
            return str(t).strip() if t is not None else None

        for k in ("Make", "Model", "DateTimeOriginal", "LensModel"):
            v = txt(k)
            if v:
                out[k] = v
        for k, t in (("ISO", "ISOSpeedRatings"), ("ExposureTime", "ExposureTime"),
                     ("FNumber", "FNumber"), ("ExposureCompensation", "ExposureBiasValue")):
            v = num(t)
            if v is not None:
                out[k] = v
        out["_source"] = "exifread"
    except Exception:
        pass
    if path.suffix.lower() == ".dng":
        try:
            import tifffile
            with tifffile.TiffFile(str(path)) as t:
                tags = t.pages[0].tags
                for code, name in DNG_TAGS.items():
                    tg = tags.get(code)
                    if tg is None:
                        continue
                    v = tg.value
                    if name in ("ColorMatrix1", "ColorMatrix2", "AsShotNeutral", "BaselineExposure"):
                        v = _rational(v)
                    elif isinstance(v, bytes):
                        v = v.decode("latin1", "replace").rstrip("\x00")
                    elif isinstance(v, (tuple, list)):
                        v = [int(x) for x in v]
                    out[name] = v
            out["_source"] = out.get("_source", "") + "+tifffile"
        except Exception:
            pass
    return out


def scale_factor(mode: str, exif: dict) -> tuple[float, str]:
    """Return (factor, description) for --scale."""
    m = mode.strip().lower()
    if m == "sensor":
        return 1.0, "sensor: 1.0 = sensor saturation (white level)"
    if m == "iso12232":
        return math.sqrt(2.0), ("iso12232: ×√2 — saturation-based ISO puts clipping at 141% of "
                                "diffuse white, so diffuse white → 1.0, 18% grey → 0.18 at nominal exposure")
    if m == "dng":
        be = exif.get("BaselineExposure")
        if be is None:
            raise ValueError("--scale dng needs a DNG BaselineExposure tag (not found)")
        return 2.0 ** float(be), f"dng: ×2^BaselineExposure (BaselineExposure = {float(be):+.3f} EV)"
    if m.startswith("ev:"):
        ev = float(m[3:])
        return 2.0 ** ev, f"ev: ×2^{ev:+g}"
    if m.startswith("mul:"):
        k = float(m[4:])
        return k, f"mul: ×{k:g}"
    if m.startswith("grey:") or m.startswith("gray:"):
        v = float(m.split(":", 1)[1])
        return 0.18 / v, f"grey: 18% card measured at {v:g} (sensor scale) → ×{0.18 / v:.4f} so it lands at 0.18"
    raise ValueError(f"unknown --scale mode {mode!r}")


def detect_cfa(colors_vis: np.ndarray):
    """Return (period, pattern_string). Colour indices: 0=R 1=G 2=B 3=G2."""
    for p in (2, 6):
        tile = colors_vis[:p, :p]
        h = min(colors_vis.shape[0], 4 * p)
        w = min(colors_vis.shape[1], 4 * p)
        if np.array_equal(colors_vis[:h, :w], np.tile(tile, (4, 4))[:h, :w]):
            s = "".join("RGBG"[int(c)] for c in tile.ravel())
            if p == 2:
                return 2, s
            return 6, "/".join(s[i:i + 6] for i in range(0, 36, 6))
    return None, "unknown"


def demosaic_algorithm(name: str):
    import rawpy
    return getattr(rawpy.DemosaicAlgorithm, name.upper())


def block_bin(vals: np.ndarray, colors: np.ndarray, period: int) -> tuple[np.ndarray, int]:
    """Bin the CFA into RGB with no interpolation. Bayer → 2×2, X-Trans → 3×3 (every 3×3 block
    of the 6×6 X-Trans pattern contains all three colours). Returns (H/b, W/b, 3) and b."""
    b = 2 if period == 2 else 3
    H = vals.shape[0] // b * b
    W = vals.shape[1] // b * b
    vals = vals[:H, :W]
    colors = colors[:H, :W]
    out = np.empty((H // b, W // b, 3), np.float64)
    for c in range(3):
        m = (colors == c)
        s = (vals * m).reshape(H // b, b, W // b, b).sum(axis=(1, 3), dtype=np.float64)
        n = m.reshape(H // b, b, W // b, b).sum(axis=(1, 3))
        if (n == 0).any():
            raise RuntimeError("CFA block lacks a colour; cannot bin this pattern")
        out[..., c] = s / n
    return out, b


def recover_black_pattern(raw_vis: np.ndarray, got: np.ndarray, denom: float, factor: float):
    """LibRaw may subtract a per-pixel black *pattern* (cblack[4..]) that rawpy's per-channel
    black levels cannot express. Recover it from LibRaw's own output: at unclipped CFA sites
    black_eff = raw − got·denom/factor. Returns (period, tile[period,period] in counts) or None."""
    H, W = raw_vis.shape
    for r0, c0 in ((0, 0), ((H // 2) // 24 * 24, (W // 2) // 24 * 24), ((H - 96) // 24 * 24, (W - 96) // 24 * 24)):
        if r0 < 0 or c0 < 0:
            continue
        wr = raw_vis[r0:r0 + 96, c0:c0 + 96].astype(np.float64)
        wg = got[r0:r0 + 96, c0:c0 + 96].astype(np.float64)
        valid = (wg > 0) & (wg < factor * 0.995)
        if valid.mean() < 0.5:
            continue
        beff = wr - wg * (denom / factor)
        for per in (2, 3, 4, 6, 8):
            tile = np.zeros((per, per))
            ok = True
            for i in range(per):
                for j in range(per):
                    v = beff[i::per, j::per][valid[i::per, j::per]]
                    if v.size < 4:
                        ok = False
                        break
                    tile[i, j] = np.median(v)
                if not ok:
                    break
            if not ok:
                continue
            tiled = np.tile(tile, (96 // per, 96 // per))
            if np.abs(tiled - beff)[valid].max() < 0.75:
                tile -= 0.5 * denom / 65535.0          # LibRaw's 16-bit stage truncates: remove the half-LSB bias
                if np.abs(tile - np.round(tile)).max() < 0.25:
                    tile = np.round(tile)             # black levels are integer counts
                return per, np.round(tile, 3)
    return None


# ----------------------------------------------------------------------------- metadata
def collect_metadata(raw, path: Path, exif: dict, args) -> dict:
    """Everything we know about the raw, as plain Python types (JSON-able)."""
    colors_vis = raw.raw_colors_visible
    period, cfa = detect_cfa(colors_vis)
    black = [float(b) for b in raw.black_level_per_channel]
    white_libraw = int(raw.white_level)
    cam_white = raw.camera_white_level_per_channel
    cam_white = [int(x) for x in cam_white] if cam_white is not None else None

    if args.white_level == "libraw":
        white = white_libraw
    elif args.white_level == "camera":
        if not cam_white:
            raise ValueError("--white-level camera: file has no camera-reported white level")
        white = max(cam_white)
    else:
        white = int(args.white_level)

    black_ref = min(black)          # LibRaw's common black; identical gain for all channels
    denom = white - black_ref
    factor, scale_desc = scale_factor(args.scale, exif)

    # --- colour matrices --------------------------------------------------------------
    cam_xyz = np.array(raw.rgb_xyz_matrix, dtype=np.float64)[:3, :3]   # XYZ → camera
    matrix_source = "LibRaw/Adobe table"
    illum_code = 21
    illum_name = "D65 (assumed: LibRaw's built-in Adobe matrices are the D65 ones)"
    dng = {}
    for i in (1, 2):
        cm = exif.get(f"ColorMatrix{i}")
        if cm is not None and len(cm) >= 9:
            ic = exif.get(f"CalibrationIlluminant{i}")
            dng[i] = (np.array(cm[:9], dtype=np.float64).reshape(3, 3), int(ic) if ic is not None else None)
    if dng:
        # Prefer the D65 matrix, else the highest-numbered one (DNG convention: 2 is usually D65).
        pick = next((i for i, (_, ic) in dng.items() if ic == 21), max(dng))
        dng_m, dng_ic = dng[pick]
        if not np.any(cam_xyz) or args.prefer_dng_matrix:
            cam_xyz, matrix_source = dng_m, f"DNG ColorMatrix{pick}"
            illum_code = dng_ic
            illum_name = ILLUMINANT_NAMES.get(dng_ic, str(dng_ic)) if dng_ic is not None else "unknown"
    has_matrix = bool(np.any(cam_xyz)) and abs(np.linalg.det(cam_xyz)) > 1e-12
    if has_matrix:
        xyz_from_cam = np.linalg.inv(cam_xyz)
        rec709_from_cam = XYZ_TO_REC709 @ xyz_from_cam
        white_xyz = ILLUMINANTS.get(illum_code, (None, None))[1]
        neutral = (cam_xyz @ np.array(white_xyz)).tolist() if white_xyz else None
    else:
        xyz_from_cam = rec709_from_cam = None
        neutral = None

    wb_shot = list(raw.camera_whitebalance)
    wb_day = list(raw.daylight_whitebalance)

    def norm_g(v):
        """Multipliers normalised to G = 1, or None if unusable (zero / negative / non-finite)."""
        try:
            v = [float(x) for x in list(v)[:3]]
        except Exception:
            return None
        if len(v) != 3 or not all(math.isfinite(x) and x > 0 for x in v):
            return None
        return [x / v[1] for x in v]

    as_shot = norm_g(wb_shot)
    if as_shot is None and exif.get("AsShotNeutral"):            # DNG fallback when LibRaw's cam_mul is unusable
        n = exif["AsShotNeutral"]
        if len(n) >= 3 and all(isinstance(x, (int, float)) and x > 0 for x in n[:3]):
            as_shot = norm_g([1.0 / x for x in n[:3]])
    daylight = norm_g(wb_day)
    calib = norm_g([1.0 / x for x in neutral]) if neutral and all(x > 0 for x in neutral) else None

    # --- white balance gains (only applied when --wb != none) ---------------------------
    wb_mode = args.wb.strip().lower()
    if wb_mode == "none":
        wb_gains = None
    elif wb_mode in ("as-shot", "asshot", "camera"):
        wb_gains = as_shot
        if wb_gains is None:
            raise ValueError(f"--wb as-shot: no usable as-shot multipliers (LibRaw cam_mul={wb_shot}, "
                             f"DNG AsShotNeutral={exif.get('AsShotNeutral')}); try --wb daylight/calibration/custom")
    elif wb_mode == "daylight":
        wb_gains = daylight
        if wb_gains is None:
            raise ValueError(f"--wb daylight: no usable daylight multipliers (LibRaw pre_mul={wb_day})")
    elif wb_mode == "calibration":
        wb_gains = calib
        if wb_gains is None:
            raise ValueError("--wb calibration: needs a colour matrix with a known calibration illuminant")
    elif wb_mode.startswith("custom:"):
        wb_gains = norm_g([float(x) for x in wb_mode[7:].split(",")])
        if wb_gains is None:
            raise ValueError("--wb custom:R,G,B needs three positive numbers")
    else:
        raise ValueError(f"unknown --wb mode {args.wb!r}")
    if wb_gains is not None and has_matrix:
        inv_g = np.diag([1.0 / g for g in wb_gains])          # balanced → native, then native → XYZ
        xyz_from_cam_wb = xyz_from_cam @ inv_g
        rec709_from_cam_wb = XYZ_TO_REC709 @ xyz_from_cam_wb
    else:
        xyz_from_cam_wb = rec709_from_cam_wb = None

    meta = {
        "raw2exr": {"version": __version__, "source_file": path.name,
                    "libraw_version": ".".join(map(str, __import__("rawpy").libraw_version))},
        "camera": {"make": exif.get("Make"), "model": exif.get("Model") or exif.get("UniqueCameraModel"),
                   "lens": exif.get("LensModel")},
        "exposure": {"iso": exif.get("ISO"), "shutter_s": exif.get("ExposureTime"),
                     "fnumber": exif.get("FNumber"), "compensation_ev": exif.get("ExposureCompensation"),
                     "datetime_original": exif.get("DateTimeOriginal"),
                     "dng_baseline_exposure_ev": exif.get("BaselineExposure"),
                     "scale_mode": args.scale, "scale_factor": factor, "scale_description": scale_desc,
                     "sensor_saturation_maps_to": factor},
        "raw": {"black_level_per_channel_RGBG2": black, "black_reference": black_ref,
                "white_level": white, "white_level_libraw": white_libraw,
                "white_level_camera_per_channel": cam_white, "denominator": denom,
                "cfa_period": period, "cfa_pattern": cfa, "color_desc": raw.color_desc.decode(),
                "num_colors": int(raw.num_colors), "raw_type": str(raw.raw_type).split(".")[-1],
                "visible_size": [int(raw.sizes.height), int(raw.sizes.width)],
                "pixel_aspect": float(raw.sizes.pixel_aspect),
                "formula": "value_c = clip((raw_c - black_c) / (white - min(black)), 0, 1) * scale_factor"},
        "color": {"space": ("camera-native linear RGB: no white balance, no matrix, no gamma" if wb_gains is None else
                            f"camera linear RGB, white-balanced ({wb_mode}); no matrix, no gamma"),
                  "xyz_to_camera": cam_xyz.tolist() if has_matrix else None,
                  "xyz_to_camera_source": matrix_source if has_matrix else None,
                  "calibration_illuminant_code": illum_code if has_matrix else None,
                  "calibration_illuminant": illum_name if has_matrix else None,
                  "camera_to_xyz": xyz_from_cam.tolist() if has_matrix else None,
                  "camera_to_rec709_linear": rec709_from_cam.tolist() if has_matrix else None,
                  "camera_rgb_of_calibration_white": neutral,
                  "matrices_apply_to": "camera-native (unbalanced) RGB",
                  "wb_applied_multipliers_RGB": wb_gains,
                  "camera_wb_to_xyz": xyz_from_cam_wb.tolist() if xyz_from_cam_wb is not None else None,
                  "camera_wb_to_rec709_linear": rec709_from_cam_wb.tolist() if rec709_from_cam_wb is not None else None,
                  "dng_color_matrix_1": dng[1][0].tolist() if 1 in dng else None,
                  "dng_calibration_illuminant_1": dng[1][1] if 1 in dng else None,
                  "dng_color_matrix_2": dng[2][0].tolist() if 2 in dng else None,
                  "dng_calibration_illuminant_2": dng[2][1] if 2 in dng else None,
                  "wb_as_shot_multipliers_RGB": as_shot,
                  "wb_daylight_multipliers_RGB": daylight,
                  "dng_as_shot_neutral": exif.get("AsShotNeutral"),
                  "notes": ("xyz_to_camera is the matrix shipped with the raw (unnormalised, "
                            "relative to the calibration illuminant, no chromatic adaptation). "
                            "camera_to_xyz = inv(xyz_to_camera); its absolute scale is arbitrary. "
                            "To white-balance to the calibration illuminant divide each channel by "
                            "camera_rgb_of_calibration_white before applying camera_to_xyz. If wb_applied_multipliers_RGB "
                            "is set the pixels are already balanced: use camera_wb_to_xyz / camera_wb_to_rec709_linear, "
                            "or divide by the multipliers to get back to native.")},
        "processing": {"mode": "binned" if args.binned else f"demosaic:{args.demosaic.upper()}",
                       "orientation_flip_code": int(raw.sizes.flip),
                       "orientation_applied": not args.no_orient,
                       "sub_floor_negatives": "kept" if (args.binned and args.keep_negatives) else "clipped_to_0",
                       "values_above_white": ("clipped_to_1.0" if wb_gains is None else
                                              "clipped_to_1.0 before WB (balanced R/B of clipped sensels exceed 1.0)"),
                       "white_balance": "none" if wb_gains is None else wb_mode,
                       "color_matrix": "none", "gamma": "none (linear)",
                       "pixel_type": "half" if args.half else "float32",
                       "exif_source": exif.get("_source")},
    }
    return meta


def build_header(meta: dict, args) -> dict:
    import OpenEXR
    comp = getattr(OpenEXR, f"{args.compression.upper()}_COMPRESSION")
    h = {"compression": comp, "type": OpenEXR.scanlineimage,
         "comments": ("raw2exr: scene-linear CAMERA-NATIVE RGB. No white balance, no colour matrix, "
                      "no gamma. See raw2exr.* / color.* attributes."),
         "raw2exr.version": __version__,
         "raw2exr.metadata_json": json.dumps(meta, default=float),
         "raw2exr.formula": meta["raw"]["formula"],
         "colorSpace": "camera-native-linear",
         }
    if meta["raw"]["pixel_aspect"] not in (0.0, 1.0):
        h["pixelAspectRatio"] = float(meta["raw"]["pixel_aspect"])

    def put(key, v):
        if v is None:
            return
        if isinstance(v, dict):
            for kk, vv in v.items():
                put(f"{key}.{kk}", vv)
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
            h[key] = np.array(v, dtype=np.float32)            # M33f
        elif isinstance(v, (list, tuple)):
            h[key] = [float(x) for x in v]                    # float vector
        elif isinstance(v, bool):
            h[key] = int(v)
        elif isinstance(v, (int, float, str)):
            h[key] = v
        else:
            h[key] = str(v)

    for sect in ("camera", "exposure", "raw", "color", "processing"):
        for k, v in meta[sect].items():
            if k in ("notes", "formula"):
                continue
            put(f"{sect}.{k}", v)
    return h


# ----------------------------------------------------------------------------- conversion
def convert_one(src: str, dst: str, args) -> dict:
    import rawpy
    import OpenEXR

    t0 = time.time()
    src_p, dst_p = Path(src), Path(dst)
    exif = read_exif(src_p)
    with rawpy.imread(str(src_p)) as raw:
        meta = collect_metadata(raw, src_p, exif, args)
        r = meta["raw"]
        factor = meta["exposure"]["scale_factor"]
        white, denom = r["white_level"], r["denominator"]
        black_map = np.array(r["black_level_per_channel_RGBG2"], np.float32)[raw.raw_colors_visible]
        colors = raw.raw_colors_visible.copy()
        colors[colors == 3] = 1

        if args.binned:
            vals = (raw.raw_image_visible.astype(np.float32) - black_map) / np.float32(denom)   # counts/blacks exact in float32
            np.minimum(vals, 1.0, out=vals)
            if not args.keep_negatives:
                np.maximum(vals, 0.0, out=vals)
            if r["cfa_period"] is None:
                raise RuntimeError("--binned needs a 2×2 (Bayer) or 6×6 (X-Trans) CFA")
            rgb, b = block_bin(vals, colors, r["cfa_period"])
            rgb = (rgb * factor).astype(np.float32)
            meta["processing"]["bin_size"] = b
            verify = None
        else:
            pp = dict(
                demosaic_algorithm=demosaic_algorithm(args.demosaic),
                user_wb=[1.0, 1.0, 1.0, 1.0],          # no white balance (LibRaw would otherwise apply daylight multipliers)
                use_camera_wb=False, use_auto_wb=False,
                output_color=rawpy.ColorSpace.raw,     # no colour matrix
                gamma=(1.0, 1.0),                      # linear
                no_auto_bright=True, bright=1.0,       # no auto exposure
                highlight_mode=rawpy.HighlightMode.Clip,
                adjust_maximum_thr=0.0,                # never rescale to the image's own max
                output_bps=16,
                user_flip=0,                           # we handle orientation ourselves
                fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,
                median_filter_passes=0,
            )
            if args.white_level != "libraw":
                pp["user_sat"] = white
            out16 = raw.postprocess(**pp)              # = clip((raw − black_c) · 65535 / (white − min(black)))
            rgb = out16.astype(np.float32) * np.float32(factor / 65535.0)
            verify = None
            if not args.no_verify:
                got = np.take_along_axis(rgb, colors[..., None], axis=2)[..., 0]
                exp = raw.raw_image_visible.astype(np.float32)
                exp -= black_map
                exp /= np.float32(denom)
                np.clip(exp, 0, 1, out=exp)
                exp *= np.float32(factor)
                dev = np.abs(exp - got)
                q = factor / 65535.0
                bad = dev > 4 * q
                i = int(np.argmax(dev))
                rr, cc = divmod(i, dev.shape[1])
                verify = {"max_abs_dev": float(dev.max()),
                          "rms_dev": float(np.sqrt(np.mean(dev.astype(np.float64) ** 2))),
                          "quantisation_step": q,
                          "sites_over_4q": int(bad.sum()),
                          "frac_sites_over_4q": float(bad.mean()),
                          "per_channel_max_dev_RGB": [float(dev[colors == c].max()) if (colors == c).any() else 0.0
                                                      for c in range(3)],
                          "worst": {"row": rr, "col": cc, "channel": "RGB"[int(colors[rr, cc])],
                                    "raw": int(raw.raw_image_visible[rr, cc]), "black": float(black_map[rr, cc]),
                                    "expected": float(exp[rr, cc]), "got": float(got[rr, cc])}}
                if bad.any():   # where do the deviating sites live? (helps diagnose per-camera quirks)
                    br, bc = np.nonzero(bad)
                    verify["bad_bbox"] = [int(br.min()), int(bc.min()), int(br.max()), int(bc.max())]
                    verify["bad_raw_range"] = [int(raw.raw_image_visible[bad].min()), int(raw.raw_image_visible[bad].max())]
                    patt = recover_black_pattern(raw.raw_image_visible, got, denom, factor)
                    if patt is not None:      # does a per-pixel black pattern explain it exactly?
                        per, tile = patt
                        H, W = got.shape
                        exp = raw.raw_image_visible.astype(np.float32)
                        exp -= np.tile(tile.astype(np.float32), (H // per + 1, W // per + 1))[:H, :W]
                        exp /= np.float32(denom)
                        np.clip(exp, 0, 1, out=exp)
                        exp *= np.float32(factor)
                        dev2 = float(np.abs(exp - got).max())
                        if dev2 <= 1.5 * q + 1e-9:
                            verify["explained_by"] = f"per-pixel black pattern {per}x{per} (LibRaw cblack pattern)"
                            verify["max_abs_dev_with_pattern"] = dev2
                            meta["raw"]["black_level_pattern_size"] = per
                            meta["raw"]["black_level_pattern_counts"] = tile.tolist()
                            meta["raw"]["formula"] += "  [black_c is the per-pixel black_level_pattern_counts for this file]"
                del exp, got, dev, bad
                meta["processing"]["verify_cfa_sites"] = verify

        wb_gains = meta["color"]["wb_applied_multipliers_RGB"]
        if wb_gains is not None:                                # after verify (which checks native values), before flip
            rgb *= np.array(wb_gains, dtype=np.float32)

        flip = meta["processing"]["orientation_flip_code"]   # read before postprocess: LibRaw resets sizes.flip after user_flip=0
        if not args.no_orient and flip:
            rgb = apply_flip(rgb, flip)

    if args.half:
        rgb = rgb.astype(np.float16)
    header = build_header(meta, args)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    with OpenEXR.File(header, {"RGB": np.ascontiguousarray(rgb)}) as f:
        f.write(str(dst_p))
    if args.sidecar:
        dst_p.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=float))
    return {"src": src, "dst": dst, "shape": list(rgb.shape), "seconds": time.time() - t0,
            "verify": verify, "max": float(rgb.max()), "min": float(rgb.min()),
            "scale_factor": factor, "camera": f'{meta["camera"]["make"] or ""} {meta["camera"]["model"] or ""}'.strip()}


def _worker(job):
    src, dst, args = job
    try:
        return convert_one(src, dst, args)
    except Exception as e:  # keep the batch going
        return {"src": src, "dst": dst, "error": f"{type(e).__name__}: {e}"}


def collect_inputs(paths, outdir: Path, suffix: str):
    jobs = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in RAW_EXTS:
                    rel = f.relative_to(p)
                    jobs.append((f, outdir / rel.with_name(rel.stem + suffix + ".exr")))
        elif p.is_file():
            jobs.append((p, outdir / (p.stem + suffix + ".exr")))
        else:
            print(f"warning: {p} not found", file=sys.stderr)
    return jobs


def print_info(path: Path, args):
    import rawpy
    exif = read_exif(path)
    with rawpy.imread(str(path)) as raw:
        meta = collect_metadata(raw, path, exif, args)
    print(f"== {path}")
    print(json.dumps(meta, indent=2, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="raw2exr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="raw files and/or directories (directories are scanned recursively)")
    ap.add_argument("-o", "--outdir", default="exr", help="output directory (default: ./exr)")
    ap.add_argument("--demosaic", default="AHD", choices=["AHD", "DHT", "DCB", "AAHD", "VNG", "PPG", "LINEAR"],
                    help="LibRaw demosaic (default AHD; all preserve the original samples). X-Trans always uses LibRaw's X-Trans interpolator.")
    ap.add_argument("--binned", action="store_true", help="no demosaic: bin 2×2 (Bayer) / 3×3 (X-Trans) → half/third resolution, G = mean of greens")
    ap.add_argument("--keep-negatives", action="store_true", help="(--binned only) keep sub-floor noise instead of clipping to 0")
    ap.add_argument("--scale", default="sensor",
                    help="global exposure scale: sensor (default, 1.0 = saturation) | iso12232 (×√2: diffuse white→1.0, grey→0.18 "
                         "for saturation-based ISO) | dng (×2^BaselineExposure) | ev:X | mul:X | grey:V (V = sensor-scale value of an 18%% card)")
    ap.add_argument("--wb", default="none",
                    help="white balance to bake in (default none = camera-native): as-shot | daylight | calibration "
                         "(camera's response to the matrix's calibration white) | custom:R,G,B. Gains are normalised to G=1, "
                         "recorded in the header, and the camera_to_xyz / rec709 matrices are re-expressed for the balanced data.")
    ap.add_argument("--white-level", default="libraw", help="libraw (default) | camera (use camera-reported white level) | integer")
    ap.add_argument("--prefer-dng-matrix", action="store_true", help="use DNG ColorMatrix tags over LibRaw's table when both exist")
    ap.add_argument("--half", action="store_true", help="write 16-bit half floats (default: float32, lossless)")
    ap.add_argument("--compression", default="zip", choices=["none", "rle", "zips", "zip", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab"],
                    help="EXR compression (default zip; zip/zips/piz/rle are lossless)")
    ap.add_argument("--no-orient", action="store_true", help="keep sensor orientation (do not apply the EXIF rotation)")
    ap.add_argument("--no-verify", action="store_true", help="skip the per-file check of output vs. formula at CFA sites")
    ap.add_argument("--sidecar", action="store_true", help="also write <name>.json with all metadata")
    ap.add_argument("--suffix", default="", help="filename suffix before .exr")
    ap.add_argument("-j", "--jobs", type=int, default=min(4, os.cpu_count() or 1), help="parallel workers")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--info", action="store_true", help="print metadata for the inputs and exit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.keep_negatives and not args.binned:
        ap.error("--keep-negatives is only available with --binned (LibRaw clips sub-floor values before demosaic)")
    scale_factor(args.scale, {"BaselineExposure": 0})   # validate syntax early

    outdir = Path(args.outdir)
    jobs = collect_inputs(args.inputs, outdir, args.suffix)
    if not jobs:
        print("no raw files found", file=sys.stderr)
        return 1
    if args.info:
        for src, _ in jobs:
            print_info(src, args)
        return 0
    if not args.overwrite:
        skipped = [j for j in jobs if j[1].exists()]
        jobs = [j for j in jobs if not j[1].exists()]
        if skipped:
            print(f"skipping {len(skipped)} existing output(s) (use --overwrite)")
    print(f"{len(jobs)} file(s) → {outdir}/  [{'binned' if args.binned else args.demosaic}, scale={args.scale}, wb={args.wb}, "
          f"{'half' if args.half else 'float32'}, {args.compression}, {args.jobs} worker(s)]")
    if args.dry_run:
        for src, dst in jobs:
            print(f"  {src}  →  {dst}")
        return 0

    failures = 0
    t0 = time.time()
    work = [(str(s), str(d), args) for s, d in jobs]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")           # LibRaw uses OpenMP; fork() can deadlock
    with ProcessPoolExecutor(max_workers=max(1, args.jobs), mp_context=ctx) as ex:
        for fut in as_completed([ex.submit(_worker, w) for w in work]):
            res = fut.result()
            if "error" in res:
                failures += 1
                hint = ""
                if "Unsupported file format" in res["error"]:
                    hint = "  (LibRaw cannot decode this file: e.g. Nikon HE/HE* NEF, ARRI, very new models — convert to DNG first)"
                print(f"  FAIL {res['src']}: {res['error']}{hint}")
                continue
            v = res["verify"]
            vs = ""
            if v:
                ok = v["max_abs_dev"] <= 1.5 * v["quantisation_step"] + 1e-9
                if ok:
                    vs = f"  verify max|Δ|={v['max_abs_dev']:.2e} OK"
                elif v.get("explained_by"):
                    vs = (f"  verify max|Δ|={v['max_abs_dev_with_pattern']:.2e} OK "
                          f"({v['explained_by']}; recorded in raw.black_level_pattern_counts)")
                else:
                    w = v["worst"]
                    vs = (f"  verify DEVIATES max|Δ|={v['max_abs_dev']:.2e} rms={v['rms_dev']:.1e} "
                          f"{100 * v['frac_sites_over_4q']:.3f}% of sites; worst {w['channel']}@({w['row']},{w['col']}) "
                          f"raw={w['raw']} black={w['black']:g} expected={w['expected']:.5f} got={w['got']:.5f}; "
                          f"bad rows {v['bad_bbox'][0]}-{v['bad_bbox'][2]} cols {v['bad_bbox'][1]}-{v['bad_bbox'][3]} "
                          f"raw∈[{v['bad_raw_range'][0]},{v['bad_raw_range'][1]}]")
            print(f"  {Path(res['src']).name} → {Path(res['dst']).name}  {res['shape'][1]}×{res['shape'][0]}  "
                  f"{res['camera']}  range [{res['min']:.4g}, {res['max']:.4g}]  {res['seconds']:.1f}s{vs}")
    print(f"done: {len(jobs) - failures} ok, {failures} failed, {time.time() - t0:.1f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
