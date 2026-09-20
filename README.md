# raw2exr

Batch-convert camera raws to scene-linear, **camera-native** RGB OpenEXR. Floor removed, nothing else.

```
pip install rawpy OpenEXR numpy          # required
pip install exifread tifffile            # optional: EXIF fields + DNG tags (ColorMatrix, BaselineExposure)
# exiftool on PATH is used first when present (best coverage: CR3, RAF, …)

python raw2exr.py /path/to/raws -o exr                  # whole tree, mirrors sub-dirs
python raw2exr.py a.NEF b.CR2 -o exr --half --sidecar   # half float + .json sidecars
python raw2exr.py shoot/ -o exr --binned                # no demosaic at all (2×2 / 3×3 bin)
python raw2exr.py shoot/ -o exr --scale iso12232        # diffuse white → 1.0, 18% grey → 0.18
python raw2exr.py shoot/ -o exr --wb as-shot            # bake in the camera's white balance
python raw2exr.py x.dng --info                          # print everything it knows, no output
```

## The pixel formula

    value_c = clip((raw_c − black_c) / (white − min(black)), 0, 1) · scale

* `black_c` is the per-channel black level, subtracted per CFA site. **Nothing else** is subtracted or applied:
  no white balance, no matrix, no gamma, no auto-brightening, no highlight recovery, no noise reduction.
* One common denominator for all channels, so every channel gets the *same* gain (per-channel normalisation
  would be a tiny hidden white balance). `1.0 × scale` = sensor saturation. This is also exactly what LibRaw
  does internally, which is why the demosaic path can be verified against it bit-for-bit.
* Demosaic (AHD by default) only fills in the two missing channels at each site — the original sample is kept
  exactly. Every file is checked at all CFA sites against the formula (`verify max|Δ|` in the log; the only
  deviation is the 1/65535 quantisation of LibRaw's 16-bit stage). `--binned` avoids interpolation entirely.
* Sub-floor noise is clipped to 0 (LibRaw does this before demosaic); `--binned --keep-negatives` preserves it.
* Values are clipped to 1.0 (× scale) after black subtraction, as LibRaw does. Orientation is applied from EXIF (`--no-orient` to keep sensor layout).
* **Camera-native data is green.** That is the correct unbalanced signal (green sensels collect the most light). To bake in a
  white balance use `--wb as-shot | daylight | calibration | custom:R,G,B`; gains are normalised to G = 1, recorded as
  `color.wb_applied_multipliers_RGB`, and `color.camera_wb_to_xyz` / `color.camera_wb_to_rec709_linear` are the matrices
  re-expressed for the balanced data (`camera_to_xyz` always refers to native data). Balanced R/B of clipped sensels exceed 1.0.

## Exposure scale (`--scale`)

| mode | factor | meaning |
|---|---|---|
| `sensor` (default) | 1 | 1.0 = sensor saturation; no assumptions |
| `iso12232` | √2 | ISO 12232 saturation-based speed puts clipping at 141% of diffuse white → white 1.0, grey 0.18 at nominal exposure. Real cameras are usually rated SOS/REI with more headroom, so this is approximate |
| `dng` | 2^BaselineExposure | Adobe's per-camera calibration, DNG only |
| `grey:V` | 0.18/V | exact per-camera route: shoot an 18% card at nominal exposure, read its sensor-scale value V once, reuse |
| `ev:X`, `mul:X` | 2^X, X | explicit |

Exposure compensation dialled on the camera is *not* undone — the data reflects the light that actually hit the sensor.

## Metadata in the EXR header (typed attributes + `raw2exr.metadata_json` blob)

* `color.xyz_to_camera` — the XYZ→camera matrix shipped with the raw (LibRaw/Adobe table, or DNG `ColorMatrix1/2`;
  `--prefer-dng-matrix`), unnormalised, relative to `color.calibration_illuminant` (usually D65).
* `color.camera_to_xyz` = its inverse, `color.camera_to_rec709_linear` = XYZ→Rec.709(D65) · camera_to_xyz. Absolute
  scale is arbitrary; no chromatic adaptation is applied.
* `color.camera_rgb_of_calibration_white` — the camera's response to the calibration white. Dividing the channels by
  it is exactly "white balance to the calibration illuminant"; it reproduces LibRaw's daylight multipliers.
* `color.wb_as_shot_multipliers_RGB`, `color.wb_daylight_multipliers_RGB`, DNG `AsShotNeutral` when present.
* `raw.*` — black levels (R,G,B,G2), white level (LibRaw and camera-reported), denominator, CFA pattern, sizes.
* `exposure.*` — ISO, shutter, f-number, compensation, BaselineExposure, and the scale factor applied.
* `processing.*` — mode, orientation, clipping policy, verify result.
* No `chromaticities` attribute is written on purpose: the data is not in a colorimetric RGB space.

## Notes / limits

* Float32 by default (lossless for ≤16-bit sensors). `--half` halves the size; its ~1/2048 relative precision near 1.0
  is coarser than a 14-bit sensor's step but far below photon shot noise there.
* Per-pixel black *patterns* (LibRaw `cblack[4..]`, e.g. some Panasonic/Sony/DNG files) are honoured by LibRaw in demosaic
  mode. rawpy only exposes per-channel blacks, so the verify step recovers the pattern from LibRaw's own output, checks that it
  explains the deviation to quantisation level, and records it as `raw.black_level_pattern_counts`. Anything else that deviates
  is reported with where/how much (worst site, affected region, raw range) so the cause can be pinned down.
* Some vendor calibrations are applied by LibRaw before demosaic and are not "just the floor" (Phase One IIQ defect/flat-field
  correction is the known one); these show up as DEVIATES with a smooth, spatially varying error.
* EXIF (make/model/ISO/shutter/aperture) is read with exiftool when installed (recommended; single .exe on Windows), else
  with exifread — including CR3, RAF and MRW containers, which exifread cannot open on its own.
* LibRaw cannot decode Nikon HE/HE* (TicoRAW) NEFs or ARRI files; convert those to DNG first (Adobe DNG Converter).
* LibRaw's "white level" for some cameras (e.g. Nikon) differs from the camera-reported linear maximum; both are recorded,
  `--white-level camera` switches the denominator.
* Foveon / 4-colour CFAs are not supported. X-Trans is (LibRaw's interpolator, or 3×3 binning).
