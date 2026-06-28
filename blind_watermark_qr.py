#!/usr/bin/env python3
"""
Blind QR Watermark Tool (盲水印QR)

Robustness:
  • Screenshot / camera  — DWT LL2 subband survives JPEG, blur, noise.
  • Resize               — multi-scale probe (0.25×–4×) at detection.
  • Crop (any position)  — periodic embedding (same PN in every 256×256 tile)
                           + shift-invariant fold detection: fold any crop
                           into 256×256, cyclic-correlate DWT LL2 with PN.
  • Edits (colour, etc.) — CLAHE + sharpen pre-processing + PN correlation
                           which is insensitive to global brightness shifts.

Technique:
  2-level Haar DWT on each 256×256 tile (non-overlapping, same PN every tile).
  QR (centred ±QR_ALPHA) + PN signature (×SIG_ALPHA) are added to LL2.
  Detection: multi-scale scan for resize, periodic-fold + cyclic-FFT for crop.
"""

import sys
import argparse
import io
import numpy as np
import cv2
import pywt
import qrcode
from PIL import Image
from scipy.ndimage import uniform_filter

# ── constants ─────────────────────────────────────────────────────────────────
WAVELET      = "haar"
TILE_SIZE    = 256                # pixels per embedding tile
LL2_SIZE     = TILE_SIZE // 4    # = 64  (2-level DWT of TILE_SIZE)
QR_ALPHA     = 48.0              # QR embedding strength in LL2
SIG_ALPHA    = 32.0              # PN detection strength in LL2
SIG_SEED     = 0x6D616E          # fixed seed ("man" for 盲)
DETECT_THRESH = 8.0              # DWT correlation threshold (clean≈0, marked≈SIG_ALPHA)
FOLD_THRESH   = 0.5              # spatial fold threshold (clean≈0.1, marked≈2.0)

# Probe window sizes tried during detection (handles resize from ≈0.25× to 4×)
PROBE_SIZES  = [64, 96, 128, 192, 256, 320, 384, 512, 640, 768, 1024]


# ── PN sequence ───────────────────────────────────────────────────────────────

def _pn() -> np.ndarray:
    """Fixed 64×64 pseudo-random ±1 sequence."""
    rng = np.random.default_rng(SIG_SEED)
    return rng.choice([-1.0, 1.0], size=(LL2_SIZE, LL2_SIZE))


# ── per-tile DWT helpers ──────────────────────────────────────────────────────

def _tile_to_256(region: np.ndarray) -> np.ndarray:
    """Bilinear-resize any region to TILE_SIZE×TILE_SIZE float64."""
    if region.shape == (TILE_SIZE, TILE_SIZE):
        return region.astype(np.float64)
    return cv2.resize(region.astype(np.float64),
                      (TILE_SIZE, TILE_SIZE),
                      interpolation=cv2.INTER_LINEAR)


def _embed_tile(tile: np.ndarray, qr64: np.ndarray, pn: np.ndarray) -> np.ndarray:
    """
    Embed QR + PN into a TILE_SIZE×TILE_SIZE float64 grayscale tile.
    Returns the watermarked tile (same shape).
    """
    ll1, d1 = pywt.dwt2(tile, WAVELET)
    ll2, d2 = pywt.dwt2(ll1, WAVELET)

    qr_c = qr64 * 2.0 - 1.0          # centre: 0→-1 (white), 1→+1 (black)
    ll2_wm = ll2 + QR_ALPHA * qr_c + SIG_ALPHA * pn

    ll1_wm = pywt.idwt2((ll2_wm, d2), WAVELET)[:TILE_SIZE // 2, :TILE_SIZE // 2]
    tile_wm = pywt.idwt2((ll1_wm, d1), WAVELET)[:TILE_SIZE, :TILE_SIZE]
    return tile_wm


def _corr_of_region(region: np.ndarray, pn: np.ndarray) -> float:
    """
    Resize region to TILE_SIZE×TILE_SIZE, compute DWT LL2,
    return correlation with pn.
    """
    t = _tile_to_256(region)
    ll1, _ = pywt.dwt2(t, WAVELET)
    ll2, _ = pywt.dwt2(ll1, WAVELET)
    return float(np.dot(ll2.ravel(), pn.ravel())) / ll2.size


# ── multi-scale scanner ───────────────────────────────────────────────────────

def _scan(gray: np.ndarray, pn: np.ndarray) -> tuple[float, np.ndarray | None]:
    """
    Slide windows of each probe size across gray with 50% overlap.
    Also checks the full image resized to TILE_SIZE.
    Returns (best_correlation, best_256×256_tile_or_None).
    """
    h, w = gray.shape
    best_corr = -np.inf
    best_tile: np.ndarray | None = None

    # Full-image probe (handles extreme resize or tiny cropped remains)
    c = _corr_of_region(gray, pn)
    if c > best_corr:
        best_corr = c
        best_tile = _tile_to_256(gray)

    for ps in PROBE_SIZES:
        if ps > min(h, w):
            continue
        step = max(ps // 2, 16)
        ys = list(range(0, h - ps + 1, step))
        xs = list(range(0, w - ps + 1, step))
        # Always include the last position so corners are checked
        if ys and ys[-1] + ps < h:
            ys.append(h - ps)
        if xs and xs[-1] + ps < w:
            xs.append(w - ps)
        for y in ys:
            for x in xs:
                region = gray[y: y + ps, x: x + ps]
                c = _corr_of_region(region, pn)
                if c > best_corr:
                    best_corr = c
                    best_tile = _tile_to_256(region)

    return best_corr, best_tile


# ── shift-invariant fold detection ───────────────────────────────────────────

def _fold_detect(gray: np.ndarray, pn: np.ndarray) -> float:
    """
    Shift-invariant watermark detection via periodic fold + spatial cross-correlation.

    The embedded watermark is TILE_SIZE-periodic in spatial domain (identical PN
    signal in every TILE_SIZE block).  Folding any crop into TILE_SIZE×TILE_SIZE
    recovers the periodic signal regardless of crop offset.  A cyclic spatial
    cross-correlation with the expected PN spatial pattern finds the peak without
    knowing the offset (unlike DWT correlation, this is not fooled by the
    non-shift-invariance of the Haar wavelet).

    Returns peak / TILE_SIZE².  Watermarked ≈ 2.0, clean ≈ 0.1.
    Use with FOLD_THRESH.  Handles any crop ≥ TILE_SIZE/2 in each dimension.
    """
    h, w = gray.shape
    # Fold into TILE_SIZE×TILE_SIZE by summing all aligned blocks, then averaging
    ph = int(np.ceil(h / TILE_SIZE)) * TILE_SIZE
    pw = int(np.ceil(w / TILE_SIZE)) * TILE_SIZE
    padded = np.zeros((ph, pw), dtype=np.float64)
    padded[:h, :w] = gray
    count = np.zeros((ph, pw), dtype=np.float64)
    count[:h, :w] = 1.0

    nb_y = ph // TILE_SIZE
    nb_x = pw // TILE_SIZE
    folded = padded.reshape(nb_y, TILE_SIZE, nb_x, TILE_SIZE).sum(axis=(0, 2))
    cnt    = count.reshape(nb_y, TILE_SIZE, nb_x, TILE_SIZE).sum(axis=(0, 2))
    folded = np.where(cnt > 0, folded / cnt, 0.0)

    # Compute the expected spatial PN pattern: IDWT2 of pn placed in LL2 position
    zeros_ll2 = (np.zeros((LL2_SIZE, LL2_SIZE)),) * 3
    zeros_ll1 = (np.zeros((TILE_SIZE // 2, TILE_SIZE // 2)),) * 3
    pn_ll1  = pywt.idwt2((pn, zeros_ll2), WAVELET)[:TILE_SIZE // 2, :TILE_SIZE // 2]
    pn_spat = pywt.idwt2((pn_ll1, zeros_ll1), WAVELET)[:TILE_SIZE, :TILE_SIZE]

    # Max cyclic cross-correlation finds peak at the (unknown) crop phase offset
    ff   = np.fft.rfft2(folded)
    ft   = np.fft.rfft2(pn_spat)
    corr = np.real(np.fft.irfft2(ff * np.conj(ft), s=(TILE_SIZE, TILE_SIZE)))
    return float(corr.max()) / (TILE_SIZE * TILE_SIZE)


# ── QR helpers ────────────────────────────────────────────────────────────────

def _make_qr_array(text: str) -> np.ndarray:
    """
    Generate a QR code and resize to LL2_SIZE×LL2_SIZE (float64, 0=white 1=black).
    Uses error-correction level H (30 % recoverable).
    """
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(text)
    qr.make(fit=True)
    pil = qr.make_image(fill_color="black", back_color="white").convert("L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    arr = np.array(Image.open(buf).convert("L"), dtype=np.float64)
    arr = (arr < 128).astype(np.float64)
    return cv2.resize(arr, (LL2_SIZE, LL2_SIZE), interpolation=cv2.INTER_NEAREST)


# ── camera / screenshot pre-processing ───────────────────────────────────────

def _preprocess(gray: np.ndarray) -> np.ndarray:
    """CLAHE + bilateral denoise + unsharp-mask to counter camera degradation."""
    u8 = np.clip(gray, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    u8 = clahe.apply(u8)
    u8 = cv2.bilateralFilter(u8, d=5, sigmaColor=20, sigmaSpace=20)
    blurred = cv2.GaussianBlur(u8, (0, 0), sigmaX=1.2)
    u8 = cv2.addWeighted(u8, 1.6, blurred, -0.6, 0)
    return u8.astype(np.float64)


# ── public API ────────────────────────────────────────────────────────────────

def has_watermark(image_path: str, camera_mode: bool = False) -> bool:
    """
    Return True if the image contains a blind QR watermark.
    camera_mode applies extra pre-processing for camera/screenshot captures.
    Robust to resize (0.25×–4×) and arbitrary crop (any portion ≥ 50% of a
    tile survives via periodic-fold detection).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    if camera_mode:
        gray = _preprocess(gray)
    pn = _pn()
    # Primary: multi-scale DWT scan (handles resize well)
    best_corr, _ = _scan(gray, pn)
    if best_corr > DETECT_THRESH:
        return True
    # Fallback: spatial fold detection (handles arbitrary crop position and phase)
    return _fold_detect(gray, pn) > FOLD_THRESH


def embed(image_path: str, output_path: str, text: str) -> None:
    """
    Embed the same QR watermark in every 256×256 tile of the image.
    The mark is TILE_SIZE-periodic so any crop of the image (any position,
    any size ≥ TILE_SIZE/2) is detectable via the fold-based detector.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    if has_watermark(image_path):
        raise RuntimeError(
            "Image already contains a blind watermark — "
            "embedding blocked to prevent double-marking."
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape
    pn  = _pn()
    qr64 = _make_qr_array(text)

    diff = np.zeros((h, w), dtype=np.float64)

    # Non-overlapping TILE_SIZE grid — the PN signal is the same in every tile,
    # making the embedded mark periodic with period TILE_SIZE.
    n_tiles = 0
    for y in range(0, h, TILE_SIZE):
        for x in range(0, w, TILE_SIZE):
            ty2 = min(y + TILE_SIZE, h)
            tx2 = min(x + TILE_SIZE, w)
            tile_h, tile_w = ty2 - y, tx2 - x

            if tile_h < TILE_SIZE // 4 or tile_w < TILE_SIZE // 4:
                continue

            patch = gray[y:ty2, x:tx2]
            patch256 = _tile_to_256(patch)
            patch256_wm = _embed_tile(patch256, qr64, pn)
            d256 = patch256_wm - patch256

            if tile_h == TILE_SIZE and tile_w == TILE_SIZE:
                diff[y:ty2, x:tx2] = d256
            else:
                diff[y:ty2, x:tx2] = cv2.resize(
                    d256, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR
                )
            n_tiles += 1

    # Apply luminance diff to all colour channels
    img_out = img.copy()
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(np.float64) + diff
        img_out[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, img_out)
    mse  = float(np.mean(diff ** 2))
    psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-9))
    print(f"Watermark embedded → {output_path}  (PSNR {psnr:.1f} dB, tiles: {n_tiles})")


def _extract_qr_arr(gray: np.ndarray,
                    camera_mode: bool = False) -> np.ndarray | None:
    """
    Locate the tile with the highest watermark correlation, extract and
    return the embedded QR pattern as a binary uint8 ndarray.
    Returns None if no watermark is detected.
    """
    if camera_mode:
        gray = _preprocess(gray)

    pn = _pn()
    best_corr, best_tile = _scan(gray, pn)

    if best_corr <= DETECT_THRESH or best_tile is None:
        return None

    # Isolate the QR: subtract PN, subtract smooth background, threshold
    ll1, _ = pywt.dwt2(best_tile, WAVELET)
    ll2, _ = pywt.dwt2(ll1, WAVELET)

    ll2_clean = ll2 - SIG_ALPHA * pn
    bg_size   = max(LL2_SIZE // 6, 4)
    residual  = ll2_clean - uniform_filter(ll2_clean, size=bg_size)
    residual  = np.clip(residual, 0, None)

    qr_vis = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, qr_bin = cv2.threshold(qr_vis, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    scale = max(300 // min(qr_bin.shape), 1)
    return cv2.resize(qr_bin,
                      (qr_bin.shape[1] * scale, qr_bin.shape[0] * scale),
                      interpolation=cv2.INTER_NEAREST)


def extract(image_path: str,
            output_qr_path: str | None = None,
            camera_mode: bool = False) -> None:
    """Extract the blind QR watermark and optionally save the QR image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    qr_big = _extract_qr_arr(gray, camera_mode=camera_mode)
    if qr_big is None:
        print("No blind watermark detected in this image.")
        return

    if output_qr_path:
        cv2.imwrite(output_qr_path, qr_big)
        print(f"Extracted QR pattern saved → {output_qr_path}")
    else:
        print("Watermark detected. Use --output to save the extracted QR image.")

    _try_decode(qr_big)


def _try_decode(qr_img: np.ndarray) -> None:
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        from PIL import Image as _PIL
        for polarity in (qr_img, 255 - qr_img):
            decoded = pyzbar_decode(_PIL.fromarray(polarity))
            if decoded:
                for obj in decoded:
                    print(f"QR content: {obj.data.decode('utf-8', errors='replace')}")
                return
        print("QR extracted but auto-decode failed — "
              "save with --output and scan with a QR reader.")
    except ImportError:
        print("Tip: pip install pyzbar  for automatic QR decoding.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Blind QR Watermark Tool (盲水印QR) — "
                    "survives resize, crop, screenshot & camera capture"
    )
    ap.add_argument("--camera", action="store_true",
                    help="Apply camera/screenshot pre-processing (CLAHE + denoise + sharpen)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    emb = sub.add_parser("embed", help="Add a blind QR watermark")
    emb.add_argument("input")
    emb.add_argument("output")
    emb.add_argument("--text", "-t", required=True)

    chk = sub.add_parser("check", help="Check whether an image has a watermark")
    chk.add_argument("input")

    ext = sub.add_parser("extract", help="Extract the QR watermark")
    ext.add_argument("input")
    ext.add_argument("--output", "-o")

    args = ap.parse_args()

    if args.cmd == "embed":
        try:
            embed(args.input, args.output, args.text)
        except (RuntimeError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.cmd == "check":
        try:
            present = has_watermark(args.input, camera_mode=args.camera)
            label = "YES" if present else "NO"
            verb  = "contains" if present else "does not contain"
            print(f"Watermark present: {label} — this image {verb} a blind QR watermark.")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.cmd == "extract":
        try:
            extract(args.input, args.output, camera_mode=args.camera)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
