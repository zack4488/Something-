#!/usr/bin/env python3
"""
Blind QR Watermark Tool (盲水印QR)

Embedding technique: 2-level Haar DWT — QR and PN signature are added
to the LL2 (lowest-frequency) subband so the mark survives JPEG
compression, screen capture, and moderate camera capture.

Detection uses spread-spectrum PN correlation (no pixel-exact marker),
which is robust to any processing that preserves low-frequency structure.

Watermark removal is intentionally not provided.
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

# ── tunable constants ─────────────────────────────────────────────────────────
WAVELET       = "haar"   # Haar for exact-size IDWT; robust LL subband
DWT_LEVELS    = 2        # two levels → LL2 at 1/4 image size
QR_ALPHA      = 48.0     # QR embedding strength in LL2
SIG_ALPHA     = 32.0     # PN-signature strength in LL2
SIG_SEED      = 0x6D616E # fixed seed ("man" in hex, for 盲 watermark)
DETECT_THRESH = 10.0     # correlation threshold; ~0 for clean, ~SIG_ALPHA for watermarked

# ── DWT helpers ───────────────────────────────────────────────────────────────

def _dwt2_levels(gray: np.ndarray):
    """Return (LL2, subbands_L2, subbands_L1) for a grayscale float array."""
    ll1, (lh1, hl1, hh1) = pywt.dwt2(gray, WAVELET)
    ll2, (lh2, hl2, hh2) = pywt.dwt2(ll1, WAVELET)
    return ll2, (lh2, hl2, hh2), (lh1, hl1, hh1), ll1.shape


def _idwt2_levels(ll2, subbands_l2, subbands_l1, ll1_shape):
    """Reconstruct a grayscale float array from modified LL2."""
    ll1_wm = pywt.idwt2((ll2, subbands_l2), WAVELET)
    # Crop back to the expected LL1 shape (Haar is exact, but guard anyway)
    h, w = ll1_shape
    ll1_wm = ll1_wm[:h, :w]
    gray_wm = pywt.idwt2((ll1_wm, subbands_l1), WAVELET)
    return gray_wm


# ── PN signature ──────────────────────────────────────────────────────────────

def _pn_sequence(shape: tuple) -> np.ndarray:
    """Fixed pseudo-random ±1 sequence tied to the secret seed."""
    rng = np.random.default_rng(SIG_SEED)
    return rng.choice([-1.0, 1.0], size=shape).astype(np.float64)


# ── QR helpers ────────────────────────────────────────────────────────────────

def _make_qr_array(text: str, h: int, w: int) -> np.ndarray:
    """
    Generate a QR code (float64, 0=white 1=black) sized to (h, w).
    Uses H error-correction (30 % recoverable) and minimum version to
    keep cells as large as possible (better blur robustness).
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
    arr = (arr < 128).astype(np.float64)   # 1=black, 0=white
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)


# ── camera / screenshot pre-processing ───────────────────────────────────────

def _preprocess_for_detection(gray: np.ndarray) -> np.ndarray:
    """
    Normalise an image that may have been captured by a camera or
    screenshot tool.  CLAHE corrects uneven lighting; mild denoise
    reduces sensor noise; mild sharpening partially reverses lens blur.
    """
    u8 = np.clip(gray, 0, 255).astype(np.uint8)
    # CLAHE — restores contrast lost to display / exposure variation
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    u8 = clahe.apply(u8)
    # Bilateral denoise — preserves edges better than Gaussian
    u8 = cv2.bilateralFilter(u8, d=5, sigmaColor=20, sigmaSpace=20)
    # Unsharp mask — partially reverses camera blur
    blurred = cv2.GaussianBlur(u8, (0, 0), sigmaX=1.2)
    u8 = cv2.addWeighted(u8, 1.6, blurred, -0.6, 0)
    return u8.astype(np.float64)


# ── public API ────────────────────────────────────────────────────────────────

def has_watermark(image_path: str, camera_mode: bool = False) -> bool:
    """
    Return True if the image contains a blind QR watermark.

    camera_mode=True applies extra pre-processing for images captured
    by a camera or screenshot tool before correlation.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    if camera_mode:
        gray = _preprocess_for_detection(gray)
    ll2, _, _, _ = _dwt2_levels(gray)
    pn = _pn_sequence(ll2.shape)
    corr = float(np.dot(ll2.ravel(), pn.ravel())) / ll2.size
    return corr > DETECT_THRESH


def embed(image_path: str, output_path: str, text: str) -> None:
    """
    Embed a QR code as a blind watermark and save to output_path.

    The QR is embedded in the LL2 (lowest-frequency) DWT subband at
    QR_ALPHA strength.  A PN detection signature at SIG_ALPHA is added
    alongside it.  Both survive JPEG compression and camera capture.
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
    ll2, subs_l2, subs_l1, ll1_shape = _dwt2_levels(gray)

    # Build QR and PN at LL2 size
    qr_arr  = _make_qr_array(text, ll2.shape[0], ll2.shape[1])
    pn      = _pn_sequence(ll2.shape)

    # Centre QR around 0 so black cells push UP and white cells push DOWN
    qr_c = qr_arr * 2.0 - 1.0   # maps 0→-1, 1→+1

    ll2_wm = ll2 + QR_ALPHA * qr_c + SIG_ALPHA * pn

    gray_wm = _idwt2_levels(ll2_wm, subs_l2, subs_l1, ll1_shape)
    gray_wm = np.clip(gray_wm, 0, 255)

    # Apply the luminance delta to all colour channels proportionally
    img_out = img.copy()
    diff = (gray_wm - gray).astype(np.float64)
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(np.float64) + diff
        img_out[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, img_out)
    psnr = 10 * np.log10(255 ** 2 / max(np.mean(diff ** 2), 1e-9))
    print(f"Watermark embedded → {output_path}  (PSNR {psnr:.1f} dB)")


def _extract_qr_arr(gray: np.ndarray,
                    camera_mode: bool = False) -> np.ndarray | None:
    """
    Return the extracted QR pattern as a binary uint8 ndarray, or None if
    no watermark is detected.  Used internally by extract() and the GUI.
    """
    if camera_mode:
        gray = _preprocess_for_detection(gray)

    ll2, _, _, _ = _dwt2_levels(gray)
    pn = _pn_sequence(ll2.shape)
    corr = float(np.dot(ll2.ravel(), pn.ravel())) / ll2.size
    if corr <= DETECT_THRESH:
        return None

    # Subtract the PN signature to isolate QR signal + image background
    ll2_clean = ll2 - SIG_ALPHA * pn

    bg_size  = max(ll2_clean.shape[0] // 6, 8)
    residual = ll2_clean - uniform_filter(ll2_clean, size=bg_size)
    residual = np.clip(residual, 0, None)

    qr_vis = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, qr_bin = cv2.threshold(qr_vis, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Scale to a scannable size (≥ 300 px on the short side)
    scale = max(300 // min(qr_bin.shape), 1)
    return cv2.resize(qr_bin,
                      (qr_bin.shape[1] * scale, qr_bin.shape[0] * scale),
                      interpolation=cv2.INTER_NEAREST)


def extract(image_path: str,
            output_qr_path: str | None = None,
            camera_mode: bool = False) -> None:
    """
    Extract the blind QR watermark and optionally save the QR image.

    camera_mode=True pre-processes the image to handle blur / exposure
    changes from camera capture before extraction.
    """
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
    """Try pyzbar on both polarities; print decoded content or guidance."""
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
        description="Blind QR Watermark Tool (盲水印QR) — DWT-based, "
                    "survives screenshots and camera capture"
    )
    ap.add_argument("--camera", action="store_true",
                    help="Apply camera/screenshot pre-processing before "
                         "check or extract (CLAHE + denoise + sharpen)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    emb = sub.add_parser("embed", help="Add a blind QR watermark")
    emb.add_argument("input")
    emb.add_argument("output")
    emb.add_argument("--text", "-t", required=True,
                     help="Text or URL to encode as QR")

    chk = sub.add_parser("check",
                         help="Check whether an image already has a watermark")
    chk.add_argument("input")

    ext = sub.add_parser("extract", help="Extract the QR watermark")
    ext.add_argument("input")
    ext.add_argument("--output", "-o",
                     help="Save extracted QR image to this path")

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
