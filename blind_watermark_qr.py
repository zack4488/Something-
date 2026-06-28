#!/usr/bin/env python3
"""
Blind QR Watermark Tool (盲水印QR)
Embeds an invisible QR code into images using DFT frequency-domain technique.
Presence is flagged with an LSB marker so re-watermarking is blocked.
"""

import sys
import argparse
import struct
import numpy as np
import cv2
import qrcode
from PIL import Image
import io

# ── constants ────────────────────────────────────────────────────────────────
EMBED_ALPHA = 36.0          # DFT embedding strength
MAGIC = b"\xBF\xA5\x3C\xD9\xE7\x12\x6B\x4F"   # 8-byte presence marker
MARKER_PIXEL_STEP = 17      # stride between marker pixels (prime, avoids aliasing)

# ── LSB presence marker ───────────────────────────────────────────────────────

def _marker_positions(h: int, w: int) -> list[tuple[int, int]]:
    """64 fixed pixel positions for the 8-byte (64-bit) LSB presence marker."""
    positions = []
    idx = 0
    for i in range(64):
        row = (idx * MARKER_PIXEL_STEP) % h
        col = (idx * MARKER_PIXEL_STEP * 3) % w
        positions.append((row, col))
        idx += 1
    return positions


def _write_marker(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    channel = 0  # R channel (or single channel for grayscale)
    bits = []
    for byte in MAGIC:
        for bit_idx in range(8):
            bits.append((byte >> bit_idx) & 1)
    for i, (r, c) in enumerate(_marker_positions(h, w)):
        if len(out.shape) == 3:
            pixel = out[r, c, channel]
        else:
            pixel = out[r, c]
        pixel = (int(pixel) & 0xFE) | bits[i]
        if len(out.shape) == 3:
            out[r, c, channel] = pixel
        else:
            out[r, c] = pixel
    return out


def _read_marker(img: np.ndarray) -> bool:
    h, w = img.shape[:2]
    channel = 0
    bits = []
    for r, c in _marker_positions(h, w):
        if len(img.shape) == 3:
            bits.append(int(img[r, c, channel]) & 1)
        else:
            bits.append(int(img[r, c]) & 1)
    recovered = []
    for byte_i in range(8):
        byte = 0
        for bit_i in range(8):
            byte |= bits[byte_i * 8 + bit_i] << bit_i
        recovered.append(byte)
    return bytes(recovered) == MAGIC


# ── QR helpers ────────────────────────────────────────────────────────────────

def _make_qr_array(text: str, h: int, w: int) -> np.ndarray:
    """Generate a QR code (float64, 0.0=white 1.0=black) resized to (h,w)."""
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
    arr = (arr < 128).astype(np.float64)          # 1 = black, 0 = white
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)


# ── DFT watermark embed / extract ────────────────────────────────────────────

def _embed_qr_dft(gray: np.ndarray, qr_arr: np.ndarray) -> np.ndarray:
    """
    Add qr_arr to the DFT magnitude of gray.
    Symmetric embedding keeps IFFT approximately real.
    """
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)

    magnitude = np.abs(dft_shift)
    phase = np.angle(dft_shift)

    wm = qr_arr * EMBED_ALPHA
    magnitude_wm = magnitude + wm + wm[::-1, ::-1]   # symmetric

    dft_wm = magnitude_wm * np.exp(1j * phase)
    img_wm = np.real(np.fft.ifft2(np.fft.ifftshift(dft_wm)))
    return np.clip(img_wm, 0, 255).astype(np.uint8)


def _extract_qr_dft(gray_wm: np.ndarray) -> np.ndarray:
    """
    Recover the embedded QR from the DFT magnitude.
    Returns a normalized grayscale image of the extracted pattern.
    """
    dft = np.fft.fft2(gray_wm.astype(np.float64))
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # Remove DC region (bright centre blob) before analysis
    mag_ndc = magnitude.copy()
    dc_r, dc_c = max(1, h // 20), max(1, w // 20)
    mag_ndc[cy - dc_r : cy + dc_r, cx - dc_c : cx + dc_c] = 0

    # Approximate the background spectrum with a large median filter and subtract
    # to reveal the structured QR signature above the smooth background
    from scipy.ndimage import uniform_filter
    background = uniform_filter(mag_ndc, size=max(h // 16, 4))
    residual = mag_ndc - background

    # Keep only the positive residual (where QR bits raised the magnitude)
    residual = np.clip(residual, 0, None)

    # Normalise to [0,255] and take the upper-right quadrant (positive frequencies)
    # The symmetric embedding means both quadrants carry the QR and its mirror
    qr_raw = residual[:cy, cx:]      # top-right quadrant
    qr_raw = cv2.resize(qr_raw, (w, h), interpolation=cv2.INTER_AREA)
    qr_vis = cv2.normalize(qr_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Binarise
    _, qr_bin = cv2.threshold(qr_vis, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return qr_bin


# ── public API ────────────────────────────────────────────────────────────────

def has_watermark(image_path: str) -> bool:
    """Return True if the image already contains a blind QR watermark."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return _read_marker(img)


def embed(image_path: str, output_path: str, text: str) -> None:
    """Embed a QR code as a blind watermark and save to output_path."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    if has_watermark(image_path):
        raise RuntimeError(
            "Image already contains a blind watermark. "
            "Use a different image or extract first."
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape
    qr_arr = _make_qr_array(text, h, w)

    gray_wm = _embed_qr_dft(gray, qr_arr)

    # Apply DFT watermark change to all colour channels proportionally
    img_out = img.copy()
    diff = gray_wm.astype(np.int16) - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(np.int16) + diff
        img_out[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

    # Write LSB presence marker
    img_out = _write_marker(img_out)
    cv2.imwrite(output_path, img_out)
    print(f"Blind QR watermark embedded successfully -> {output_path}")


def extract(image_path: str, output_qr_path: str | None = None) -> None:
    """Extract and display the blind QR watermark from an image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    if not _read_marker(img):
        print("No blind watermark detected in this image.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    qr_bin = _extract_qr_dft(gray)

    if output_qr_path:
        cv2.imwrite(output_qr_path, qr_bin)
        print(f"Extracted QR pattern saved -> {output_qr_path}")
    else:
        print("Watermark detected. Use --output/-o to save the extracted QR image.")

    # Attempt QR decode
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        decoded = pyzbar_decode(Image.fromarray(qr_bin))
        if decoded:
            for obj in decoded:
                print(f"QR content: {obj.data.decode('utf-8', errors='replace')}")
        else:
            # Try inverting
            decoded = pyzbar_decode(Image.fromarray(255 - qr_bin))
            if decoded:
                for obj in decoded:
                    print(f"QR content: {obj.data.decode('utf-8', errors='replace')}")
            else:
                print("QR pattern extracted but could not be auto-decoded. "
                      "Save with --output and scan with a QR reader.")
    except ImportError:
        print("Tip: install pyzbar for automatic QR decoding: pip install pyzbar")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Blind QR Watermark Tool (盲水印QR) — invisibly embed or read QR codes in images"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    emb = sub.add_parser("embed", help="Add a blind QR watermark to an image")
    emb.add_argument("input", help="Input image path")
    emb.add_argument("output", help="Output (watermarked) image path")
    emb.add_argument("--text", "-t", required=True, help="Text or URL to encode as QR")

    chk = sub.add_parser("check", help="Check whether an image already has a blind watermark")
    chk.add_argument("input", help="Input image path")

    ext = sub.add_parser("extract", help="Extract the blind QR watermark from an image")
    ext.add_argument("input", help="Input image path")
    ext.add_argument("--output", "-o", help="Save extracted QR image to this path")

    args = parser.parse_args()

    if args.command == "embed":
        try:
            embed(args.input, args.output, args.text)
        except (RuntimeError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "check":
        try:
            present = has_watermark(args.input)
            status = "YES" if present else "NO"
            msg = "already contains" if present else "does not contain"
            print(f"Watermark present: {status} — this image {msg} a blind QR watermark.")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "extract":
        try:
            extract(args.input, args.output)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
