#!/usr/bin/env python3
"""
Blind QR Watermark GUI (盲水印QR 图形界面)
Single-page drag-and-drop interface.
Drop (or click) an image → auto-detects watermark →
  • If found:  shows QR pattern + decoded content
  • If absent: shows embed form → export watermarked image
Watermark removal is intentionally not provided.
"""

import re
import sys
import threading
import tempfile
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    _DND_AVAILABLE = True
except ImportError:
    import tkinter as _tk
    TkinterDnD = _tk  # fallback: plain Tk
    DND_FILES = None
    _DND_AVAILABLE = False

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2

import blind_watermark_qr as wm_core


# ── palette ───────────────────────────────────────────────────────────────────
BG         = "#1e1e2e"
PANEL      = "#24273a"
CARD       = "#2e3045"
ACCENT     = "#7c6af7"
ACCENT_HVR = "#9d8fff"
ACCENT_DIM = "#4a4380"
TEXT       = "#cdd6f4"
SUBTEXT    = "#8087a2"
SUCCESS    = "#a6e3a1"
SUCCESS_BG = "#1e3a2e"
WARNING    = "#f9e2af"
WARNING_BG = "#3a3020"
ERROR      = "#f38ba8"
ERROR_BG   = "#3a1e28"
BORDER     = "#363a55"
DROP_HL    = "#7c6af7"


# ── helpers ───────────────────────────────────────────────────────────────────

def _pil_fit(pil: Image.Image, max_w: int, max_h: int) -> ImageTk.PhotoImage:
    c = pil.copy()
    c.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(c)


def _bgr_to_pil(arr: np.ndarray) -> Image.Image:
    if len(arr.shape) == 3:
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    return Image.fromarray(arr)


def _parse_dnd_path(raw: str) -> str:
    """Extract a single file path from tkinterdnd2 drop data."""
    raw = raw.strip()
    # Paths with spaces come wrapped in { }
    m = re.match(r"^\{(.+)\}$", raw)
    if m:
        return m.group(1)
    # Multiple files — take first
    parts = raw.split()
    return parts[0] if parts else raw


SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def _is_image_path(path: str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXT


# ── reusable widget pieces ────────────────────────────────────────────────────

def _btn(parent, text, cmd, accent=False, width=0, state="normal"):
    bg  = ACCENT if accent else CARD
    fg  = "#ffffff" if accent else TEXT
    abg = ACCENT_HVR if accent else BORDER
    kw  = dict(width=width) if width else {}
    b = tk.Button(
        parent, text=text, command=cmd,
        bg=bg, fg=fg, activebackground=abg, activeforeground=fg,
        relief="flat", bd=0, padx=14, pady=9,
        font=("Helvetica", 10, "bold" if accent else "normal"),
        cursor="hand2", state=state, **kw,
    )
    return b


def _lbl(parent, text="", bold=False, fg=TEXT, size=10, bg=None, anchor="w", wraplength=0):
    kw = dict(wraplength=wraplength) if wraplength else {}
    return tk.Label(
        parent, text=text,
        bg=bg or PANEL, fg=fg,
        font=("Helvetica", size, "bold" if bold else "normal"),
        anchor=anchor, **kw,
    )


def _sep(parent):
    f = tk.Frame(parent, bg=BORDER, height=1)
    f.pack(fill="x", pady=10)
    return f


class ImageCanvas(tk.Canvas):
    """Fixed-size canvas that displays a PIL image centred, or a placeholder."""

    def __init__(self, parent, w, h, placeholder="Drop or open an image", **kw):
        super().__init__(parent, width=w, height=h,
                         bg=CARD, highlightthickness=1,
                         highlightbackground=BORDER, **kw)
        self._w, self._h = w, h
        self._placeholder = placeholder
        self._ref = None
        self._draw_placeholder()

    def _draw_placeholder(self):
        self.delete("all")
        cx, cy = self._w // 2, self._h // 2
        self.create_text(cx, cy - 16, text="🖼",
                         font=("Segoe UI Emoji", 28), fill=BORDER)
        self.create_text(cx, cy + 18, text=self._placeholder,
                         font=("Helvetica", 10), fill=SUBTEXT)

    def show(self, pil: Image.Image):
        self._ref = _pil_fit(pil, self._w - 8, self._h - 8)
        self.delete("all")
        self.create_image(self._w // 2, self._h // 2,
                          anchor="center", image=self._ref)

    def reset(self):
        self._ref = None
        self._draw_placeholder()


# ── right-side panels (stacked, one shown at a time) ─────────────────────────

class _RightPanel(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=CARD,
                         highlightthickness=1, highlightbackground=BORDER)

    def show(self):
        self.pack(fill="both", expand=True)

    def hide(self):
        self.pack_forget()


class IdlePanel(_RightPanel):
    def __init__(self, parent):
        super().__init__(parent)
        tk.Label(self, text="←  Drop or open an image\n    to get started",
                 bg=CARD, fg=SUBTEXT, font=("Helvetica", 11),
                 justify="left").pack(expand=True)


class LoadingPanel(_RightPanel):
    def __init__(self, parent):
        super().__init__(parent)
        self._lbl = tk.Label(self, text="⏳  Analysing image…",
                             bg=CARD, fg=SUBTEXT, font=("Helvetica", 11))
        self._lbl.pack(expand=True)

    def set_text(self, t):
        self._lbl.config(text=t)


class WatermarkFoundPanel(_RightPanel):
    """Shown when a blind watermark IS detected."""

    def __init__(self, parent):
        super().__init__(parent)
        self._qr_arr = None
        self._build()

    def _build(self):
        p = tk.Frame(self, bg=CARD, padx=18, pady=14)
        p.pack(fill="both", expand=True)

        # Badge
        badge = tk.Frame(p, bg=SUCCESS_BG, padx=10, pady=6)
        badge.pack(fill="x", pady=(0, 14))
        tk.Label(badge, text="✅  Blind Watermark Detected",
                 bg=SUCCESS_BG, fg=SUCCESS,
                 font=("Helvetica", 11, "bold")).pack(anchor="w")

        # QR preview
        _lbl(p, "Embedded QR Pattern", bold=True, fg=TEXT, size=10, bg=CARD).pack(anchor="w")
        self._qr_canvas = ImageCanvas(p, 220, 220, placeholder="No QR extracted yet")
        self._qr_canvas.pack(pady=(6, 12))

        # Decoded content
        _lbl(p, "Decoded Content", bold=True, fg=TEXT, size=10, bg=CARD).pack(anchor="w")
        self._content_box = tk.Text(
            p, height=4, bg=BG, fg=SUCCESS,
            font=("Helvetica", 10), relief="flat",
            highlightthickness=1, highlightbackground=BORDER,
            padx=8, pady=6, wrap="word", state="disabled",
            insertbackground=TEXT,
        )
        self._content_box.pack(fill="x", pady=(4, 14))

        _sep(p)
        self._save_btn = _btn(p, "💾  Save QR Image", self._save, state="disabled")
        self._save_btn.pack(fill="x")

    def populate(self, qr_arr: np.ndarray | None, content: str | None, src_path: Path):
        self._qr_arr = qr_arr
        self._src_path = src_path

        if qr_arr is not None:
            self._qr_canvas.show(Image.fromarray(qr_arr))
            self._save_btn.config(state="normal")
        else:
            self._qr_canvas.reset()
            self._save_btn.config(state="disabled")

        self._content_box.config(state="normal")
        self._content_box.delete("1.0", "end")
        if content:
            self._content_box.insert("end", content)
            self._content_box.config(fg=SUCCESS)
        else:
            self._content_box.insert("end",
                "(Could not auto-decode — scan the saved QR image with a reader)")
            self._content_box.config(fg=SUBTEXT)
        self._content_box.config(state="disabled")

    def _save(self):
        if self._qr_arr is None:
            return
        default = (self._src_path.stem + "_extracted_qr.png")
        out = filedialog.asksaveasfilename(
            title="Save Extracted QR Image",
            initialfile=default,
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if out:
            Image.fromarray(self._qr_arr).save(out)
            messagebox.showinfo("Saved", f"QR image saved to:\n{out}")


class NoWatermarkPanel(_RightPanel):
    """Shown when no watermark is present — lets user add one and export."""

    def __init__(self, parent, on_embed_done):
        super().__init__(parent)
        self._on_embed_done = on_embed_done   # callback(wm_array, src_path)
        self._src_path: Path | None = None
        self._wm_array: np.ndarray | None = None
        self._build()

    def _build(self):
        p = tk.Frame(self, bg=CARD, padx=18, pady=14)
        p.pack(fill="both", expand=True)

        # Badge
        badge = tk.Frame(p, bg=WARNING_BG, padx=10, pady=6)
        badge.pack(fill="x", pady=(0, 14))
        tk.Label(badge, text="⚠  No Blind Watermark Found",
                 bg=WARNING_BG, fg=WARNING,
                 font=("Helvetica", 11, "bold")).pack(anchor="w")

        # Input
        _lbl(p, "Add Blind QR Watermark", bold=True, fg=TEXT, size=10, bg=CARD).pack(anchor="w")
        _lbl(p, "Enter text or URL to encode as an\ninvisible QR watermark:",
             fg=SUBTEXT, size=9, bg=CARD).pack(anchor="w", pady=(3, 6))

        self._entry = tk.Text(
            p, height=5, bg=BG, fg=TEXT, insertbackground=TEXT,
            font=("Helvetica", 10), relief="flat",
            highlightthickness=1, highlightbackground=BORDER,
            padx=8, pady=6, wrap="word",
        )
        self._entry.pack(fill="x", pady=(0, 10))

        self._embed_btn = _btn(p, "⬛  Add Blind Watermark", self._embed, accent=True)
        self._embed_btn.pack(fill="x")

        _sep(p)

        self._export_btn = _btn(p, "💾  Export Watermarked Image",
                                self._export, state="disabled")
        self._export_btn.pack(fill="x")

        self._note = _lbl(p, "", fg=SUBTEXT, size=9, bg=CARD)
        self._note.pack(anchor="w", pady=(8, 0))

    def set_source(self, path: Path):
        self._src_path = path
        self._wm_array = None
        self._export_btn.config(state="disabled")
        self._embed_btn.config(state="normal",
                               text="⬛  Add Blind Watermark", bg=ACCENT)
        self._note.config(text="", fg=SUBTEXT)
        self._entry.config(state="normal")

    def _embed(self):
        text = self._entry.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Missing Text",
                                   "Please enter text or URL for the watermark.")
            return
        if self._src_path is None:
            return

        self._embed_btn.config(state="disabled", text="⏳  Embedding…", bg=ACCENT_DIM)
        self._entry.config(state="disabled")
        self._note.config(text="", fg=SUBTEXT)
        self.update_idletasks()

        src = str(self._src_path)
        def _worker():
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.close()
                wm_core.embed(src, tmp.name, text)
                arr = cv2.imread(tmp.name)
                os.unlink(tmp.name)
                self.after(0, self._embed_done, arr, None)
            except Exception as e:
                self.after(0, self._embed_done, None, str(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _embed_done(self, arr, error):
        self._embed_btn.config(text="⬛  Add Blind Watermark")
        if error:
            self._embed_btn.config(state="normal", bg=ACCENT)
            self._entry.config(state="normal")
            self._note.config(text=f"Error: {error}", fg=ERROR)
            return

        self._wm_array = arr
        self._embed_btn.config(state="disabled", bg=ACCENT_DIM)
        self._entry.config(state="disabled")
        self._export_btn.config(state="normal")
        self._note.config(text="✓  Watermark embedded. Ready to export.", fg=SUCCESS)
        self._on_embed_done(arr, self._src_path)

    def _export(self):
        if self._wm_array is None:
            return
        default = (self._src_path.stem + "_watermarked.png") if self._src_path else "watermarked.png"
        out = filedialog.asksaveasfilename(
            title="Export Watermarked Image",
            initialfile=default,
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"),
                       ("JPEG image", "*.jpg"),
                       ("All files", "*.*")],
        )
        if not out:
            return
        cv2.imwrite(out, self._wm_array)
        messagebox.showinfo("Exported", f"Watermarked image saved to:\n{out}")


# ── drop zone ─────────────────────────────────────────────────────────────────

class DropZone(tk.Canvas):
    """Large dashed-border drop target shown before any image is loaded."""

    def __init__(self, parent, on_file, **kw):
        super().__init__(parent, bg=PANEL, highlightthickness=0, **kw)
        self._on_file = on_file
        self._draw()
        self.bind("<Button-1>", self._browse)
        self.bind("<Enter>",    lambda e: self._draw(hover=True))
        self.bind("<Leave>",    lambda e: self._draw(hover=False))
        if _DND_AVAILABLE:
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._on_drop)
            self.dnd_bind("<<DragEnter>>", lambda e: self._draw(hover=True))
            self.dnd_bind("<<DragLeave>>", lambda e: self._draw(hover=False))

    def _draw(self, hover=False):
        self.delete("all")
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        if w < 10:
            w, h = 560, 340
        border_col = DROP_HL if hover else BORDER
        # Dashed border via segments
        dash = 10
        for x in range(0, w, dash * 2):
            self.create_line(x, 4, min(x + dash, w), 4,
                             fill=border_col, width=2)
            self.create_line(x, h - 4, min(x + dash, w), h - 4,
                             fill=border_col, width=2)
        for y in range(0, h, dash * 2):
            self.create_line(4, y, 4, min(y + dash, h),
                             fill=border_col, width=2)
            self.create_line(w - 4, y, w - 4, min(y + dash, h),
                             fill=border_col, width=2)
        cx, cy = w // 2, h // 2
        icon_col = DROP_HL if hover else SUBTEXT
        self.create_text(cx, cy - 38, text="⬇",
                         font=("Helvetica", 36, "bold"), fill=icon_col)
        self.create_text(cx, cy + 14,
                         text="Drop image here  or  click to browse",
                         font=("Helvetica", 13), fill=TEXT if hover else SUBTEXT)
        self.create_text(cx, cy + 38,
                         text="PNG · JPG · JPEG · BMP · TIFF",
                         font=("Helvetica", 9), fill=SUBTEXT)

    def _browse(self, _=None):
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._on_file(path)

    def _on_drop(self, event):
        self._draw(hover=False)
        path = _parse_dnd_path(event.data)
        if _is_image_path(path):
            self._on_file(path)
        else:
            messagebox.showwarning("Unsupported File",
                                   f"Only image files are supported.\nDropped: {path}")


# ── main window ───────────────────────────────────────────────────────────────

class App(TkinterDnD.Tk if _DND_AVAILABLE else tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("盲水印QR  ·  Blind QR Watermark Tool")
        self.geometry("900x580")
        self.minsize(720, 480)
        self.configure(bg=PANEL)
        self._src_path: Path | None = None
        self._build()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build(self):
        # ── header ──────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=ACCENT, padx=18, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="盲水印QR  Blind QR Watermark Tool",
                 bg=ACCENT, fg="#ffffff",
                 font=("Helvetica", 14, "bold")).pack(side="left")
        tk.Label(hdr,
                 text="Invisible QR watermarks — drop an image to begin",
                 bg=ACCENT, fg="#cdc8ff",
                 font=("Helvetica", 9)).pack(side="left", padx=(14, 0))

        # ── status bar ───────────────────────────────────────────────────
        self._status_lbl = tk.Label(self, text="Ready — drag & drop or click to open an image",
                                    bg=BG, fg=SUBTEXT,
                                    font=("Helvetica", 9), anchor="w",
                                    padx=12, pady=5)
        self._status_lbl.pack(side="bottom", fill="x")
        tk.Frame(self, bg=BORDER, height=1).pack(side="bottom", fill="x")

        # ── toolbar (shown once image loaded) ────────────────────────────
        self._toolbar = tk.Frame(self, bg=PANEL, padx=12, pady=6)
        # (packed dynamically)
        self._file_lbl = _lbl(self._toolbar, "", fg=SUBTEXT, size=9, bg=PANEL)
        self._file_lbl.pack(side="left")
        _btn(self._toolbar, "↩  Load Another Image", self._reset, width=22).pack(side="right")

        # ── body ────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=PANEL)
        body.pack(fill="both", expand=True, padx=12, pady=(8, 8))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2, minsize=260)
        body.rowconfigure(0, weight=1)

        # Left: drop zone / image preview
        left = tk.Frame(body, bg=PANEL)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self._drop_zone = DropZone(left, self._load_file,
                                   width=520, height=400)
        self._drop_zone.pack(fill="both", expand=True)

        self._img_canvas = ImageCanvas(left, 520, 400,
                                       placeholder="Image preview")
        # (shown when image loaded)

        # Right: stacked panels
        right = tk.Frame(body, bg=PANEL)
        right.grid(row=0, column=1, sticky="nsew")

        self._idle_panel    = IdlePanel(right)
        self._loading_panel = LoadingPanel(right)
        self._found_panel   = WatermarkFoundPanel(right)
        self._none_panel    = NoWatermarkPanel(right, self._on_embed_done)

        self._idle_panel.show()

    # ── state transitions ─────────────────────────────────────────────────────

    def _set_status(self, msg, colour=SUBTEXT):
        self._status_lbl.config(text=msg, fg=colour)
        self.update_idletasks()

    def _show_panel(self, panel):
        for p in (self._idle_panel, self._loading_panel,
                  self._found_panel, self._none_panel):
            p.hide()
        panel.show()

    def _reset(self):
        self._src_path = None
        self._img_canvas.pack_forget()
        self._drop_zone.pack(fill="both", expand=True)
        self._toolbar.pack_forget()
        self._show_panel(self._idle_panel)
        self._set_status("Ready — drag & drop or click to open an image")

    def _load_file(self, path: str):
        path = path.strip()
        if not os.path.isfile(path):
            messagebox.showerror("Not Found", f"File not found:\n{path}")
            return
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open image:\n{e}")
            return

        self._src_path = Path(path)

        # Switch to image view
        self._drop_zone.pack_forget()
        self._img_canvas.show(pil)
        self._img_canvas.pack(fill="both", expand=True)

        self._toolbar.pack(fill="x", before=self._status_lbl)
        self._file_lbl.config(text=f"  {self._src_path.name}")

        self._show_panel(self._loading_panel)
        self._loading_panel.set_text("⏳  Checking for blind watermark…")
        self._set_status(f"Analysing: {self._src_path.name}")

        threading.Thread(target=self._analyse, args=(path,), daemon=True).start()

    def _analyse(self, path: str):
        try:
            img = cv2.imread(path)
            if img is None:
                raise ValueError("Cannot read image data.")

            has_wm = wm_core._read_marker(img)

            if has_wm:
                # Extract QR
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                qr_arr = wm_core._extract_qr_dft(gray)

                # Try to decode
                content = None
                try:
                    from pyzbar.pyzbar import decode as pyzbar_decode
                    pil_qr = Image.fromarray(qr_arr)
                    decoded = pyzbar_decode(pil_qr)
                    if not decoded:
                        decoded = pyzbar_decode(Image.fromarray(255 - qr_arr))
                    if decoded:
                        content = decoded[0].data.decode("utf-8", errors="replace")
                except ImportError:
                    content = None

                self.after(0, self._show_found, qr_arr, content)
            else:
                self.after(0, self._show_none)

        except Exception as e:
            self.after(0, self._show_error, str(e))

    def _show_found(self, qr_arr, content):
        self._found_panel.populate(qr_arr, content, self._src_path)
        self._show_panel(self._found_panel)
        msg = f"✓  Watermark found in {self._src_path.name}"
        if content:
            short = content[:60] + ("…" if len(content) > 60 else "")
            msg += f"  ·  {short}"
        self._set_status(msg, SUCCESS)

    def _show_none(self):
        self._none_panel.set_source(self._src_path)
        self._show_panel(self._none_panel)
        self._set_status(
            f"No watermark in {self._src_path.name}  — enter text below to add one.",
            WARNING,
        )

    def _show_error(self, err):
        self._show_panel(self._idle_panel)
        self._set_status(f"Error: {err}", ERROR)
        messagebox.showerror("Error", err)

    def _on_embed_done(self, wm_array, src_path):
        """Called by NoWatermarkPanel after successful embedding."""
        pil = _bgr_to_pil(wm_array)
        self._img_canvas.show(pil)
        self._set_status(
            f"✓  Blind watermark embedded in {src_path.name}  — export to save.",
            SUCCESS,
        )


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    if not _DND_AVAILABLE:
        print("Note: tkinterdnd2 not found — drag-and-drop disabled; use click-to-browse.")
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
