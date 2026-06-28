#!/usr/bin/env python3
"""
Blind QR Watermark GUI (盲水印QR 图形界面)
Three-tab interface: Embed · Check · Extract
Watermark removal is intentionally not provided.
"""

import sys
import io
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

import blind_watermark_qr as wm_core


# ── colour palette ────────────────────────────────────────────────────────────
BG         = "#1e1e2e"
PANEL      = "#2a2a3e"
CARD       = "#313149"
ACCENT     = "#7c6af7"
ACCENT_HVR = "#9d8fff"
TEXT       = "#cdd6f4"
SUBTEXT    = "#a6adc8"
SUCCESS    = "#a6e3a1"
WARNING    = "#f9e2af"
ERROR      = "#f38ba8"
BORDER     = "#45475a"


def _pil_to_tk(pil_img: Image.Image, max_w: int, max_h: int) -> ImageTk.PhotoImage:
    pil_img = pil_img.copy()
    pil_img.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(pil_img)


def _cv2_array_to_pil(arr: np.ndarray) -> Image.Image:
    import cv2
    if len(arr.shape) == 3:
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    return Image.fromarray(arr)


class PlaceholderCanvas(tk.Canvas):
    """Canvas that shows a placeholder until an image is loaded."""

    def __init__(self, parent, width: int, height: int, **kw):
        super().__init__(
            parent,
            width=width, height=height,
            bg=CARD, highlightthickness=1,
            highlightbackground=BORDER,
            **kw,
        )
        self._w = width
        self._h = height
        self._draw_placeholder()

    def _draw_placeholder(self):
        self.delete("all")
        self.create_rectangle(0, 0, self._w, self._h, fill=CARD, outline="")
        cx, cy = self._w // 2, self._h // 2
        self.create_text(cx, cy - 14, text="🖼", font=("Segoe UI Emoji", 32), fill=BORDER)
        self.create_text(cx, cy + 22, text="No image loaded",
                         font=("Helvetica", 10), fill=SUBTEXT)

    def show_image(self, pil_img: Image.Image):
        self._img_ref = _pil_to_tk(pil_img, self._w - 4, self._h - 4)
        self.delete("all")
        cx, cy = self._w // 2, self._h // 2
        self.create_image(cx, cy, anchor="center", image=self._img_ref)

    def reset(self):
        self._img_ref = None
        self._draw_placeholder()


class StatusBar(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=PANEL, pady=6, padx=10)
        self._lbl = tk.Label(self, text="Ready", bg=PANEL, fg=SUBTEXT,
                             font=("Helvetica", 10), anchor="w")
        self._lbl.pack(fill="x")

    def set(self, msg: str, kind: str = "info"):
        colours = {"info": SUBTEXT, "ok": SUCCESS, "warn": WARNING, "error": ERROR}
        self._lbl.config(text=msg, fg=colours.get(kind, SUBTEXT))
        self.update_idletasks()


def _styled_btn(parent, text: str, command, accent=False, width=22):
    bg   = ACCENT if accent else CARD
    fg   = "#ffffff" if accent else TEXT
    abg  = ACCENT_HVR if accent else BORDER
    btn = tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, activebackground=abg, activeforeground=fg,
        relief="flat", bd=0, padx=12, pady=8,
        font=("Helvetica", 10, "bold" if accent else "normal"),
        cursor="hand2", width=width,
    )
    return btn


def _label(parent, text, bold=False, colour=TEXT, size=10):
    return tk.Label(
        parent, text=text, bg=PANEL, fg=colour,
        font=("Helvetica", size, "bold" if bold else "normal"),
        anchor="w",
    )


# ── Tab: Embed ────────────────────────────────────────────────────────────────

class EmbedTab(tk.Frame):
    def __init__(self, parent, status: StatusBar):
        super().__init__(parent, bg=PANEL)
        self._status = status
        self._src_path: Path | None = None
        self._wm_array: np.ndarray | None = None  # watermarked image (BGR numpy)
        self._build()

    def _build(self):
        top = tk.Frame(self, bg=PANEL, pady=8, padx=12)
        top.pack(fill="x")
        _label(top, "Source Image", bold=True, size=11).pack(side="left")
        _styled_btn(top, "Open Image…", self._open, width=14).pack(side="right")

        self._path_lbl = _label(top, "No file selected", colour=SUBTEXT)
        self._path_lbl.pack(side="left", padx=(10, 0))

        content = tk.Frame(self, bg=PANEL, padx=12, pady=4)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=1, minsize=240)
        content.rowconfigure(0, weight=1)

        # Image preview
        self._canvas = PlaceholderCanvas(content, 460, 360)
        self._canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=4)

        # Controls panel
        ctrl = tk.Frame(content, bg=CARD, padx=16, pady=16,
                        highlightthickness=1, highlightbackground=BORDER)
        ctrl.grid(row=0, column=1, sticky="nsew", pady=4)

        _label(ctrl, "Watermark Text / URL", bold=True, colour=TEXT, size=10).pack(anchor="w")
        _label(ctrl, "This text will be encoded as a QR code\n"
               "and invisibly embedded in the image.", colour=SUBTEXT, size=9).pack(anchor="w", pady=(2, 8))

        self._text_var = tk.StringVar()
        self._text_entry = tk.Text(ctrl, height=5, bg=BG, fg=TEXT, insertbackground=TEXT,
                                   font=("Helvetica", 10), relief="flat",
                                   highlightthickness=1, highlightbackground=BORDER,
                                   wrap="word", padx=6, pady=6)
        self._text_entry.pack(fill="x", pady=(0, 12))

        self._embed_btn = _styled_btn(ctrl, "⬛  Embed Watermark", self._embed, accent=True)
        self._embed_btn.pack(fill="x", pady=(0, 8))
        self._embed_btn.config(state="disabled")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=10)

        self._export_btn = _styled_btn(ctrl, "💾  Export Watermarked Image", self._export)
        self._export_btn.pack(fill="x")
        self._export_btn.config(state="disabled")

        self._wm_note = _label(ctrl, "", colour=SUBTEXT, size=9)
        self._wm_note.pack(anchor="w", pady=(6, 0))

    def _open(self):
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        self._src_path = Path(path)
        self._wm_array = None
        self._export_btn.config(state="disabled")
        self._wm_note.config(text="")

        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open image:\n{e}")
            return

        self._canvas.show_image(pil)
        self._path_lbl.config(text=self._src_path.name)

        # Check for existing watermark and update UI accordingly
        try:
            if wm_core.has_watermark(path):
                self._embed_btn.config(state="disabled")
                self._status.set(
                    "⚠  This image already contains a blind watermark. Embedding blocked.",
                    "warn"
                )
                self._wm_note.config(
                    text="Already watermarked — embedding not allowed.", fg=WARNING
                )
            else:
                self._embed_btn.config(state="normal")
                self._status.set("Image loaded. Enter watermark text and click Embed.", "info")
                self._wm_note.config(text="", fg=SUBTEXT)
        except Exception as e:
            self._status.set(f"Error reading image: {e}", "error")

    def _embed(self):
        if self._src_path is None:
            return
        text = self._text_entry.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Missing Text", "Please enter watermark text or URL.")
            return

        self._embed_btn.config(state="disabled", text="⏳  Embedding…")
        self._status.set("Embedding blind QR watermark…", "info")
        self.update_idletasks()

        def _worker():
            import cv2, tempfile, os
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp_path = tmp.name
                tmp.close()

                wm_core.embed(str(self._src_path), tmp_path, text)

                arr = cv2.imread(tmp_path)
                os.unlink(tmp_path)

                self.after(0, self._embed_done, arr, None)
            except Exception as e:
                self.after(0, self._embed_done, None, str(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _embed_done(self, arr, error):
        self._embed_btn.config(text="⬛  Embed Watermark")
        if error:
            self._embed_btn.config(state="normal")
            self._status.set(f"Error: {error}", "error")
            messagebox.showerror("Embed Failed", error)
            return

        self._wm_array = arr
        pil = _cv2_array_to_pil(arr)
        self._canvas.show_image(pil)
        self._export_btn.config(state="normal")
        self._embed_btn.config(state="disabled")   # prevent double-embed
        self._status.set("✓  Watermark embedded successfully. Ready to export.", "ok")
        self._wm_note.config(
            text="Watermark embedded — image ready to export.", fg=SUCCESS
        )

    def _export(self):
        if self._wm_array is None:
            return
        default_name = self._src_path.stem + "_watermarked.png" if self._src_path else "watermarked.png"
        out_path = filedialog.asksaveasfilename(
            title="Export Watermarked Image",
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("All files", "*.*")],
        )
        if not out_path:
            return
        import cv2
        cv2.imwrite(out_path, self._wm_array)
        self._status.set(f"✓  Exported to {Path(out_path).name}", "ok")
        messagebox.showinfo("Exported", f"Watermarked image saved to:\n{out_path}")


# ── Tab: Check ────────────────────────────────────────────────────────────────

class CheckTab(tk.Frame):
    def __init__(self, parent, status: StatusBar):
        super().__init__(parent, bg=PANEL)
        self._status = status
        self._src_path: Path | None = None
        self._build()

    def _build(self):
        top = tk.Frame(self, bg=PANEL, pady=8, padx=12)
        top.pack(fill="x")
        _label(top, "Detect Watermark", bold=True, size=11).pack(side="left")
        _styled_btn(top, "Open Image…", self._open, width=14).pack(side="right")
        self._path_lbl = _label(top, "No file selected", colour=SUBTEXT)
        self._path_lbl.pack(side="left", padx=(10, 0))

        content = tk.Frame(self, bg=PANEL, padx=12, pady=4)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=1, minsize=240)
        content.rowconfigure(0, weight=1)

        self._canvas = PlaceholderCanvas(content, 460, 360)
        self._canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=4)

        ctrl = tk.Frame(content, bg=CARD, padx=16, pady=16,
                        highlightthickness=1, highlightbackground=BORDER)
        ctrl.grid(row=0, column=1, sticky="nsew", pady=4)

        _label(ctrl, "Watermark Detection", bold=True).pack(anchor="w", pady=(0, 6))
        _label(ctrl, "Checks whether this image already\n"
               "contains a blind QR watermark.", colour=SUBTEXT, size=9).pack(anchor="w", pady=(0, 16))

        self._check_btn = _styled_btn(ctrl, "🔍  Check Watermark", self._check, accent=True)
        self._check_btn.pack(fill="x")
        self._check_btn.config(state="disabled")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=16)

        # Result badge
        self._result_frame = tk.Frame(ctrl, bg=CARD)
        self._result_frame.pack(fill="x")
        self._result_icon = _label(self._result_frame, "", size=28, colour=SUBTEXT)
        self._result_icon.pack()
        self._result_lbl = _label(self._result_frame, "—", colour=SUBTEXT, bold=True, size=11)
        self._result_lbl.pack()
        self._result_sub = _label(self._result_frame, "", colour=SUBTEXT, size=9)
        self._result_sub.pack(pady=(4, 0))

    def _open(self):
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        self._src_path = Path(path)
        self._result_icon.config(text="", fg=SUBTEXT)
        self._result_lbl.config(text="—", fg=SUBTEXT)
        self._result_sub.config(text="")
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open image:\n{e}")
            return
        self._canvas.show_image(pil)
        self._path_lbl.config(text=self._src_path.name)
        self._check_btn.config(state="normal")
        self._status.set("Image loaded. Click Check Watermark.", "info")

    def _check(self):
        if not self._src_path:
            return
        self._check_btn.config(state="disabled", text="⏳  Checking…")
        self._status.set("Checking for blind watermark…", "info")
        self.update_idletasks()

        def _worker():
            try:
                result = wm_core.has_watermark(str(self._src_path))
                self.after(0, self._check_done, result, None)
            except Exception as e:
                self.after(0, self._check_done, None, str(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _check_done(self, result, error):
        self._check_btn.config(state="normal", text="🔍  Check Watermark")
        if error:
            self._status.set(f"Error: {error}", "error")
            return
        if result:
            self._result_icon.config(text="✅", fg=SUCCESS)
            self._result_lbl.config(text="Watermark Found", fg=SUCCESS)
            self._result_sub.config(
                text="This image contains\na blind QR watermark.", fg=SUBTEXT
            )
            self._status.set("✓  Blind watermark detected in this image.", "ok")
        else:
            self._result_icon.config(text="❌", fg=ERROR)
            self._result_lbl.config(text="No Watermark", fg=ERROR)
            self._result_sub.config(
                text="No blind watermark\nwas found in this image.", fg=SUBTEXT
            )
            self._status.set("No blind watermark found in this image.", "info")


# ── Tab: Extract ──────────────────────────────────────────────────────────────

class ExtractTab(tk.Frame):
    def __init__(self, parent, status: StatusBar):
        super().__init__(parent, bg=PANEL)
        self._status = status
        self._src_path: Path | None = None
        self._qr_array: np.ndarray | None = None
        self._build()

    def _build(self):
        top = tk.Frame(self, bg=PANEL, pady=8, padx=12)
        top.pack(fill="x")
        _label(top, "Extract Watermark QR", bold=True, size=11).pack(side="left")
        _styled_btn(top, "Open Image…", self._open, width=14).pack(side="right")
        self._path_lbl = _label(top, "No file selected", colour=SUBTEXT)
        self._path_lbl.pack(side="left", padx=(10, 0))

        content = tk.Frame(self, bg=PANEL, padx=12, pady=4)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=2)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        # Source image
        left = tk.Frame(content, bg=PANEL)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=4)
        _label(left, "Source Image", colour=SUBTEXT, size=9).pack(anchor="w", pady=(0, 4))
        self._src_canvas = PlaceholderCanvas(left, 360, 320)
        self._src_canvas.pack(fill="both", expand=True)

        self._extract_btn = _styled_btn(left, "🔓  Extract QR Watermark", self._extract, accent=True)
        self._extract_btn.pack(fill="x", pady=(8, 0))
        self._extract_btn.config(state="disabled")

        # Extracted QR
        right = tk.Frame(content, bg=PANEL)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=4)
        _label(right, "Extracted QR Code", colour=SUBTEXT, size=9).pack(anchor="w", pady=(0, 4))
        self._qr_canvas = PlaceholderCanvas(right, 360, 320)
        self._qr_canvas.pack(fill="both", expand=True)

        self._save_btn = _styled_btn(right, "💾  Save QR Image", self._save_qr)
        self._save_btn.pack(fill="x", pady=(8, 0))
        self._save_btn.config(state="disabled")

        self._content_lbl = _label(right, "", colour=SUBTEXT, size=9)
        self._content_lbl.pack(anchor="w", pady=(6, 0))

    def _open(self):
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        self._src_path = Path(path)
        self._qr_array = None
        self._save_btn.config(state="disabled")
        self._content_lbl.config(text="")
        self._qr_canvas.reset()
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open image:\n{e}")
            return
        self._src_canvas.show_image(pil)
        self._path_lbl.config(text=self._src_path.name)
        self._extract_btn.config(state="normal")
        self._status.set("Image loaded. Click Extract QR Watermark.", "info")

    def _extract(self):
        if not self._src_path:
            return
        self._extract_btn.config(state="disabled", text="⏳  Extracting…")
        self._status.set("Extracting blind QR watermark…", "info")
        self.update_idletasks()

        def _worker():
            import cv2, tempfile, os
            try:
                img = cv2.imread(str(self._src_path))
                if img is None:
                    raise ValueError("Cannot read image file.")

                if not wm_core._read_marker(img):
                    self.after(0, self._extract_done, None, None, "no_watermark")
                    return

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                qr_arr = wm_core._extract_qr_dft(gray)

                # Try to decode QR content
                content = None
                try:
                    from pyzbar.pyzbar import decode as pyzbar_decode
                    decoded = pyzbar_decode(Image.fromarray(qr_arr))
                    if not decoded:
                        decoded = pyzbar_decode(Image.fromarray(255 - qr_arr))
                    if decoded:
                        content = decoded[0].data.decode("utf-8", errors="replace")
                except ImportError:
                    content = "(install pyzbar to auto-decode QR content)"

                self.after(0, self._extract_done, qr_arr, content, None)
            except Exception as e:
                self.after(0, self._extract_done, None, None, str(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _extract_done(self, qr_arr, content, error):
        self._extract_btn.config(state="normal", text="🔓  Extract QR Watermark")
        if error == "no_watermark":
            self._status.set("No blind watermark detected in this image.", "warn")
            messagebox.showinfo("No Watermark", "This image does not contain a blind QR watermark.")
            return
        if error:
            self._status.set(f"Error: {error}", "error")
            messagebox.showerror("Extraction Failed", str(error))
            return

        self._qr_array = qr_arr
        pil = Image.fromarray(qr_arr)
        self._qr_canvas.show_image(pil)
        self._save_btn.config(state="normal")

        if content:
            self._content_lbl.config(
                text=f"QR Content:\n{content[:80]}{'…' if len(content) > 80 else ''}",
                fg=SUCCESS,
            )
            self._status.set(f"✓  QR extracted. Content: {content[:60]}", "ok")
        else:
            self._content_lbl.config(
                text="QR extracted. Use a scanner to read the content.", fg=SUBTEXT
            )
            self._status.set("✓  QR watermark pattern extracted successfully.", "ok")

    def _save_qr(self):
        if self._qr_array is None:
            return
        default_name = (self._src_path.stem + "_extracted_qr.png") if self._src_path else "extracted_qr.png"
        out_path = filedialog.asksaveasfilename(
            title="Save Extracted QR Image",
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not out_path:
            return
        Image.fromarray(self._qr_array).save(out_path)
        self._status.set(f"✓  QR image saved to {Path(out_path).name}", "ok")
        messagebox.showinfo("Saved", f"Extracted QR image saved to:\n{out_path}")


# ── Main window ───────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("盲水印QR  ·  Blind QR Watermark Tool")
        self.geometry("820x600")
        self.minsize(700, 520)
        self.configure(bg=BG)
        self._build()

    def _build(self):
        # Header
        header = tk.Frame(self, bg=ACCENT, padx=18, pady=10)
        header.pack(fill="x")
        tk.Label(
            header, text="盲水印QR  Blind QR Watermark Tool",
            bg=ACCENT, fg="#ffffff", font=("Helvetica", 14, "bold"),
        ).pack(side="left")
        tk.Label(
            header,
            text="Invisible QR watermarks — embed · detect · extract",
            bg=ACCENT, fg="#d8d0ff", font=("Helvetica", 9),
        ).pack(side="left", padx=(14, 0))

        # Status bar
        self._status = StatusBar(self)
        self._status.pack(side="bottom", fill="x")
        ttk.Separator(self, orient="horizontal").pack(side="bottom", fill="x")

        # Notebook (tabs)
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("App.TNotebook",
                         background=BG, borderwidth=0, tabmargins=[0, 0, 0, 0])
        style.configure("App.TNotebook.Tab",
                         background=PANEL, foreground=SUBTEXT,
                         padding=[18, 8], font=("Helvetica", 10),
                         borderwidth=0)
        style.map("App.TNotebook.Tab",
                  background=[("selected", PANEL)],
                  foreground=[("selected", TEXT)],
                  expand=[("selected", [0, 0, 0, 2])])

        nb = ttk.Notebook(self, style="App.TNotebook")
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        self._embed_tab   = EmbedTab(nb, self._status)
        self._check_tab   = CheckTab(nb, self._status)
        self._extract_tab = ExtractTab(nb, self._status)

        nb.add(self._embed_tab,   text="  ⬛  Embed Watermark  ")
        nb.add(self._check_tab,   text="  🔍  Detect Watermark  ")
        nb.add(self._extract_tab, text="  🔓  Extract QR  ")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
