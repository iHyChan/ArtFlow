import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from PIL import Image, ImageTk, ImageDraw
import os

from model import run_transfer, device

STYLE_PRESETS = {
    "Starry Night · Van Gogh":    "style_images/starry_night.jpg",
    "Impression Sunrise · Monet": "style_images/impression_sunrise.jpg",
    "The Scream · Munch":         "style_images/the_scream.jpg",
    "The Great Wave · Hokusai":   "style_images/the_wave.jpg",
}

BG        = "#F5F6FA"
WHITE     = "#FFFFFF"
BORDER    = "#E0E3EA"
ACCENT    = "#4361EE"
ACCENT2   = "#3A0CA3"
TEXT      = "#1A1A2E"
SUBTEXT   = "#6B7280"
SUCCESS   = "#10B981"
TAG_BG    = "#EEF2FF"
CANVAS_BG = "#F0F2F8"
TOPBAR_H  = 58
STATUS_H  = 52
SIDEBAR_W = 280


def make_placeholder(w, h, text="Upload Image"):
    w, h = max(w, 10), max(h, 10)
    img  = Image.new("RGB", (w, h), CANVAS_BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([1, 1, w - 2, h - 2], outline=BORDER, width=1)
    if w > 60 and h > 60:
        cx, cy = w // 2, h // 2 - 18
        r = min(20, w // 6, h // 6)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     outline="#C0C8E0", width=2)
        draw.line([cx, cy - r + 4, cx, cy + r - 4],
                  fill="#C0C8E0", width=2)
        draw.line([cx - r + 4, cy, cx + r - 4, cy],
                  fill="#C0C8E0", width=2)
        tw = draw.textlength(text)
        draw.text((w // 2 - tw // 2, h // 2 + 10),
                  text, fill=SUBTEXT)
    return img


def fit_image(pil_img, w, h):
    w, h = max(w, 1), max(h, 1)
    iw, ih = pil_img.size
    scale  = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = pil_img.resize((nw, nh), Image.LANCZOS)
    canvas  = Image.new("RGB", (w, h), CANVAS_BG)
    ox, oy  = (w - nw) // 2, (h - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas


class ImagePanel(tk.Frame):
    def __init__(self, parent, title, placeholder_text, **kw):
        super().__init__(parent, bg=WHITE,
                         highlightbackground=BORDER,
                         highlightthickness=1, **kw)
        self._pil_img         = None
        self._placeholder_text = placeholder_text
        self._tk_img          = None

        hdr = tk.Frame(self, bg=WHITE, height=36)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Frame(hdr, bg=BORDER, height=1).pack(
            side="bottom", fill="x")
        tk.Label(hdr, text=title,
                 bg=WHITE, fg=TEXT,
                 font=("Segoe UI", 10, "bold"),
                 padx=14).pack(side="left", fill="y")

        self._lbl = tk.Label(self, bg=CANVAS_BG)
        self._lbl.pack(fill="both", expand=True,
                       padx=10, pady=10)

        self._lbl.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        w, h = event.width, event.height
        if w < 4 or h < 4:
            return
        if self._pil_img:
            rendered = fit_image(self._pil_img, w, h)
        else:
            rendered = make_placeholder(w, h, self._placeholder_text)
        tk_img = ImageTk.PhotoImage(rendered)
        self._lbl.configure(image=tk_img)
        self._lbl.image = tk_img
        self._tk_img    = tk_img

    def set_image(self, pil_img):
        self._pil_img = pil_img
        self._lbl.update_idletasks()
        w = self._lbl.winfo_width()
        h = self._lbl.winfo_height()
        if w > 4 and h > 4:
            rendered = fit_image(pil_img, w, h)
            tk_img   = ImageTk.PhotoImage(rendered)
            self._lbl.configure(image=tk_img)
            self._lbl.image = tk_img
            self._tk_img    = tk_img

    def clear(self):
        self._pil_img = None
        self._lbl.update_idletasks()
        w = self._lbl.winfo_width()
        h = self._lbl.winfo_height()
        rendered = make_placeholder(
            max(w, 10), max(h, 10), self._placeholder_text)
        tk_img = ImageTk.PhotoImage(rendered)
        self._lbl.configure(image=tk_img)
        self._lbl.image = tk_img


class StyleTransferApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ArtFlow · Neural Style Transfer")
        self.root.geometry("1400x980")
        self.root.minsize(1000, 680)
        self.root.configure(bg=BG)

        self.content_path    = None
        self.style_path      = None
        self.result_image    = None
        self.running         = False
        self.selected_preset = tk.StringVar(
            value="Starry Night · Van Gogh")

        self._apply_ttk_styles()
        self._build_ui()
        self._update_style_preview()

    def _apply_ttk_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TProgressbar",
                    troughcolor=BORDER,
                    background=ACCENT,
                    bordercolor=BORDER,
                    lightcolor=ACCENT,
                    darkcolor=ACCENT,
                    thickness=5)
        s.configure("TScale",
                    background=WHITE,
                    troughcolor="#D1D5E8",
                    sliderlength=14,
                    sliderrelief="flat")
        s.map("TScale",
              background=[("active", WHITE)])

    def _build_ui(self):
        self._build_topbar()

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        self._build_sidebar(body)
        self._build_main_area(body)

    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=WHITE,
                       height=TOPBAR_H)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        tk.Frame(bar, bg=ACCENT, width=5).pack(
            side="left", fill="y")

        tk.Label(bar, text="ArtFlow",
                 bg=WHITE, fg=ACCENT,
                 font=("Segoe UI", 17, "bold"),
                 padx=18).pack(side="left")

        tk.Label(bar, text="·  Neural Style Transfer",
                 bg=WHITE, fg=TEXT,
                 font=("Segoe UI", 13)).pack(side="left")

        dev_color = SUCCESS if device.type == "cuda" else SUBTEXT
        dev_text  = ("● GPU (CUDA)" if device.type == "cuda"
                     else "● CPU")
        tk.Label(bar, text=dev_text,
                 bg=WHITE, fg=dev_color,
                 font=("Segoe UI", 9),
                 padx=24).pack(side="right")

        tk.Frame(self.root, bg=BORDER, height=1).pack(
            fill="x", side="top")

    def _build_sidebar(self, parent):
        self._sidebar = tk.Frame(parent, bg=WHITE,
                                 width=SIDEBAR_W,
                                 highlightbackground=BORDER,
                                 highlightthickness=1)
        self._sidebar.pack(side="left", fill="y",
                           padx=(16, 0), pady=16)
        self._sidebar.pack_propagate(False)
        self._populate_sidebar(self._sidebar)

    def _populate_sidebar(self, sb):
        self._sb_section(sb, "CONTENT IMAGE")

        self._thumb_content = tk.Label(
            sb, bg=CANVAS_BG, cursor="hand2",
            highlightbackground=BORDER,
            highlightthickness=1)
        self._thumb_content.pack(
            padx=14, pady=(4, 6), fill="x")
        self._thumb_content.configure(height=148)
        self._render_thumb(self._thumb_content,
                           make_placeholder(240, 148,
                                            "Click to upload"))
        self._thumb_content.bind(
            "<Button-1>", lambda e: self._upload_content())

        self._sb_btn(sb, "Browse Content Image",
                     self._upload_content)
        self._sb_divider(sb)

        self._sb_section(sb, "STYLE PRESET")
        for name in STYLE_PRESETS:
            tk.Radiobutton(
                sb, text=name,
                variable=self.selected_preset,
                value=name,
                bg=WHITE, fg=TEXT,
                selectcolor=TAG_BG,
                activebackground=WHITE,
                activeforeground=ACCENT,
                font=("Segoe UI", 9),
                command=self._update_style_preview
            ).pack(anchor="w", padx=18, pady=2)

        self._thumb_style = tk.Label(
            sb, bg=CANVAS_BG,
            highlightbackground=BORDER,
            highlightthickness=1)
        self._thumb_style.pack(
            padx=14, pady=(6, 4), fill="x")
        self._thumb_style.configure(height=130)

        self._sb_divider(sb)
        self._sb_section(sb, "PARAMETERS")

        self._make_slider(sb, "Style Strength (α)",
                          "alpha_var", 0.1, 1.0, 0.8, float)
        self._make_slider(sb, "Iterations",
                          "steps_var", 50, 300, 100, int)
        self._make_slider(sb, "Image Size (px)",
                          "size_var", 128, 512, 256, int)

        self._sb_divider(sb)

        self.run_btn = tk.Button(
            sb,
            text="▶   Run Style Transfer",
            bg=ACCENT, fg=WHITE,
            font=("Segoe UI", 10, "bold"),
            relief="flat", cursor="hand2",
            activebackground=ACCENT2,
            activeforeground=WHITE,
            bd=0,
            command=self._start_transfer)
        self.run_btn.pack(padx=14, pady=(4, 18),
                          fill="x", ipady=11)

    def _sb_section(self, parent, text):
        f = tk.Frame(parent, bg=WHITE)
        f.pack(fill="x", padx=14, pady=(14, 4))
        tk.Label(f, text=text,
                 bg=WHITE, fg=ACCENT,
                 font=("Segoe UI", 8, "bold")).pack(side="left")

    def _sb_divider(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(
            fill="x", padx=14, pady=5)

    def _sb_btn(self, parent, text, cmd):
        tk.Button(parent, text=text,
                  bg=ACCENT, fg=WHITE,
                  font=("Segoe UI", 9),
                  relief="flat", cursor="hand2",
                  activebackground=ACCENT2,
                  activeforeground=WHITE,
                  bd=0, command=cmd
                  ).pack(padx=14, pady=(0, 6),
                         fill="x", ipady=7)

    def _make_slider(self, parent, label, attr,
                     from_, to, default, cast):
        var  = tk.DoubleVar(value=default)
        setattr(self, attr, var)
        disp = tk.StringVar(
            value=str(default if cast == float
                       else int(default)))

        def on_change(*_):
            v = (round(var.get(), 2) if cast == float
                 else int(var.get()))
            disp.set(str(v))

        var.trace_add("write", on_change)

        row = tk.Frame(parent, bg=WHITE)
        row.pack(fill="x", padx=14, pady=(4, 0))
        tk.Label(row, text=label,
                 bg=WHITE, fg=TEXT,
                 font=("Segoe UI", 9)).pack(side="left")

        tag = tk.Frame(row, bg=TAG_BG, padx=7, pady=1)
        tag.pack(side="right")
        tk.Label(tag, textvariable=disp,
                 bg=TAG_BG, fg=ACCENT,
                 font=("Segoe UI", 9, "bold")).pack()

        ttk.Scale(parent,
                  from_=from_, to=to,
                  variable=var,
                  orient="horizontal"
                  ).pack(fill="x", padx=14, pady=(2, 4))

    def _build_main_area(self, parent):
        main = tk.Frame(parent, bg=BG)
        main.pack(side="left", fill="both",
                  expand=True, padx=16, pady=16)

        panels_frame = tk.Frame(main, bg=BG)
        panels_frame.pack(fill="both", expand=True)

        self._panel_content = ImagePanel(
            panels_frame,
            "Content Image",
            "Upload a content image")
        self._panel_content.pack(side="left", fill="both",
                                 expand=True, padx=(0, 6))

        self._panel_style = ImagePanel(
            panels_frame,
            "Style Reference",
            "Select a style preset")
        self._panel_style.pack(side="left", fill="both",
                               expand=True, padx=6)

        self._panel_result = ImagePanel(
            panels_frame,
            "Generated Result",
            "Result will appear here")
        self._panel_result.pack(side="left", fill="both",
                                expand=True, padx=(6, 0))

        self._build_statusbar(main)

    def _build_statusbar(self, parent):
        bar = tk.Frame(parent, bg=WHITE,
                       height=STATUS_H,
                       highlightbackground=BORDER,
                       highlightthickness=1)
        bar.pack(fill="x", pady=(12, 0), side="bottom")
        bar.pack_propagate(False)

        inner = tk.Frame(bar, bg=WHITE)
        inner.pack(fill="both", expand=True,
                   padx=16, pady=0)

        self.status_dot = tk.Label(inner, text="●",
                                   bg=WHITE, fg=SUBTEXT,
                                   font=("Segoe UI", 11))
        self.status_dot.pack(side="left")

        self.status_lbl = tk.Label(inner, text="Ready",
                                   bg=WHITE, fg=SUBTEXT,
                                   font=("Segoe UI", 9))
        self.status_lbl.pack(side="left", padx=(6, 0))

        self.save_btn = tk.Button(
            inner,
            text="💾  Save Result",
            bg=TAG_BG, fg=ACCENT,
            font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2",
            activebackground="#DDE4FF",
            activeforeground=ACCENT2,
            bd=0, state="disabled",
            command=self._save_result)
        self.save_btn.pack(side="right",
                           ipadx=14, ipady=5)

        tk.Frame(inner, bg=BORDER, width=1).pack(
            side="right", fill="y", padx=12)

        self.pct_lbl = tk.Label(inner, text="0%",
                                 bg=WHITE, fg=ACCENT,
                                 font=("Segoe UI", 9, "bold"),
                                 width=5, anchor="e")
        self.pct_lbl.pack(side="right")

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            inner,
            variable=self.progress_var,
            maximum=100)
        self.progress_bar.pack(side="right", fill="x",
                               expand=True, padx=(0, 6))

    def _render_thumb(self, label, pil_img):
        label.update_idletasks()
        w = label.winfo_width()
        if w < 10:
            w = SIDEBAR_W - 28
        h = label.winfo_reqheight() or 148
        rendered = fit_image(pil_img, w, h)
        tk_img   = ImageTk.PhotoImage(rendered)
        label.configure(image=tk_img)
        label.image = tk_img

    def _upload_content(self):
        path = filedialog.askopenfilename(
            title="Select Content Image",
            filetypes=[("Image Files",
                        "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path:
            return
        self.content_path = path
        img = Image.open(path).convert("RGB")
        self._render_thumb(self._thumb_content, img)
        self._panel_content.set_image(img)
        self._set_status("Content image loaded.", SUCCESS)

    def _update_style_preview(self):
        name = self.selected_preset.get()
        path = STYLE_PRESETS.get(name)
        if path and os.path.exists(path):
            self.style_path = path
            img = Image.open(path).convert("RGB")
            self._render_thumb(self._thumb_style, img)
            self._panel_style.set_image(img)

    def _set_status(self, text, color=SUBTEXT):
        self.status_lbl.configure(text=text, fg=color)
        self.status_dot.configure(fg=color)

    def _start_transfer(self):
        if self.running:
            return
        if not self.content_path:
            messagebox.showwarning(
                "Warning",
                "Please upload a content image first.")
            return
        if not self.style_path or \
                not os.path.exists(self.style_path):
            messagebox.showwarning(
                "Warning",
                "Style image not found.\n"
                "Please check the style_images folder.")
            return

        self.running = True
        self.run_btn.configure(state="disabled",
                               text="⏳  Running...")
        self.save_btn.configure(state="disabled")
        self.progress_var.set(0)
        self.pct_lbl.configure(text="0%")
        self.result_image = None
        self._panel_result.clear()
        self._set_status("Initializing model...", ACCENT)

        alpha = round(self.alpha_var.get(), 2)
        steps = int(self.steps_var.get())
        size  = int(self.size_var.get())

        def callback(step, total, loss, pct, preview):
            self.root.after(
                0, lambda: self.progress_var.set(pct))
            self.root.after(
                0, lambda: self.pct_lbl.configure(
                    text=f"{pct}%"))
            self.root.after(
                0, lambda: self._set_status(
                    f"Step {step}/{total}   "
                    f"Loss: {loss:.2f}", ACCENT))
            if preview:
                self.root.after(
                    0,
                    lambda p=preview:
                    self._panel_result.set_image(p))

        def worker():
            try:
                result, save_path = run_transfer(
                    self.content_path, self.style_path,
                    alpha=alpha, size=size, steps=steps,
                    callback=callback)
                self.result_image = result
                self.root.after(
                    0,
                    lambda: self._on_done(result, save_path))
            except Exception as e:
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Error", str(e)))
                self.root.after(0, self._reset_btn)

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, result, save_path):
        self._panel_result.set_image(result)
        self._set_status(
            f"Done!  Auto-saved → {save_path}", SUCCESS)
        self.pct_lbl.configure(text="100%")
        self.save_btn.configure(state="normal")
        self._reset_btn()

    def _reset_btn(self):
        self.running = False
        self.run_btn.configure(
            state="normal",
            text="▶   Run Style Transfer")

    def _save_result(self):
        if not self.result_image:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG Image", "*.jpg"),
                       ("PNG Image",  "*.png")],
            title="Save Result Image")
        if path:
            self.result_image.save(path, quality=95)
            messagebox.showinfo(
                "Saved", f"Image saved to:\n{path}")


if __name__ == "__main__":
    root = tk.Tk()
    app  = StyleTransferApp(root)
    root.mainloop()
