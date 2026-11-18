#!/usr/bin/env python3
"""
Background Remover - a cross-platform (Windows / macOS / Linux) tool that cuts
the subject out of photos, in batches, using AI models that run on your own
computer. Saves transparent PNGs, or puts the subject on a solid colour.

The models are the ones the rembg project publishes (IS-Net and BiRefNet),
run directly with onnxruntime, the same way rembg runs them. Each model is
downloaded once, the first time it's used. Photos are never uploaded.

Requirements:
    Python 3.8+ (with Tkinter, included in the python.org installers)
    numpy, onnxruntime -> run the AI model
    Pillow, pillow-heif -> read and write images (pillow-heif opens iPhone HEIC photos)
    certifi            -> trusted certificates for downloading the model
    tkinterdnd2        -> drag and drop (optional: the app works without it)
    The app offers to install them on first run. Manual install, using the Python that runs the app:
        python -m pip install numpy onnxruntime pillow pillow-heif certifi tkinterdnd2

Run:
    python background_remover.py
"""

import hashlib
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

# Filled in by load_libraries(), so the window can still open (and offer to
# install them) when they're missing.
np = ort = Image = ImageOps = ImageTk = None

PACKAGES = ["numpy", "onnxruntime", "pillow", "pillow-heif", "certifi", "tkinterdnd2"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff"}
FILE_TYPES = [
    ("Images", " ".join("*" + e for e in sorted(IMAGE_EXTS))),
    ("All files", "*.*"),
]

# The models, as published by rembg (https://github.com/danielgatis/rembg).
# Each takes a 1024×1024 picture and returns how likely each pixel is to be
# part of the subject. 'mean'/'std' are the input normalisation it was trained with.
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
MODELS = {
    "Best quality — BiRefNet": dict(
        name="birefnet-general", file="BiRefNet-general-epoch_244.onnx",
        md5="7a35a0141cbbc80de11d9c9a28f52697", size=972666916, norm=IMAGENET, sigmoid=True, ram_gb=16,
        about="Cleanest edges on hair, fur and busy backgrounds. 973 MB download, about 15 s "
              "per photo, needs about 8 GB of memory."),
    "Faster — BiRefNet lite": dict(
        name="birefnet-general-lite", file="BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        md5="4fab47adc4ff364be1713e97b7e66334", size=224005088, norm=IMAGENET, sigmoid=True, ram_gb=12,
        about="Nearly as good as Best, about twice as fast. 224 MB download, about 8 s per photo, "
              "needs about 6 GB of memory."),
    "Fast, low memory — IS-Net": dict(
        name="isnet-general-use", file="isnet-general-use.onnx",
        md5="fc16ebd8b0c10d971d3513d564d01e29", size=178648008, norm=((0.5,) * 3, (1.0,) * 3),
        sigmoid=False, ram_gb=0,
        about="About 1 s per photo and under 1 GB of memory. Fine for simple photos (products, "
              "plain backgrounds); struggles with hair and cluttered scenes."),
}
FORMATS = ["JPG", "PNG"]  # for a colour background; transparent is always PNG
JPG_QUALITY = 92
MARGINS = ["0", "10", "20", "50", "100"]
SUBJECT_ALPHA = 8  # pixels more see-through than this don't count as subject when cropping
MASK_LEVELS = (3, 248)  # mask values at or below / above these become fully clear / fully solid


def load_libraries():
    """Import the image and AI libraries. Returns the pip names of missing
    packages (tkinterdnd2 is optional and checked separately)."""
    global np, ort, Image, ImageOps, ImageTk
    import importlib
    importlib.invalidate_caches()  # see packages installed since startup
    missing = []
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
    try:
        import onnxruntime as ort
    except ImportError:
        missing.append("onnxruntime")
    try:
        from PIL import Image, ImageOps, ImageTk
        Image.MAX_IMAGE_PIXELS = 250_000_000  # big panoramas are fine; refuse absurd ("bomb") files
    except ImportError:
        missing.append("pillow")
    try:
        import pillow_heif
        if Image:
            pillow_heif.register_heif_opener()
    except ImportError:
        missing.append("pillow-heif")
    try:
        import certifi  # noqa: F401
    except ImportError:
        missing.append("certifi")
    try:
        import tkinterdnd2  # noqa: F401
    except ImportError:
        missing.append("tkinterdnd2")
    return missing


# Hide the console window that would otherwise flash up on Windows.
_POPEN_FLAGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def install_requirements(packages):
    """pip-install the packages into *this* Python interpreter.

    Using sys.executable guarantees they land in the same Python that's running
    the app — the usual cause of 'I installed it but it's not found' is having
    several Pythons, where `pip` and double-clicking the script use different ones.
    """
    exe = sys.executable
    if exe.lower().endswith("pythonw.exe"):  # pythonw has no console; use python.exe for pip
        exe = exe[:-len("pythonw.exe")] + "python.exe"
    r = subprocess.run([exe, "-m", "pip", "install", "--upgrade"] + packages,
                       capture_output=True, text=True, errors="replace", **_POPEN_FLAGS)
    add_user_site()
    return r.returncode == 0, (r.stderr or r.stdout).strip().splitlines()[-3:]


def add_user_site():
    """Make packages pip just put in the per-user folder importable right away.

    When Python's own folder isn't writable (e.g. installed under Program
    Files), pip installs to the user folder instead (%APPDATA%\\Python\\...
    on Windows, ~/Library/Python/... on macOS). Python only adds that folder
    to sys.path at startup if it already existed, so on a first install it
    isn't searched until the app is restarted.
    """
    import site
    path = site.getusersitepackages() if site.ENABLE_USER_SITE else None  # off in venvs
    if path and os.path.isdir(path) and path not in sys.path:
        site.addsitedir(path)


def total_ram_gb():
    """Installed memory in GB, or None if it can't be read."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong) for n in ("total", "avail", "page_total", "page_avail",
                                                      "virt_total", "virt_avail", "ext_avail")]
            m = MemoryStatus()
            m.dwLength = ctypes.sizeof(m)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.total / 2 ** 30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  Models: where they live, downloading, running
# --------------------------------------------------------------------------- #
def models_home():
    """The folder rembg uses (~/.rembg), so a model downloaded by either is shared.
    REMBG_HOME / U2NET_HOME override it, the same way they do for rembg."""
    if os.getenv("U2NET_HOME"):
        return os.path.expanduser(os.getenv("U2NET_HOME"))
    xdg = os.getenv("XDG_DATA_HOME")
    return os.path.expanduser(os.getenv("REMBG_HOME", os.path.join(xdg, "rembg") if xdg else "~/.rembg"))


def model_path(cfg):
    """Where the model is (or will be) stored. Also finds one in rembg's older
    flat ~/.u2net folder, so nothing is downloaded twice."""
    path = os.path.join(models_home(), "models", cfg["name"], cfg["name"] + ".onnx")
    legacy = os.path.join(os.path.expanduser(os.getenv("U2NET_HOME", "~/.u2net")), cfg["name"] + ".onnx")
    return legacy if not os.path.exists(path) and os.path.exists(legacy) else path


def download_model(cfg, on_progress, check_cancel):
    """Download a model file, checking it against its published MD5.
    Written under a temporary name, so a cancelled or broken download is never used."""
    import ssl
    import urllib.request
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    dest = model_path(cfg)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    md5 = hashlib.md5()
    req = urllib.request.Request(MODEL_URL + cfg["file"], headers={"User-Agent": "Background-Remover"})
    try:
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or cfg["size"])
            done = 0
            while True:
                check_cancel()
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                md5.update(chunk)
                done += len(chunk)
                on_progress(done, total)
        if md5.hexdigest() != cfg["md5"]:
            raise DownloadError("The downloaded file is damaged (its checksum doesn't match). Try again.")
        os.replace(part, dest)
    finally:
        if os.path.exists(part):
            os.remove(part)


def load_session(cfg):
    """Load a model. The memory arena is turned off: it keeps the peak working
    memory of the big BiRefNet models about 1.5 GB lower, for a second or so
    of extra time per photo."""
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    return ort.InferenceSession(model_path(cfg), so, providers=["CPUExecutionProvider"])


def predict_mask(session, cfg, img):
    """The model's subject mask for `img`, as an 'L' image the size of `img`.

    Same steps as rembg: resize to 1024×1024, scale to 0–1, normalise, run,
    (sigmoid for BiRefNet), stretch to the full 0–255 range, resize back.
    """
    im = img.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
    a = np.asarray(im, dtype=np.float32)
    a = a / max(float(a.max()), 1e-6)
    mean, std = cfg["norm"]
    a = (a - np.array(mean, np.float32)) / np.array(std, np.float32)
    x = np.ascontiguousarray(a.transpose(2, 0, 1)[None], dtype=np.float32)
    out = session.run(None, {session.get_inputs()[0].name: x})[0][0, 0]
    if cfg["sigmoid"]:
        out = 1 / (1 + np.exp(-out))
    out = (out - out.min()) / max(float(out.max() - out.min()), 1e-6)
    mask = Image.fromarray((out * 255).astype(np.uint8), "L")
    return mask.resize(img.size, Image.Resampling.LANCZOS)


# --------------------------------------------------------------------------- #
#  Clean edges: foreground colour estimation
# --------------------------------------------------------------------------- #
def _box1d(a, r, axis):
    """Mean over a window of 2r+1 along one axis (shrinking at the edges), via a running sum."""
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    r = min(r, n - 1)
    c = np.empty((n + 1,) + a.shape[1:], np.float32)
    c[0] = 0
    np.cumsum(a, axis=0, out=c[1:])
    out = np.empty_like(a)
    w = 2 * r + 1
    if n > 2 * r:
        out[r:n - r] = (c[w:n + 1] - c[0:n - 2 * r]) / w
    for i in list(range(min(r, n))) + list(range(max(n - r, r), n)):  # the shrinking windows at both ends
        lo, hi = max(i - r, 0), min(i + r + 1, n)
        out[i] = (c[hi] - c[lo]) / (hi - lo)
    return np.moveaxis(out, 0, axis)


def _blur(a, r):
    return _box1d(_box1d(a, r, 0), r, 1)


def clean_edges(img, mask):
    """Estimate the subject's own colours where it's partly see-through (hair,
    fur, soft edges), so the old background doesn't show as a coloured fringe
    when it's put on a new one.

    "Approximate Fast Foreground Colour Estimation" (Forte & Pitié, 2021): a
    blurred-average estimate of foreground and background colours, refined
    twice. The wide first pass is done at quarter size (it's a heavy blur, so
    nothing is lost). The fine second pass only changes pixels that are partly
    see-through (fully opaque ones keep their colour, fully transparent ones
    aren't seen), so it runs only on the tiles that contain some, padded so
    the result is the same as doing the whole picture. One channel at a time,
    to keep memory down.
    """
    rgb = img.convert("RGB")
    small = (max(1, rgb.width // 4), max(1, rgb.height // 4))
    alpha = np.asarray(mask, np.float32) / 255
    alpha_s = np.asarray(mask.resize(small, Image.Resampling.BILINEAR), np.float32) / 255
    h, w = alpha.shape
    tile, r, pad = 256, 6, 8
    edge = (alpha > 0.002) & (alpha < 0.998)
    tiles = [(y, x) for y in range(0, h, tile) for x in range(0, w, tile) if edge[y:y + tile, x:x + tile].any()]
    up = lambda a: np.asarray(Image.fromarray(a.astype(np.float32), "F").resize(rgb.size, Image.Resampling.BILINEAR))  # noqa: E731
    channels = []
    for ch, ch_s in zip(rgb.split(), rgb.resize(small, Image.Resampling.BILINEAR).split()):
        image = np.asarray(ch, np.float32) / 255
        # Pass 1 (radius 90 px at full size = 22 px at quarter size).
        i_s = np.asarray(ch_s, np.float32) / 255
        ba = _blur(alpha_s, 22)
        f_s = _blur(i_s * alpha_s, 22) / (ba + 1e-5)
        b_s = _blur(i_s * (1 - alpha_s), 22) / ((1 - ba) + 1e-5)
        f_s = np.clip(f_s + alpha_s * (i_s - alpha_s * f_s - (1 - alpha_s) * b_s), 0, 1)
        fg, bg = up(f_s), up(b_s)
        # Pass 2 (radius 6 px), at full size, on the edge tiles only.
        out = image.copy()
        for y, x in tiles:
            y0, x0, y1, x1 = max(0, y - pad), max(0, x - pad), min(h, y + tile + pad), min(w, x + tile + pad)
            a, i_, f_, b_ = (arr[y0:y1, x0:x1] for arr in (alpha, image, fg, bg))
            ba = _blur(a, r)
            f = _blur(f_ * a, r) / (ba + 1e-5)
            b = _blur(b_ * (1 - a), r) / ((1 - ba) + 1e-5)
            f = np.clip(f + a * (i_ - a * f - (1 - a) * b), 0, 1)
            ys, xs = y - y0, x - x0
            out[y:y + tile, x:x + tile] = f[ys:ys + tile, xs:xs + tile]
        channels.append(Image.fromarray((out * 255 + 0.5).astype(np.uint8), "L"))
    return Image.merge("RGB", channels)


# --------------------------------------------------------------------------- #
#  Per-photo work (worker thread; no Tk calls in here)
# --------------------------------------------------------------------------- #
def open_photo(path):
    """Open a photo the right way up. Returns (image, colour profile or None)."""
    with Image.open(path) as im:
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
        im.load()
        im = ImageOps.exif_transpose(im)
    return im, im.info.get("icc_profile") if im.mode != "CMYK" else None


def subject_mask(session, cfg, img):
    """The model's mask, also keeping anything already transparent in the photo transparent.

    The raw mask almost never reaches exactly 0 or 255 (rounding and resizing),
    which would leave the subject 1–2% see-through and the background faintly
    there. Values within a few steps of either end are snapped to it, and the
    rest stretched to fill the range.
    """
    lo, hi = MASK_LEVELS
    mask = predict_mask(session, cfg, img).point(
        [0 if v <= lo else 255 if v >= hi else round((v - lo) * 255 / (hi - lo)) for v in range(256)])
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        from PIL import ImageChops
        mask = ImageChops.darker(mask, img.convert("RGBA").getchannel("A"))
    return mask


def render(img, mask, s):
    """The finished picture: the cut-out subject, cropped if asked, on
    transparency or on a colour. Returns (image, cropped_box or None)."""
    fg = clean_edges(img, mask)
    box = None
    if s["crop"]:
        box = mask.point(lambda v: 255 if v > SUBJECT_ALPHA else 0).getbbox()
        if box:
            m = s["margin"]
            box = (max(0, box[0] - m), max(0, box[1] - m), min(img.width, box[2] + m), min(img.height, box[3] + m))
            fg, mask = fg.crop(box), mask.crop(box)
    if s["colour"] is None:
        # Fully clear pixels are black: their colour is never seen, and the
        # leftover background colours would make the PNG several times bigger.
        fg.paste((0, 0, 0), mask=mask.point(lambda v: 255 if v == 0 else 0))
        fg.putalpha(mask)
        return fg, box
    out = Image.new("RGB", fg.size, s["colour"])
    out.paste(fg, mask=mask)
    return out, box


def encode(result, s, icc):
    import io
    buf = io.BytesIO()
    kw = {"icc_profile": icc} if icc else {}
    if s["colour"] is not None and s["fmt"] == "JPG":
        result.save(buf, "JPEG", quality=JPG_QUALITY, optimize=True, **kw)
    else:
        result.save(buf, "PNG", **kw)
    return buf.getvalue()


def probe_image(path):
    """Return a dict (width, height, size, error) for one image, from its header."""
    info = dict(width=None, height=None, size=None, error=None)
    try:
        info["size"] = os.path.getsize(path)
        with Image.open(path) as im:
            w, h = im.size
            if im.getexif().get(0x0112) in (5, 6, 7, 8):  # rotated 90°: shown the other way round
                w, h = h, w
            info.update(width=w, height=h)
    except Exception as e:
        info["error"] = explain_error(e)
    return info


def explain_error(e):
    """Turn an exception into a short reason."""
    name = type(e).__name__
    if name == "UnidentifiedImageError":
        return "Not an image, or the file is damaged."
    if name == "DecompressionBombError":
        return "Image is too large (over 250 megapixels)."
    if isinstance(e, PermissionError):
        return "Permission denied — the file may be open in another program."
    if isinstance(e, FileNotFoundError):
        return "File not found."
    if isinstance(e, MemoryError) or "bad allocation" in str(e) or "Failed to allocate" in str(e):
        return "Not enough memory. Try the Fast, low memory model, or close other programs."
    text = str(e)
    if "truncated" in text or "broken" in text.lower():
        return "The file is damaged or incomplete."
    return text or name


def fmt_size(n):
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{max(1, round(n / 1e3))} KB"


class Cancelled(Exception):
    pass


class DownloadError(Exception):
    pass


def enable_drag_and_drop(root, on_drop, on_enter=None, on_leave=None):
    """Accept files dragged onto the window, using tkinterdnd2 if it's installed.
    Returns False when it isn't (or its native part can't load): the Add
    buttons still work, so drag and drop is optional.

    on_drop gets the list of dropped paths. Every widget is registered, so a
    drop works anywhere in the window.
    """
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
        TkinterDnD.require(root)
    except Exception:
        return False

    def drop(event):
        # The paths come as a Tcl list: names with spaces are in {braces}.
        on_drop(list(root.tk.splitlist(event.data)))
        return event.action  # "copy": the files stay where they are

    def hook(fn):
        def handler(event):
            if fn:
                fn()
            return event.action
        return handler

    widgets, todo = [], list(root.winfo_children())
    while todo:
        w = todo.pop()
        widgets.append(w)
        todo += w.winfo_children()
    for w in widgets:
        w.drop_target_register(DND_FILES)
        w.dnd_bind("<<Drop>>", drop)
        w.dnd_bind("<<DropEnter>>", hook(on_enter))
        w.dnd_bind("<<DropLeave>>", hook(on_leave))
    return True


# --------------------------------------------------------------------------- #
#  GUI
# --------------------------------------------------------------------------- #
class RemoverApp:
    def __init__(self, root):
        self.root = root
        root.title("Background Remover")
        root.minsize(720, 0)
        self.missing = load_libraries()
        self.cancelled = False
        self.running = False
        self.items = {}  # tree row id -> dict(path, info)
        self.sessions = {}  # model name -> loaded onnxruntime session (loading takes seconds)
        self.masks = {}  # (path, model name) -> mask, so a previewed photo isn't redone
        self.ram = total_ram_gb()
        self.ui_queue = queue.Queue()
        self.probe_queue = queue.Queue()  # files waiting to be read, one background reader
        self._pump_ui()
        threading.Thread(target=self._probe_worker, daemon=True).start()
        # Show unexpected errors in a dialog. By default Tk prints them to a
        # console, which is invisible when the script is double-clicked.
        root.report_callback_exception = self._report_error
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        home = os.path.expanduser("~")
        pictures = os.path.join(home, "Pictures")
        # Start on the best model this computer has the memory for.
        default = next(label for label, cfg in MODELS.items()
                       if label != "Best quality — BiRefNet" and (self.ram or 0) >= cfg["ram_gb"])
        self.v = {
            "model": tk.StringVar(value=default),
            "background": tk.StringVar(value="transparent"),
            "colour": tk.StringVar(value="#ffffff"),
            "fmt": tk.StringVar(value="JPG"),
            "crop": tk.BooleanVar(value=False),
            "margin": tk.StringVar(value="20"),
            "where": tk.StringVar(value="same"),
            "out_dir": tk.StringVar(value=pictures if os.path.isdir(pictures) else home),
            "reveal": tk.BooleanVar(value=True),
        }
        self._build_ui()
        for name, fn in (("model", self._on_model), ("background", self._on_background),
                         ("crop", self._on_crop), ("where", self._on_where)):
            self.v[name].trace_add("write", lambda *_, fn=fn: fn())
            fn()
        self.dnd = enable_drag_and_drop(root, self._on_drop, self._on_drag_enter, self._update_list_label)
        self._update_list_label()

        if not self._required_missing():
            self.status.config(text="Ready.")
        else:
            self.status.config(text="Some components are missing.")
            # Defer so the main window is drawn before the dialog appears.
            self.root.after(200, self.offer_install)

    def _required_missing(self):
        return [m for m in self.missing if m != "tkinterdnd2"]

    # ---- dependency installer ------------------------------------------ #
    def offer_install(self):
        if getattr(sys, "frozen", False):
            # The downloadable app bundles everything, and sys.executable is the
            # app itself, so "pip install" would just launch another copy of it.
            messagebox.showerror("Missing components", "Missing: " + ", ".join(self.missing) +
                                 "\n\nThis copy of Background Remover is incomplete. Please download it again.")
            return
        names = ", ".join(self.missing)
        if not messagebox.askyesno(
                "Install components?",
                f"Background Remover needs: {names}\n\n"
                "Install now? (about 50 MB download)\n\n"
                f"This runs pip for the Python running this app:\n{sys.executable}"):
            self.status.config(text="Components missing — nothing can be done until they're installed.")
            return
        self._set_busy(True)
        self.cancel_btn.config(state="disabled")
        self.status.config(text="Installing… this can take a minute.")
        packages = list(self.missing)

        def work():
            try:
                ok, tail = install_requirements(packages)
            except Exception as e:
                ok, tail = False, [str(e)]
            self.ui(lambda: self._install_done(ok, tail))
        threading.Thread(target=work, daemon=True).start()

    def _install_done(self, ok, tail):
        self._set_busy(False)
        self.missing = load_libraries()
        if ok and not self._required_missing():
            if not self.dnd:
                self.dnd = enable_drag_and_drop(self.root, self._on_drop, self._on_drag_enter,
                                                self._update_list_label)
            self._update_list_label()
            self.status.config(text="Installed! Ready.")
            for iid, item in self.items.items():  # re-read files added before Pillow existed
                item["info"] = None
                self._set_row(iid, status="Reading…")
                self.probe_queue.put((iid, item["path"]))
        else:
            self.status.config(text="Install failed — see the error message.")
            messagebox.showerror(
                "Install failed",
                "Couldn't install automatically:\n\n" + "\n".join(tail) +
                "\n\nTry running this in a terminal:\n"
                f'"{sys.executable}" -m pip install ' + " ".join(PACKAGES))

    # ---- layout --------------------------------------------------------- #
    def _build_ui(self):
        pad = {"padx": 6, "pady": 3}
        app_header(self.root, "Background Remover", "Cut out the subject with AI on your own PC · free, private, no limits", "✂").pack(fill="x")
        main = ttk.Frame(self.root, padding=(12, 10))
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        f = ttk.LabelFrame(main, text="1. Photos", padding=8)
        f.grid(row=0, column=0, sticky="nsew", **pad)
        f.columnconfigure(0, weight=1)
        f.rowconfigure(1, weight=1)
        btns = ttk.Frame(f)
        btns.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.edit_btns = [
            ttk.Button(btns, text="Add files…", command=self.add_files),
            ttk.Button(btns, text="Add folder…", command=self.add_folder),
            ttk.Button(btns, text="Remove", command=self.remove_selected),
            ttk.Button(btns, text="Clear", command=self.clear),
        ]
        for i, b in enumerate(self.edit_btns):
            b.grid(row=0, column=i, padx=(0, 6))
        btns.columnconfigure(len(self.edit_btns), weight=1)
        self.preview_btn = ttk.Button(btns, text="Preview selected", command=self.preview)
        self.preview_btn.grid(row=0, column=len(self.edit_btns) + 1, sticky="e")
        self.edit_btns.append(self.preview_btn)
        cols = ("dims", "size", "status")
        self.tree = ttk.Treeview(f, columns=cols, height=8, selectmode="extended")
        for col, text, width, anchor in (("#0", "File", 230, "w"), ("dims", "Pixels", 90, "e"),
                                         ("size", "Size", 65, "e"), ("status", "Status", 300, "w")):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor=anchor, stretch=col in ("#0", "status"))
        self.tree.grid(row=1, column=0, sticky="nsew")
        self.tree.bind("<Double-1>", lambda e: self.preview())
        sb = ttk.Scrollbar(f, orient="vertical", command=self.tree.yview)
        sb.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=sb.set)
        self.list_lbl = ttk.Label(f, text="", foreground=MUTED)
        self.list_lbl.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        f = ttk.LabelFrame(main, text="2. AI model (runs on this computer)", padding=8)
        f.grid(row=1, column=0, sticky="ew", **pad)
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="Model").grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(f, values=list(MODELS), textvariable=self.v["model"], state="readonly",
                     width=30).grid(row=0, column=1, sticky="w", **pad)
        self.model_lbl = ttk.Label(f, text="", foreground=MUTED, wraplength=640)
        self.model_lbl.grid(row=1, column=0, columnspan=2, sticky="w", **pad)

        f = ttk.LabelFrame(main, text="3. Result", padding=8)
        f.grid(row=2, column=0, sticky="ew", **pad)
        ttk.Radiobutton(f, text="Transparent background (PNG)", value="transparent",
                        variable=self.v["background"]).grid(row=0, column=0, columnspan=4, sticky="w", **pad)
        ttk.Radiobutton(f, text="Colour background:", value="colour",
                        variable=self.v["background"]).grid(row=1, column=0, sticky="w", **pad)
        self.swatch = tk.Label(f, width=3, relief="solid", borderwidth=1, bg=self.v["colour"].get())
        self.swatch.grid(row=1, column=1, sticky="w", **pad)
        self.colour_btn = ttk.Button(f, text="Choose…", command=self.pick_colour)
        self.colour_btn.grid(row=1, column=2, sticky="w", **pad)
        self.fmt_lbl = ttk.Label(f, text="Save as")
        self.fmt_lbl.grid(row=1, column=3, sticky="e", **pad)
        self.fmt_cb = ttk.Combobox(f, values=FORMATS, textvariable=self.v["fmt"], state="readonly", width=5)
        self.fmt_cb.grid(row=1, column=4, sticky="w", **pad)
        ttk.Checkbutton(f, text="Crop to the subject, with a margin of", variable=self.v["crop"]).grid(
            row=2, column=0, columnspan=2, sticky="w", **pad)
        self.margin_cb = ttk.Combobox(f, values=MARGINS, textvariable=self.v["margin"], width=5)
        self.margin_cb.grid(row=2, column=2, sticky="w", **pad)
        ttk.Label(f, text="pixels").grid(row=2, column=3, sticky="w", **pad)

        f = ttk.LabelFrame(main, text="4. Save to", padding=8)
        f.grid(row=3, column=0, sticky="ew", **pad)
        f.columnconfigure(1, weight=1)
        ttk.Radiobutton(f, text="Same folder as each photo", value="same",
                        variable=self.v["where"]).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(f, text="This folder:", value="folder",
                        variable=self.v["where"]).grid(row=1, column=0, sticky="w")
        self.dir_e = ttk.Entry(f, textvariable=self.v["out_dir"])
        self.dir_e.grid(row=1, column=1, sticky="ew", padx=6)
        self.dir_btn = ttk.Button(f, text="Choose…", command=self.pick_folder)
        self.dir_btn.grid(row=1, column=2)
        ttk.Checkbutton(f, text="Show the results in their folder when done",
                        variable=self.v["reveal"]).grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

        f = ttk.Frame(main)
        f.grid(row=4, column=0, sticky="ew", **pad)
        f.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(f, mode="determinate", maximum=1000)
        self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.cancel_btn = ttk.Button(f, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_btn.grid(row=0, column=1)
        self.go_btn = PillButton(f, "Remove backgrounds", self.start)
        self.go_btn.grid(row=0, column=2, padx=(6, 0))
        foot = ttk.Frame(main)  # the status line, with the support link on its right
        foot.grid(row=5, column=0, sticky="ew", **pad)
        foot.columnconfigure(0, weight=1)
        self.status = ttk.Label(foot, text="Ready.", wraplength=530)
        self.status.grid(row=0, column=0, sticky="w")
        support_link(foot).grid(row=0, column=1, sticky="ne", padx=(12, 0))

    # ---- small helpers -------------------------------------------------- #
    def _on_model(self):
        label = self.v["model"].get()
        cfg = MODELS[label]
        text = cfg["about"]
        if self.ram and self.ram < cfg["ram_gb"]:
            text += f"\nThis computer has {self.ram:.0f} GB of memory, so this model may be very slow or fail."
        if not os.path.exists(model_path(cfg)):
            text += "  (Downloads the first time you use it.)"
        self.model_lbl.config(text=text, foreground=ERROR if self.ram and self.ram < cfg["ram_gb"] else MUTED)

    def _on_background(self):
        state = "normal" if self.v["background"].get() == "colour" else "disabled"
        self.colour_btn.config(state=state)
        self.fmt_lbl.config(state=state)
        self.fmt_cb.config(state="readonly" if state == "normal" else "disabled")

    def _on_crop(self):
        self.margin_cb.config(state="normal" if self.v["crop"].get() else "disabled")

    def _on_where(self):
        state = "normal" if self.v["where"].get() == "folder" else "disabled"
        self.dir_e.config(state=state)
        self.dir_btn.config(state=state)

    def pick_colour(self):
        rgb, hexcode = colorchooser.askcolor(color=self.v["colour"].get(), title="Background colour")
        if hexcode:
            self.v["colour"].set(hexcode)
            self.swatch.config(bg=hexcode)

    def pick_folder(self):
        d = filedialog.askdirectory(title="Choose output folder", initialdir=self.v["out_dir"].get())
        if d:
            self.v["out_dir"].set(d)

    def ui(self, fn):
        """Run fn on the main (UI) thread. Safe to call from any thread.

        Tkinter isn't thread-safe: touching widgets, or even calling
        root.after(), from a worker thread works on some systems and silently
        does nothing on others. So workers only put callbacks in a queue, and
        the UI thread drains it every 50 ms in _pump_ui().
        """
        self.ui_queue.put(fn)

    def _pump_ui(self):
        # Reschedule FIRST: if a queued callback raises, the pump must keep
        # running, or every later message (status, progress, errors) is lost.
        self.root.after(50, self._pump_ui)
        while True:
            try:
                fn = self.ui_queue.get_nowait()
            except queue.Empty:
                return
            try:
                fn()
            except Exception:
                self._report_error(*sys.exc_info())

    # ---- the file list -------------------------------------------------- #
    def add_files(self):
        self._add_paths(filedialog.askopenfilenames(title="Choose photos", filetypes=FILE_TYPES))

    def add_folder(self):
        d = filedialog.askdirectory(title="Choose a folder of photos")
        if d:
            self._on_drop([d])

    def _folder_images(self, d):
        return sorted(os.path.join(d, n) for n in os.listdir(d)
                      if os.path.splitext(n)[1].lower() in IMAGE_EXTS
                      and os.path.isfile(os.path.join(d, n)))

    def _on_drag_enter(self):
        if not self.running:
            self.list_lbl.config(text="Drop to add these photos.", foreground="")

    def _on_drop(self, paths):
        """Files or folders dropped or chosen (a folder adds the images directly inside it)."""
        self._update_list_label()
        if self.running:
            self.status.config(text="Wait for this batch to finish before adding more photos.")
            return
        found, skipped, empty = [], 0, []
        for p in paths:
            if os.path.isdir(p):
                imgs = self._folder_images(p)
                found += imgs
                if not imgs:
                    empty.append(os.path.basename(p) or p)
            elif os.path.splitext(p)[1].lower() in IMAGE_EXTS:
                found.append(p)
            else:
                skipped += 1
        self._add_paths(found)
        notes = []
        if skipped:
            notes.append(f"{skipped} file{'s' * (skipped != 1)} skipped (not JPG, PNG, WebP, HEIC, BMP or TIFF)")
        if empty:
            notes.append("no images directly inside " + ", ".join(empty) + " (subfolders aren't searched)")
        if notes:
            text = "; ".join(notes)
            self.status.config(text=text[0].upper() + text[1:] + ".")

    def _add_paths(self, paths):
        known = {os.path.normcase(os.path.abspath(i["path"])) for i in self.items.values()}
        for p in paths:
            key = os.path.normcase(os.path.abspath(p))
            if key in known:
                continue  # already in the list
            known.add(key)
            iid = self.tree.insert("", "end", text=os.path.basename(p),
                                   values=("", "", "Reading…" if Image else "Needs Pillow"))
            self.items[iid] = dict(path=p, info=None)
            if Image:
                self.probe_queue.put((iid, p))
        self._update_list_label()

    def _probe_worker(self):
        """Background thread: reads each added image's details, one at a time."""
        while True:
            iid, path = self.probe_queue.get()
            info = probe_image(path)
            self.ui(lambda i=iid, inf=info: self._probed(i, inf))

    def _probed(self, iid, info):
        if iid not in self.items:  # removed while it was being read
            return
        self.items[iid]["info"] = info
        if info["error"]:
            self._set_row(iid, status=info["error"])
        else:
            self._set_row(iid, dims=f"{info['width']}×{info['height']}", size=fmt_size(info["size"]),
                          status="Ready")
        self._update_list_label()

    def _set_row(self, iid, **cols):
        if self.tree.exists(iid):
            for col, val in cols.items():
                self.tree.set(iid, col, val)

    def remove_selected(self):
        for iid in self.tree.selection():
            self.tree.delete(iid)
            self.items.pop(iid, None)
        self._update_list_label()

    def clear(self):
        self.tree.delete(*self.tree.get_children())
        self.items.clear()
        self._update_list_label()

    def _update_list_label(self):
        n = len(self.items)
        if not n:
            text = ("Drag photos or folders here, or use Add files." if getattr(self, "dnd", False)
                    else "Add photos, or a folder (the images directly inside it).")
        else:
            ok = [i["info"] for i in self.items.values() if i["info"] and not i["info"]["error"]]
            text = f"{n} photo{'s' * (n != 1)}. Double-click one (or select it and click Preview) to check it first."
            if len(ok) < n and all(i["info"] for i in self.items.values()):
                text += f" {n - len(ok)} can't be opened."
        self.list_lbl.config(text=text, foreground=MUTED)

    # ---- settings ------------------------------------------------------- #
    def _read_settings(self, need_folder=True):
        """Validate the options. Returns a settings dict, or None after showing why not."""
        margin = 0
        if self.v["crop"].get():
            try:
                margin = int(self.v["margin"].get().strip())
                if margin < 0:
                    raise ValueError
            except ValueError:
                messagebox.showwarning("Check settings", "The margin must be a whole number of pixels "
                                       "(0 or more).")
                return None
        if need_folder and self.v["where"].get() == "folder" and not os.path.isdir(self.v["out_dir"].get().strip()):
            messagebox.showwarning("Check settings", "The output folder doesn't exist.")
            return None
        colour = self.v["colour"].get() if self.v["background"].get() == "colour" else None
        return dict(model=self.v["model"].get(), colour=colour, fmt=self.v["fmt"].get(),
                    crop=self.v["crop"].get(), margin=margin)

    def _plan_outputs(self, todo, s):
        """Each photo's output path. Never the photo itself, and never two photos
        writing the same file (e.g. dog.jpg and dog.png)."""
        ext = "jpg" if s["colour"] is not None and s["fmt"] == "JPG" else "png"
        tag = " (no background)" if s["colour"] is None else " (new background)"
        taken, plans = set(), []
        for iid, item in todo:
            src = item["path"]
            folder = (os.path.dirname(src) if self.v["where"].get() == "same"
                      else self.v["out_dir"].get().strip())
            stem = os.path.splitext(os.path.basename(src))[0]
            out = os.path.join(folder, f"{stem}{tag}.{ext}")
            base, n = os.path.splitext(out)[0], 2
            while os.path.normcase(os.path.abspath(out)) in taken:
                out = f"{base} ({n}).{ext}"
                n += 1
            taken.add(os.path.normcase(os.path.abspath(out)))
            plans.append((iid, out))
        return plans

    def _check_ready(self):
        if self._required_missing():
            self.offer_install()
            return False
        if not self.items:
            messagebox.showwarning("No photos", "Add some photos first.")
            return False
        if any(i["info"] is None for i in self.items.values()):
            messagebox.showinfo("Still reading photos", "Still reading the photos' details. "
                                "Try again in a moment.")
            return False
        return True

    def _confirm_model(self, label):
        """True if the model is on disk, or the user agreed to download it."""
        cfg = MODELS[label]
        if os.path.exists(model_path(cfg)):
            return True
        warn = ""
        if self.ram and self.ram < cfg["ram_gb"]:
            warn = (f"\n\nNote: this computer has {self.ram:.0f} GB of memory, and this model needs "
                    f"about {cfg['ram_gb'] // 2} GB free, so it may be very slow or fail.")
        return messagebox.askyesno(
            "Download the model?",
            f"{label} needs a one-time download of {cfg['size'] / 1e6:.0f} MB "
            f"from the rembg project on GitHub.\n\nAfter that it works offline. "
            f"Your photos are never uploaded.{warn}\n\nDownload now?")

    # ---- running -------------------------------------------------------- #
    def start(self):
        if not self._check_ready():
            return
        s = self._read_settings()
        if s is None or not self._confirm_model(s["model"]):
            return
        todo = [(iid, it) for iid, it in self.items.items() if not it["info"]["error"]]
        if not todo:
            messagebox.showwarning("Nothing to do", "None of these files can be opened as an image.")
            return
        plans = self._plan_outputs(todo, s)
        existing = [out for _, out in plans if os.path.exists(out)]
        skip = set()
        if existing:
            names = "\n".join(os.path.basename(p) for p in existing[:5])
            more = f"\n…and {len(existing) - 5} more" if len(existing) > 5 else ""
            answer = messagebox.askyesnocancel(
                "Files already exist",
                f"{len(existing)} of the output files already exist:\n\n{names}{more}\n\n"
                "Replace them?\n\nYes = replace   No = skip those photos   Cancel = stop")
            if answer is None:
                return
            if answer is False:
                skip = set(existing)
        jobs = [(iid, self.items[iid]["path"], out) for iid, out in plans if out not in skip]
        for iid, out in plans:
            self._set_row(iid, status="Skipped (exists)" if out in skip else "Waiting")
        if not jobs:
            self.status.config(text="Nothing to do: every output file already exists.")
            return
        self._launch(lambda: self._worker(jobs, s, len(skip)))

    def preview(self):
        if self.running or not self._check_ready():
            return
        sel = [i for i in self.tree.selection() if not self.items[i]["info"]["error"]]
        if not sel:
            messagebox.showinfo("Preview", "Select a photo in the list first (or double-click it).")
            return
        s = self._read_settings(need_folder=False)
        if s is None or not self._confirm_model(s["model"]):
            return
        iid = sel[0]
        self._launch(lambda: self._preview_worker(iid, self.items[iid]["path"], s))

    def _launch(self, fn):
        self.cancelled = False
        self._set_busy(True)
        # The model runs off the UI thread so the window stays responsive.
        threading.Thread(target=fn, daemon=True).start()

    def _set_busy(self, busy):
        self.running = busy
        self.go_btn.config(state="disabled" if busy else "normal")
        self.cancel_btn.config(state="normal" if busy else "disabled")
        for b in self.edit_btns:
            b.config(state="disabled" if busy else "normal")
        if busy:
            self.progress.config(value=0)

    def cancel(self):
        self.cancelled = True
        if self.running:
            self.status.config(text="Stopping after this photo…")

    def _check_cancel(self):
        if self.cancelled:
            raise Cancelled()

    def _session(self, label):
        """The loaded model, downloading it first if needed (worker thread)."""
        cfg = MODELS[label]
        if label in self.sessions:
            return self.sessions[label]
        if not os.path.exists(model_path(cfg)):
            def on_progress(done, total):
                self.ui(lambda: (self.progress.config(value=done / total * 1000),
                                 self.status.config(text=f"Downloading the model: {done / 1e6:.0f} of "
                                                         f"{total / 1e6:.0f} MB…")))
            download_model(cfg, on_progress, self._check_cancel)
            self.ui(self._on_model)  # no longer "downloads the first time"
        self.ui(lambda: (self.progress.config(value=0), self.status.config(text="Loading the model…")))
        self.sessions = {label: load_session(cfg)}  # keep one at a time: they're big
        return self.sessions[label]

    def _mask(self, label, path, img):
        """The subject mask, reusing one already made for this photo and model."""
        key = (os.path.normcase(os.path.abspath(path)), label)
        if key not in self.masks:
            mask = subject_mask(self._session(label), MODELS[label], img)
            if len(self.masks) >= 4:  # masks of big photos add up
                self.masks.pop(next(iter(self.masks)))
            self.masks[key] = mask
        return self.masks[key]

    def _process(self, path, s):
        """Cut out one photo. Returns (original, result, colour profile, seconds, crop box or None)."""
        t0 = time.monotonic()
        img, icc = open_photo(path)
        mask = self._mask(s["model"], path, img)
        self._check_cancel()
        result, box = render(img, mask, s)
        return img, result, icc, time.monotonic() - t0, box

    def _preview_worker(self, iid, path, s):
        try:
            self.ui(lambda: (self._set_row(iid, status="Previewing…"),
                             self.status.config(text=f"Making a preview of {os.path.basename(path)}…")))
            img, result, _, secs, _ = self._process(path, s)
            self.ui(lambda: (self._set_row(iid, status="Ready"), PreviewWindow(self, path, img, result, secs, s),
                             self.status.config(text=f"Preview ready ({secs:.1f} s). "
                                                     "Click Remove backgrounds to save.")))
        except Cancelled:
            self.ui(lambda: (self._set_row(iid, status="Ready"), self.status.config(text="Preview cancelled.")))
        except Exception as e:
            log_error(f"{path}\n{type(e).__name__}: {e}")
            self.ui(lambda e=e: (self._set_row(iid, status=f"Failed: {explain_error(e)}"),
                                 self.status.config(text=f"Preview failed: {explain_error(e)}")))
        self.ui(lambda: self._set_busy(False))

    def _worker(self, jobs, s, skipped):
        n = len(jobs)
        results = dict(done=0, failed=0, cancelled=0)
        first_out = None
        try:
            self._session(s["model"])
        except Cancelled:
            self.ui(lambda: self._finish_early("Cancelled."))
            return
        except Exception as e:
            log_error(f"Model: {type(e).__name__}: {e}")
            msg = (str(e) if isinstance(e, DownloadError) else
                   f"Couldn't download or load the model: {explain_error(e)}")
            if "urlopen" in repr(e) or "getaddrinfo" in str(e) or "timed out" in str(e):
                msg = "Couldn't download the model. Check your internet connection and try again."
            self.ui(lambda: self._finish_early(msg, error=True))
            return
        for k, (iid, src, out) in enumerate(jobs):
            if self.cancelled:
                self.ui(lambda i=iid: self._set_row(i, status="Cancelled"))
                results["cancelled"] += 1
                continue
            self.ui(lambda i=iid, k=k: (self._set_row(i, status="Working…"),
                                        self.status.config(text=f"Photo {k + 1} of {n}: {os.path.basename(src)}")))
            try:
                _, result, icc, secs, box = self._process(src, s)
                data = encode(result, s, icc)
                part = out + ".part"
                try:
                    with open(part, "wb") as f:
                        f.write(data)
                    os.replace(part, out)  # a cancel or crash never leaves half a file
                finally:
                    if os.path.exists(part):
                        os.remove(part)
                status = f"Done in {secs:.1f} s — {fmt_size(len(data))}"
                if box:
                    status += f", cropped to {result.width}×{result.height}"
                results["done"] += 1
                first_out = first_out or out
            except Cancelled:
                status = "Cancelled"
                results["cancelled"] += 1
            except Exception as e:  # keep going with the other photos
                log_error(f"{src}\n{type(e).__name__}: {e}")
                status = f"Failed: {explain_error(e)}"
                results["failed"] += 1
            self.ui(lambda i=iid, st=status, k=k: (self._set_row(i, status=st),
                                                   self.progress.config(value=(k + 1) / n * 1000)))
        self.ui(lambda: self._finish(results, skipped, first_out))

    def _finish_early(self, text, error=False):
        self._set_busy(False)
        for iid in self.tree.get_children():
            if self.tree.set(iid, "status") == "Waiting":
                self._set_row(iid, status="Ready")
        self.status.config(text=text)
        if error:
            messagebox.showerror("Model problem", text)

    def _finish(self, r, skipped, first_out):
        self._set_busy(False)
        self.progress.config(value=1000 if r["done"] else 0)
        parts = [f"{r['done']} done"]
        if skipped:
            parts.append(f"{skipped} skipped (already existed)")
        if r["failed"]:
            parts.append(f"{r['failed']} failed — see the list")
        if r["cancelled"]:
            parts.append(f"{r['cancelled']} cancelled")
        self.status.config(text=("Done! " if not (r["failed"] or r["cancelled"]) else "") +
                           ", ".join(parts) + ".")
        if first_out and self.v["reveal"].get() and not self.cancelled:
            reveal_file(first_out)

    def _report_error(self, exc, val, tb):
        import traceback
        self._set_busy(False)
        detail = "".join(traceback.format_exception(exc, val, tb))
        log_error(detail)
        self.status.config(text=f"Unexpected error: {val}")
        messagebox.showerror("Unexpected error", detail[-1500:])

    def on_close(self):
        self.cancel()
        self.root.destroy()


class PreviewWindow:
    """Before and after, side by side. Transparent areas show as a checkerboard."""
    SIDE = 460  # each picture fits in a square this size

    def __init__(self, app, path, before, after, secs, s):
        self.win = tk.Toplevel(app.root)
        self.win.title(f"Preview — {os.path.basename(path)}")
        self.win.transient(app.root)
        frame = ttk.Frame(self.win, padding=10)
        frame.pack(fill="both", expand=True)
        self.photos = []  # keep references, or Tk shows blank images
        for col, (title, im) in enumerate((("Before", before), ("After", after))):
            ttk.Label(frame, text=title, font=("", 10, "bold")).grid(row=0, column=col, pady=(0, 4))
            shown = self._fit(im)
            self.photos.append(ImageTk.PhotoImage(shown))
            tk.Label(frame, image=self.photos[-1], borderwidth=0).grid(row=1, column=col, padx=6)
        what = "transparent background" if s["colour"] is None else f"background {s['colour']}"
        ttk.Label(frame, text=f"{s['model']}, {what}, {secs:.1f} s. If edges look wrong, "
                              "try a different model.", foreground=MUTED,
                  wraplength=2 * self.SIDE).grid(row=2, column=0, columnspan=2, pady=(8, 0))
        ttk.Button(frame, text="Close", command=self.win.destroy).grid(row=3, column=0, columnspan=2, pady=(8, 0))
        self.win.bind("<Escape>", lambda e: self.win.destroy())
        # Centre it over the main window (a new window can otherwise open on
        # another monitor, or off screen).
        self.win.withdraw()
        self.win.update_idletasks()
        root = app.root
        w, h = self.win.winfo_reqwidth(), self.win.winfo_reqheight()
        x = root.winfo_rootx() + (root.winfo_width() - w) // 2
        y = max(root.winfo_rooty(), root.winfo_rooty() + (root.winfo_height() - h) // 2)
        self.win.geometry(f"+{max(0, x)}+{max(0, y)}")
        self.win.deiconify()
        self.win.lift()
        self.win.focus_set()

    def _fit(self, im):
        im = im.copy()
        im.thumbnail((self.SIDE, self.SIDE), Image.Resampling.LANCZOS)
        if im.mode != "RGBA":
            return im.convert("RGB")
        # Checkerboard behind transparency, like image editors show it.
        board = Image.new("RGB", im.size, (255, 255, 255))
        cell = 10
        dark = Image.new("RGB", (cell, cell), (204, 204, 204))
        for y in range(0, im.height, cell):
            for x in range((y // cell) % 2 * cell, im.width, 2 * cell):
                board.paste(dark, (x, y))
        board.paste(im, mask=im.getchannel("A"))
        return board


LOG_PATH = os.path.join(tempfile.gettempdir(), "background_remover_errors.log")


def log_error(text):
    """Append an error to a log file. Errors can otherwise vanish when the
    script runs without a console (double-clicked / pythonw)."""
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n{text}\n")
    except OSError:
        pass


def reveal_file(path):
    """Open the file's folder with the file selected."""
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(path)])
    except Exception:
        pass


def self_test():
    """Used by the release build: exit code 0 if the packaged app can download
    the Fast model, cut out a test picture, and load drag and drop."""
    try:
        if load_libraries():
            return 1
        cfg = MODELS["Fast, low memory — IS-Net"]
        if not os.path.exists(model_path(cfg)):
            download_model(cfg, lambda d, t: None, lambda: None)
        # A dark disc on a light, slightly noisy background.
        rng = np.random.default_rng(0)
        a = (220 + rng.integers(-10, 10, (480, 640, 3))).astype(np.uint8)
        yy, xx = np.mgrid[:480, :640]
        a[(yy - 240) ** 2 + (xx - 320) ** 2 < 120 ** 2] = (40, 60, 160)
        img = Image.fromarray(a, "RGB")
        mask = predict_mask(load_session(cfg), cfg, img)
        result, _ = render(img, mask, dict(colour=None, crop=True, margin=0))
        if not (mask.getpixel((320, 240)) > 200 and mask.getpixel((20, 20)) < 50 and result.mode == "RGBA"):
            return 2
        root = tk.Tk()
        root.withdraw()
        from tkinterdnd2 import TkinterDnD
        TkinterDnD.require(root)
        root.destroy()
        return 0
    except Exception as e:
        log_error(f"Self-test failed: {type(e).__name__}: {e}")
        return 3


SUPPORT_URL = "https://ko-fi.com/collinbrooksdev"


def support_link(parent, bg=None, font=None):
    """A small "Support this project" link that opens the Ko-fi page in the browser."""
    import webbrowser
    kw = {"bg": bg} if bg else {}
    link = tk.Label(parent, text="\u2665 Support this project", fg="#c8364f", cursor="hand2",
                    font=font or ("", 9, "underline"), **kw)
    link.bind("<Button-1>", lambda e: webbrowser.open(SUPPORT_URL))
    return link


# --------------------------------------------------------------------------- #
#  The look (shared by the Offline Toolkit apps): white cards, a dark purple
#  header fading to blue, blue actions. Only colours and fonts: no behaviour.
# --------------------------------------------------------------------------- #
THEME = dict(
    bg="#ffffff", card="#ffffff", border="#e2ddf0", text="#1c1433", muted="#6e6787", error="#c8364f",
    purple="#2b1655", purple2="#4a2a8a", blue="#2f7cf6", blue_hover="#1d67dc", blue_soft="#e7f0ff",
    blue_text="#1756c4", lavender_line="#7b5cd6", disabled="#c9c4d8", tint="#f5f1ff")
MUTED, ERROR = THEME["muted"], THEME["error"]


def ui_family():
    try:
        import tkinter.font as tkfont
        families = set(tkfont.families())
    except tk.TclError:
        return "TkDefaultFont"
    for name in ("Segoe UI", "SF Pro Text", "Helvetica Neue", "Inter", "Cantarell", "DejaVu Sans"):
        if name in families:
            return name
    return "TkDefaultFont"


def _indicator_image(root, kind, on, enabled, n=18, k=4):
    """A checkbox (a blue box with a white tick) or radio button (a blue ring and dot),
    drawn pixel by pixel with smooth edges (no image library needed)."""
    t = THEME
    blue = t["blue"] if enabled else t["disabled"]
    edge = "#b9b0d6" if enabled else t["disabled"]

    def rgb(h):
        return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))

    def seg_dist(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        u = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        return ((px - ax - u * dx) ** 2 + (py - ay - u * dy) ** 2) ** 0.5

    def in_round_rect(x, y, r=4.6, lo=0.9, hi=n - 0.9):
        cx, cy = min(max(x, lo + r), hi - r), min(max(y, lo + r), hi - r)
        return (x - cx) ** 2 + (y - cy) ** 2 <= r * r

    def colour_at(x, y):
        """The colour of one sample point, or None where it's see-through."""
        c = n / 2
        d = ((x - c) ** 2 + (y - c) ** 2) ** 0.5
        if kind == "check":
            if not in_round_rect(x, y):
                return None
            if on:
                tick = min(seg_dist(x, y, 4.8, 9.4, 7.8, 12.4), seg_dist(x, y, 7.8, 12.4, 13.4, 5.9))
                return "#ffffff" if tick <= 1.15 else blue
            return edge if not in_round_rect(x, y, 3.4, 2.3, n - 2.3) else "#ffffff"
        if d > n / 2 - 0.9:
            return None
        if on:
            return blue if d > n / 2 - 2.5 or d < 3.6 else "#ffffff"
        return edge if d > n / 2 - 2.3 else "#ffffff"

    img = tk.PhotoImage(master=root, width=n, height=n)
    for py in range(n):
        row = []
        for px in range(n):
            acc, hits = [0, 0, 0], 0
            for sy in range(k):
                for sx in range(k):
                    col = colour_at(px + (sx + 0.5) / k, py + (sy + 0.5) / k)
                    if col is not None:
                        r, g, b = rgb(col)
                        acc[0] += r
                        acc[1] += g
                        acc[2] += b
                        hits += 1
            if hits == 0:
                row.append(None)
                continue
            # Blend with white for partly covered edge pixels (the widgets sit on white).
            a = hits / (k * k)
            r, g, b = (round(acc[i] / hits * a + 255 * (1 - a)) for i in range(3))
            row.append("#%02x%02x%02x" % (r, g, b))
        for px, col in enumerate(row):
            if col is not None:
                img.put(col, (px, py))
    return img


def apply_theme(root):
    """ttk's "clam" theme, recoloured (the native Windows and macOS themes ignore colours)."""
    import tkinter.font as tkfont
    t, fam = THEME, ui_family()
    root.configure(bg=t["bg"])
    root.option_add("*Background", t["bg"])  # plain Tk widgets (pictures, links) sit on white too
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
        try:
            tkfont.nametofont(name).configure(family=fam, size=10)
        except tk.TclError:
            pass
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=t["bg"], foreground=t["text"], font=(fam, 10), bordercolor=t["border"],
                focuscolor=t["blue"], lightcolor=t["card"], darkcolor=t["border"], troughcolor="#ece8f7")
    s.map(".", foreground=[("disabled", t["disabled"])])
    s.map("TLabel", background=[("disabled", t["card"])])  # clam greys a disabled label's background
    s.configure("TLabelframe", background=t["card"], bordercolor=t["border"], lightcolor=t["border"],
                darkcolor=t["border"], relief="solid", borderwidth=1)
    s.configure("TLabelframe.Label", background=t["card"], foreground=t["purple2"], font=(fam, 11, "bold"))
    for kind in ("TCheckbutton", "TRadiobutton"):
        s.configure(kind, background=t["card"], foreground=t["text"])
        s.map(kind, background=[("active", t["card"])], foreground=[("disabled", t["disabled"])])
    s.configure("TCombobox", fieldbackground="#ffffff", background="#ffffff", foreground=t["text"],
                arrowcolor=t["purple2"], bordercolor=t["border"], padding=(6, 3), selectbackground="#ffffff",
                selectforeground=t["text"])
    s.map("TCombobox", fieldbackground=[("readonly", "#ffffff"), ("disabled", "#f6f4fb")],
          foreground=[("disabled", t["disabled"])], arrowcolor=[("disabled", t["disabled"])],
          bordercolor=[("focus", t["blue"]), ("hover", t["lavender_line"])])
    s.configure("TEntry", fieldbackground="#ffffff", bordercolor=t["border"], padding=(6, 3))
    s.map("TEntry", bordercolor=[("focus", t["blue"])], fieldbackground=[("disabled", "#f6f4fb")])
    s.configure("TSpinbox", fieldbackground="#ffffff", bordercolor=t["border"], arrowcolor=t["purple2"], padding=(6, 3))
    s.configure("TButton", background="#ffffff", foreground=t["purple2"], bordercolor=t["border"],
                padding=(12, 5), font=(fam, 10, "bold"), relief="flat")
    s.map("TButton", background=[("disabled", t["card"]), ("pressed", t["tint"]), ("active", "#f7f4ff")],
          foreground=[("disabled", t["disabled"])], bordercolor=[("active", t["lavender_line"])])
    s.configure("Treeview", background="#ffffff", fieldbackground="#ffffff", foreground=t["text"],
                rowheight=26, font=(fam, 10))
    s.map("Treeview", background=[("selected", t["blue_soft"])], foreground=[("selected", t["blue_text"])])
    s.configure("Treeview.Heading", background="#f6f4fb", foreground=t["muted"], font=(fam, 9, "bold"),
                relief="flat", padding=(6, 4))
    s.map("Treeview.Heading", background=[("active", "#efeaf9")])
    s.configure("Horizontal.TProgressbar", troughcolor="#ece8f7", background=t["blue"], bordercolor="#ece8f7",
                lightcolor=t["blue"], darkcolor=t["blue"])
    for orient in ("Vertical", "Horizontal"):
        s.configure(f"{orient}.TScrollbar", background="#ddd6ee", troughcolor="#f6f4fb", bordercolor="#f6f4fb",
                    arrowcolor=t["muted"], lightcolor="#ddd6ee", darkcolor="#ddd6ee", gripcount=0)
    s.configure("Horizontal.TScale", background=t["blue"], troughcolor="#ece8f7", bordercolor=t["border"])
    root.option_add("*TCombobox*Listbox.background", "#ffffff")
    root.option_add("*TCombobox*Listbox.foreground", t["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", t["blue"])
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
    # Checkboxes and radio buttons: drawn ticks and dots (clam draws a tick as an X).
    imgs = root._indicator_images = {(kind, on, en): _indicator_image(root, kind, on, en)
                                     for kind in ("check", "radio") for on in (False, True) for en in (False, True)}
    for kind, widget in (("check", "Checkbutton"), ("radio", "Radiobutton")):
        name = f"Toolkit.{kind}.indicator"
        try:
            s.element_create(name, "image", imgs[(kind, False, True)],
                             ("disabled", "selected", imgs[(kind, True, False)]),
                             ("disabled", imgs[(kind, False, False)]),
                             ("selected", imgs[(kind, True, True)]), padding=(0, 0, 6, 0), sticky="")
        except tk.TclError:
            pass
        s.layout(f"T{widget}", [(f"{widget}.padding", {"sticky": "nswe", "children": [
            (name, {"side": "left", "sticky": ""}),
            (f"{widget}.focus", {"side": "left", "sticky": "w", "children": [
                (f"{widget}.label", {"sticky": "nswe"})]})]})])
    return fam


class PillButton(tk.Canvas):
    """A rounded blue button for the app's main action. Accepts .config(state=..., text=...)
    like a ttk.Button, so the rest of the app drives it the same way."""

    def __init__(self, parent, text, command, height=36, padx=24):
        import tkinter.font as tkfont
        self._font = tkfont.Font(family=ui_family(), size=11, weight="bold")
        self._padx = padx
        self._bh = height
        self._bw = self._font.measure(text) + 2 * padx
        super().__init__(parent, width=self._bw, height=height, bg=THEME["card"], highlightthickness=0, bd=0,
                         cursor="hand2")
        self._text, self._command, self._enabled, self._hover = text, command, True, False
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<ButtonRelease-1>", self._click)
        self._draw()

    def _draw(self):
        self.delete("all")
        t = THEME
        fill = t["disabled"] if not self._enabled else t["blue_hover"] if self._hover else t["blue"]
        r, w, h = self._bh // 2, self._bw, self._bh
        pts = [1 + r, 1, w - 1 - r, 1, w - 1, 1, w - 1, 1 + r, w - 1, h - 1 - r, w - 1, h - 1, w - 1 - r, h - 1,
               1 + r, h - 1, 1, h - 1, 1, h - 1 - r, 1, 1 + r, 1, 1]
        self.create_polygon(pts, smooth=True, fill=fill, outline=fill)
        self.create_text(w / 2, h / 2, text=self._text, fill="#ffffff", font=self._font)

    def _set_hover(self, on):
        self._hover = on
        self._draw()

    def _click(self, event):
        if self._enabled and 0 <= event.x <= self._bw and 0 <= event.y <= self._bh:
            self._command()

    def configure(self, cnf=None, **kw):
        if "state" in kw:
            self._enabled = kw.pop("state") != "disabled"
            super().configure(cursor="hand2" if self._enabled else "")
        if "text" in kw:
            self._text = kw.pop("text")
            self._bw = self._font.measure(self._text) + 2 * self._padx
            super().configure(width=self._bw)
        if "command" in kw:
            self._command = kw.pop("command")
        if cnf or kw:
            super().configure(cnf, **kw)
        self._draw()

    config = configure

    def cget(self, key):
        if key == "state":
            return "normal" if self._enabled else "disabled"
        if key == "text":
            return self._text
        return super().cget(key)

    def invoke(self):
        if self._enabled:
            self._command()


def app_header(parent, title, tagline, logo):
    """The app's bar: dark purple fading to blue, with its logo, name and what it does."""
    t, fam = THEME, ui_family()
    c = tk.Canvas(parent, height=60, highlightthickness=0, bd=0, bg=t["purple"])

    def draw(event=None):
        c.delete("all")
        w = max(c.winfo_width(), 300)
        a, b = (0x2b, 0x16, 0x55), (0x2f, 0x5c, 0xd6)
        steps = 48
        for i in range(steps):
            k = (i / (steps - 1)) ** 2.2  # mostly purple, turning blue on the right
            col = "#%02x%02x%02x" % tuple(round(a[j] + (b[j] - a[j]) * k) for j in range(3))
            c.create_rectangle(i * w / steps, 0, (i + 1) * w / steps + 1, 60, fill=col, outline=col)
        if not hasattr(c, "_logo"):  # the app's own icon
            c._logo = tk.PhotoImage(master=c, data=HEADER_ICON)
        c.create_image(35, 30, image=c._logo)
        c.create_text(68, 21, text=title, anchor="w", fill="#ffffff", font=(fam, 15, "bold"))
        c.create_text(69, 42, text=tagline, anchor="w", fill="#cfc6ff", font=(fam, 9))
    c.bind("<Configure>", draw)
    return c


# The app's icon (64 px PNG), for the window and the taskbar. The .exe and the Mac app get
# theirs from assets/ when they're built.
APP_ICON = (
    "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"
    "vwAAAABJRU5ErkJggg=="
)

# The same icon, small, for the app's header bar.
HEADER_ICON = (
    "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"
)


def set_app_icon(root):
    try:
        if sys.platform == "win32":
            # Its own taskbar button and icon, not Python's, when run as a script.
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("OfflineToolkit.BackgroundRemover")
        root._app_icon = tk.PhotoImage(master=root, data=APP_ICON)
        root.iconphoto(True, root._app_icon)
    except Exception:
        pass  # just the default icon


def main():
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    root = tk.Tk()
    apply_theme(root)
    set_app_icon(root)
    RemoverApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
