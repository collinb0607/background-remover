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
        main = ttk.Frame(self.root, padding=10)
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
        self.list_lbl = ttk.Label(f, text="", foreground="gray")
        self.list_lbl.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        f = ttk.LabelFrame(main, text="2. AI model (runs on this computer)", padding=8)
        f.grid(row=1, column=0, sticky="ew", **pad)
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="Model").grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(f, values=list(MODELS), textvariable=self.v["model"], state="readonly",
                     width=30).grid(row=0, column=1, sticky="w", **pad)
        self.model_lbl = ttk.Label(f, text="", foreground="gray", wraplength=640)
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
        self.go_btn = ttk.Button(f, text="Remove backgrounds", command=self.start)
        self.go_btn.grid(row=0, column=2, padx=(6, 0))
        self.status = ttk.Label(main, text="Ready.", wraplength=680)
        self.status.grid(row=5, column=0, sticky="w", **pad)

    # ---- small helpers -------------------------------------------------- #
    def _on_model(self):
        label = self.v["model"].get()
        cfg = MODELS[label]
        text = cfg["about"]
        if self.ram and self.ram < cfg["ram_gb"]:
            text += f"\nThis computer has {self.ram:.0f} GB of memory, so this model may be very slow or fail."
        if not os.path.exists(model_path(cfg)):
            text += "  (Downloads the first time you use it.)"
        self.model_lbl.config(text=text, foreground="red" if self.ram and self.ram < cfg["ram_gb"] else "gray")

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
        self.list_lbl.config(text=text, foreground="gray")

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
                              "try a different model.", foreground="gray",
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


def main():
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    root = tk.Tk()
    try:  # nicer native look where available
        ttk.Style().theme_use("vista" if sys.platform == "win32" else
                              "aqua" if sys.platform == "darwin" else "clam")
    except tk.TclError:
        pass
    RemoverApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
