"""ModusMate live GUI.

Features:
* Connect to KitProg3 serial port, send SET_ALGO / LCD / stream toggles.
* Live preview canvas with FPS, algo and inference timing.
* Per-algorithm description panel.
* Per-frame statistic (edge pixels / keypoints / on-pixels / mean intensity)
  computed locally from the preview frame.
* Benchmark dialog: pick algorithm subset and limit, run the RPS benchmark
  in a worker thread, watch progress live, save CSV / Markdown summary.
"""
from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

from PIL import Image, ImageTk

from . import benchmark as bench
from . import protocol as P
from .algos import (ALGO_NAMES, compute_preview_stats, description,
                    family_name, family_of, stat_kind)
from .link import BoardLink, Detection, FpsReport, list_serial_ports


# Per-family RGB colour for previews where the gray buffer encodes a
# response-magnitude image (edges, ridges, keypoints heatmap) so each
# algo family looks visually distinct rather than "yet another gray".
_FAMILY_COLOR: dict = {
    "edges":        (255,  60,  40),   # bright red
    "ridge":        ( 60, 220, 255),   # cyan
    "blobs":        ( 80, 255, 120),   # green
    "keypoints":    (255, 220,  60),   # yellow (drawn as dots, see below)
    "texture":      (255, 130,  40),   # orange
    "thresholding": (255, 255, 255),   # white-on-black (binary mask)
    "morphology":   (200, 160, 255),   # violet
}


def _gray_to_color_overlay(gray: bytes, w: int, h: int,
                           rgb: tuple) -> Image.Image:
    """Build an RGB image where bright pixels (response peaks / detected
    keypoints) glow in the family colour and the rest of the frame is
    rendered in plain grayscale.  This matters for algorithms whose
    firmware buffer is "dimmed scene + bright markers" (the keypoint
    detectors AGAST/BRIEF/AKAZE/FAST-12 emit ``gray>>1`` plus 255 dots):
    multiplying every pixel by the family RGB just yellow-washes the
    whole image, hiding the actual detections.  Instead we pick a
    threshold so dim background stays gray, and only strong responses
    take on the family colour, scaled by their intensity."""
    L = Image.frombytes("L", (w, h), gray)
    # Background channels = grayscale of L (so dim regions stay gray).
    # Foreground tinted = L scaled by colour where v >= THRESH; below
    # threshold, fall back to gray.  We pick THRESH=160 because the
    # firmware markers write 255, edges/blobs response peaks reach
    # 200+, while the dimmed scene from keypoint algos sits at 0..127.
    THRESH = 160
    rm, gm, bm = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    r = L.point(lambda v, m=rm: int(v if v < THRESH else v * m))
    g = L.point(lambda v, m=gm: int(v if v < THRESH else v * m))
    b = L.point(lambda v, m=bm: int(v if v < THRESH else v * m))
    return Image.merge("RGB", (r, g, b))


def _gray_preview_to_rgb(gray: bytes, w: int, h: int,
                         algo_name: str) -> Image.Image:
    """Convert an 80x60 grayscale preview to a 320x240 RGB image, applying
    a family-specific colormap so each family reads distinctively:
    edges = red, ridges = cyan, blobs = green, keypoints = yellow dots,
    texture = orange, binary masks = pure white-on-black, morphology =
    violet.  Basics keep their natural grayscale rendering."""
    fam = family_of(algo_name)
    colour = _FAMILY_COLOR.get(fam)
    if colour is not None:
        img = _gray_to_color_overlay(gray, w, h, colour)
    else:
        # basics / unknown: keep grayscale
        img = Image.frombytes("L", (w, h), gray).convert("RGB")
    # BILINEAR (instead of NEAREST) gives a smoother live preview when the
    # 80x60 firmware buffer is upscaled to 320x240; the eye reads it as a
    # less "jiggly" stream when consecutive frames differ slightly.
    return img.resize((320, 240), Image.BILINEAR)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("ModusMate — PSoC Edge image-processing console")
        root.geometry("1024x640")

        self.link: Optional[BoardLink] = None
        self._algos: List[str] = ALGO_NAMES[:]
        self._preview_img: Optional[ImageTk.PhotoImage] = None
        self._preview_item: Optional[int] = None    # persistent canvas item id
        self._cur_algo_name: str = ALGO_NAMES[0]
        self._fps_seen = False
        self._stream_warning_shown = False
        # Frame coalescing: drop preview frames if Tk hasn't drained the
        # previous one yet, so the canvas always shows the freshest image
        # instead of catching up through a backlog (which is what makes
        # the stream look "jiggly").
        self._pending_preview: Optional[Image.Image] = None
        self._pending_stat: tuple = ("", "")
        self._render_scheduled = False

        self._build_ui()
        self._refresh_ports()
        self._refresh_description()

    # ---------------- UI layout ----------------
    def _build_ui(self) -> None:
        # Top: connection bar
        top = ttk.Frame(self.root, padding=8); top.pack(fill="x")
        ttk.Label(top, text="Port:").grid(row=0, column=0, sticky="w")
        self.cmb_port = ttk.Combobox(top, width=24, state="readonly")
        self.cmb_port.grid(row=0, column=1, sticky="w", padx=4)
        ttk.Button(top, text="↻", width=3,
                   command=self._refresh_ports).grid(row=0, column=2, padx=2)
        self.btn_conn = ttk.Button(top, text="Connect",
                                   command=self._toggle_connect)
        self.btn_conn.grid(row=0, column=3, padx=8)
        self.lbl_conn = ttk.Label(top, text="disconnected", foreground="gray")
        self.lbl_conn.grid(row=0, column=4, sticky="w", padx=4)
        self.btn_bench = ttk.Button(top, text="Run benchmark…",
                                    command=self._open_benchmark_dialog)
        self.btn_bench.grid(row=0, column=5, padx=8)

        # Algo + flags row
        mid = ttk.Frame(self.root, padding=8); mid.pack(fill="x")
        ttk.Label(mid, text="Algorithm:").grid(row=0, column=0, sticky="w")
        self.cmb_algo = ttk.Combobox(mid, width=28, state="readonly",
                                     values=self._algos)
        self.cmb_algo.grid(row=0, column=1, sticky="w", padx=4)
        self.cmb_algo.current(0)
        self.cmb_algo.bind("<<ComboboxSelected>>",
                           lambda *_: self._on_algo_change())

        # LCD defaults OFF: when the host is streaming preview to the PC
        # the on-board LCD just steals camera/inference cycles. The user
        # can re-enable it explicitly.
        self.var_lcd = tk.BooleanVar(value=False)
        # Stream preview defaults ON so the canvas shows live frames
        # immediately after connect.
        self.var_stream = tk.BooleanVar(value=True)
        ttk.Checkbutton(mid, text="LCD on board", variable=self.var_lcd,
                        command=self._send_lcd).grid(row=0, column=2, padx=10)
        ttk.Checkbutton(mid, text="Stream preview to PC",
                        variable=self.var_stream,
                        command=self._send_stream).grid(row=0, column=3, padx=10)

        # Telemetry row
        tele = ttk.Frame(self.root, padding=8); tele.pack(fill="x")
        self.lbl_fps = ttk.Label(tele, text="FPS: --",
                                 font=("Helvetica", 14, "bold"))
        self.lbl_fps.grid(row=0, column=0, padx=8, sticky="w")
        self.lbl_times = ttk.Label(
            tele, text="algo: -- µs   inference: -- µs")
        self.lbl_times.grid(row=0, column=1, padx=8, sticky="w")
        self.lbl_stat = ttk.Label(tele, text="—  : --",
                                  foreground="dark green",
                                  font=("Helvetica", 12, "bold"))
        self.lbl_stat.grid(row=0, column=2, padx=8, sticky="w")
        # Detection label is hidden by default — the firmware ships with
        # a generic NN now (shapes/silhouettes/etc.), so the old hardcoded
        # ROCK / PAPER / SCISSORS readout is misleading. We still keep
        # the widget around for log diagnostics but don't pack it into
        # the layout.
        self.lbl_det = ttk.Label(tele, text="", foreground="gray")

        # Body: preview canvas (left) + description + log (right)
        body = ttk.Frame(self.root, padding=8); body.pack(fill="both",
                                                         expand=True)
        left = ttk.Frame(body); left.pack(side="left", fill="y")
        self.canvas = tk.Canvas(left, width=320, height=240, bg="black")
        self.canvas.pack()
        self.lbl_family = ttk.Label(left, text="family: —",
                                    foreground="gray")
        self.lbl_family.pack(anchor="w", pady=(6, 0))
        self.lbl_kind = ttk.Label(left, text="stat: —", foreground="gray")
        self.lbl_kind.pack(anchor="w")

        right = ttk.Frame(body); right.pack(side="left", fill="both",
                                            expand=True, padx=8)
        ttk.Label(right, text="Description",
                  font=("Helvetica", 10, "bold")).pack(anchor="w")
        self.txt_desc = tk.Text(right, height=6, wrap="word", state="disabled",
                                background="#f8f8f8", relief="flat")
        self.txt_desc.pack(fill="x", pady=(0, 6))
        ttk.Label(right, text="Log",
                  font=("Helvetica", 10, "bold")).pack(anchor="w")
        self.txt_log = tk.Text(right, height=18, width=56, state="disabled")
        self.txt_log.pack(fill="both", expand=True)

    # ---------------- helpers ----------------
    def _log(self, msg: str) -> None:
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    def _set_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _refresh_ports(self) -> None:
        ports = list_serial_ports()
        labels = [f"{dev}  ({desc})" for dev, desc in ports]
        self.cmb_port["values"] = labels
        self._port_devs = [dev for dev, _ in ports]
        if labels:
            self.cmb_port.current(0)

    def _refresh_description(self) -> None:
        name = self._cur_algo_name
        self._set_text(self.txt_desc, f"{name}\n\n{description(name)}")
        kind = stat_kind(name)
        self.lbl_kind.config(text=f"stat: {kind}")

    # ---------------- connect ----------------
    def _toggle_connect(self) -> None:
        if self.link is None:
            idx = self.cmb_port.current()
            if idx < 0 or not self._port_devs:
                messagebox.showwarning("ModusMate", "No serial port selected.")
                return
            dev = self._port_devs[idx]
            # Try the canonical 115200 first, then fall back to common
            # upgrade rates in case a previous bench left the firmware at
            # a higher baud and the host was killed before it could send
            # CMD_SET_BAUDRATE(115200) on cleanup.  We always renegotiate
            # back to 115200 so the live console runs at a single known
            # rate.
            link: Optional[BoardLink] = None
            last_err: Optional[Exception] = None
            for cand in (115200, 1000000, 921600, 460800, 2000000, 3000000):
                try:
                    cand_link = BoardLink(dev, baudrate=cand)
                    cand_link.start()
                    if cand_link.ping(timeout=1.5):
                        if cand != 115200:
                            try:
                                cand_link.set_baudrate(115200)
                                self._log(
                                    f"recovered link from {cand}→115200")
                            except Exception:
                                pass
                        link = cand_link
                        break
                    cand_link.close()
                except Exception as e:
                    last_err = e
            if link is None:
                msg = f"no response from {dev}"
                if last_err is not None:
                    msg += f" ({last_err})"
                messagebox.showerror("ModusMate", msg)
                return
            link.on_fps(self._on_fps)
            link.on_detections(self._on_detections)
            link.on_log(self._log)
            link.on_preview(self._on_preview)
            self.link = link
            self.lbl_conn.config(text=f"connected: {dev}", foreground="green")
            self.btn_conn.config(text="Disconnect")
            self._log(f"opened {dev}")
            threading.Thread(target=self._initial_handshake,
                             daemon=True).start()
        else:
            try: self.link.close()
            except Exception: pass
            self.link = None
            self.lbl_conn.config(text="disconnected", foreground="gray")
            self.btn_conn.config(text="Connect")

    def _initial_handshake(self) -> None:
        assert self.link is not None
        ok = self.link.ping(timeout=2.0)
        self._log(f"ping: {'ok' if ok else 'no response'}")
        # Try to upgrade the live link from 115200 -> 1 Mbaud so the
        # board can push preview frames quickly enough to look smooth
        # (115200 caps a 4800-byte frame at ~24 Hz; 1 Mbaud lifts that
        # well past the camera's frame rate).  If the upgrade fails,
        # fall back silently and keep streaming at the lower rate.
        try:
            if self.link.set_baudrate(1_000_000):
                self._log("link baud upgraded 115200 -> 1000000")
            else:
                self._log("baud upgrade refused; staying at 115200")
        except Exception as e:
            self._log(f"baud upgrade failed: {e}")
        infos = self.link.get_info(timeout=2.0)
        if infos:
            self._algos = [i.name for i in infos]
            fams = {i.name: i.family for i in infos}
            self._algo_families = fams

            def _apply():
                self.cmb_algo.configure(values=self._algos)
                if self._algos:
                    self.cmb_algo.current(0)
                    self._cur_algo_name = self._algos[0]
                    self._refresh_description()
                    fid = fams.get(self._cur_algo_name, -1)
                    self.lbl_family.config(
                        text=f"family: {family_name(fid)}")
            self.root.after(0, _apply)
            self._log(f"firmware advertises {len(infos)} algorithms")
        # push current toggle state
        self._send_algo(); self._send_lcd(); self._send_stream()

    # ---------------- senders ----------------
    def _on_algo_change(self) -> None:
        idx = self.cmb_algo.current()
        if 0 <= idx < len(self._algos):
            self._cur_algo_name = self._algos[idx]
        self._refresh_description()
        fams = getattr(self, "_algo_families", {})
        fid = fams.get(self._cur_algo_name, -1)
        if fid >= 0:
            self.lbl_family.config(text=f"family: {family_name(fid)}")
        self._send_algo()

    def _send_algo(self) -> None:
        if not self.link: return
        idx = self.cmb_algo.current()
        if idx < 0: return
        threading.Thread(target=lambda: self.link.set_algo(idx),
                         daemon=True).start()

    def _send_lcd(self) -> None:
        if not self.link: return
        v = self.var_lcd.get()
        threading.Thread(target=lambda: self.link.set_lcd(v),
                         daemon=True).start()

    def _send_stream(self) -> None:
        if not self.link: return
        v = self.var_stream.get()
        threading.Thread(target=lambda: self.link.set_stream(v),
                         daemon=True).start()
        if v:
            # no FPS events for ~3 s after stream-on => no camera frames flowing
            self._fps_seen = False
            self._stream_warning_shown = False
            self.root.after(3000, self._check_stream_alive)

    def _check_stream_alive(self) -> None:
        if (self.link is not None
                and self.var_stream.get()
                and not self._fps_seen
                and not self._stream_warning_shown):
            self._stream_warning_shown = True
            self._log("WARNING: stream enabled but no frames received from "
                      "the board. Is a USB camera connected to the board's "
                      "USB host port? The inference loop only runs when the "
                      "camera produces frames.")

    # ---------------- callbacks (called from reader thread) ----------------
    def _on_fps(self, r: FpsReport) -> None:
        self._fps_seen = True
        self.root.after(0, lambda: (
            self.lbl_fps.config(text=f"FPS: {r.fps:5.1f}"),
            self.lbl_times.config(
                text=f"algo: {r.algo_us:6d} µs   "
                     f"inference: {r.infer_us:6d} µs"),
        ))

    def _on_detections(self, dets: List[Detection]) -> None:
        # Detection display intentionally suppressed: the firmware no
        # longer ships a fixed Rock/Paper/Scissors classifier so showing
        # those labels for arbitrary models would be wrong. The raw
        # detections still flow through bench/log paths if needed.
        return

    def _on_preview(self, data: bytes, w: int, h: int) -> None:
        if w * h != len(data):
            return
        # local stats from the preview frame
        label, value = compute_preview_stats(self._cur_algo_name, data, w, h)
        # Build the PIL image on this (reader) thread, but defer the
        # ImageTk.PhotoImage construction to the Tk main thread - PhotoImage
        # touches the Tk interpreter and is NOT thread-safe.
        try:
            img = _gray_preview_to_rgb(data, w, h, self._cur_algo_name)
        except Exception:
            return

        # Frame coalescing: keep only the freshest frame. If the Tk main
        # loop hasn't drawn the previous one yet, just overwrite
        # _pending_preview - this drops backlog and stops the stream
        # from looking jittery when frames arrive faster than 60 Hz.
        self._pending_preview = img
        self._pending_stat = (label, value)
        if not self._render_scheduled:
            self._render_scheduled = True
            self.root.after(0, self._drain_preview)

    def _drain_preview(self) -> None:
        self._render_scheduled = False
        img = self._pending_preview
        if img is None:
            return
        self._pending_preview = None
        try:
            photo = ImageTk.PhotoImage(img)
        except Exception:
            return
        self._preview_img = photo  # keep a ref so it's not GC'd
        # Reuse the same canvas item across frames (itemconfig) instead
        # of delete+create, which removes a flush step and removes the
        # brief black flash between frames.
        if self._preview_item is None:
            self._preview_item = self.canvas.create_image(
                0, 0, anchor="nw", image=self._preview_img)
        else:
            self.canvas.itemconfig(self._preview_item,
                                   image=self._preview_img)
        label, value = self._pending_stat
        if label:
            self.lbl_stat.config(text=f"{label}: {value}")
        else:
            self.lbl_stat.config(text="—")

    # ---------------- benchmark dialog ----------------
    def _open_benchmark_dialog(self) -> None:
        idx = self.cmb_port.current()
        if idx < 0 or not self._port_devs:
            messagebox.showwarning("ModusMate", "No serial port selected.")
            return
        port = self._port_devs[idx]
        if self.link is not None:
            messagebox.showinfo(
                "ModusMate",
                "Disconnect from the live link first — the benchmark needs "
                "exclusive access to the serial port.")
            return
        BenchmarkDialog(self.root, port=port,
                        algo_choices=self._algos,
                        log=self._log)


# ===================================================================
# Benchmark dialog (modal)
# ===================================================================
class BenchmarkDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, *, port: str,
                 algo_choices: List[str], log) -> None:
        super().__init__(parent)
        self.title("Run benchmark")
        self.geometry("780x900")
        self.transient(parent)
        self._port = port
        self._algo_choices = algo_choices
        self._log_main = log
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._summary: Optional[bench.BenchSummary] = None

        self._build()

    def _build(self) -> None:
        # Right-hand pane: live preview canvases.  Pack FIRST with
        # side="right" so it claims a fixed-width vertical strip on the
        # right edge of the dialog; everything else then fills the
        # remaining left area.
        right = ttk.Frame(self, padding=8)
        right.pack(side="right", fill="y")
        ttk.Label(right, text="Input (host)",
                  font=("Helvetica", 10, "bold")).pack(anchor="w")
        self.cv_input = tk.Canvas(right, width=320, height=240, bg="black")
        self.cv_input.pack(pady=(0, 8))
        ttk.Label(right, text="Board (post-algo)",
                  font=("Helvetica", 10, "bold")).pack(anchor="w")
        self.cv_board = tk.Canvas(right, width=320, height=240, bg="black")
        self.cv_board.pack()
        # Keep refs so Tk doesn't GC the PhotoImages.
        self._photo_input: Optional[ImageTk.PhotoImage] = None
        self._photo_board: Optional[ImageTk.PhotoImage] = None
        # Persistent canvas image items - reused via itemconfig() each
        # frame to avoid the delete+create flush that causes a brief
        # black flash between frames.
        self._item_input: Optional[int] = None
        self._item_board: Optional[int] = None

        cfg = ttk.Frame(self, padding=8); cfg.pack(fill="x")
        ttk.Label(cfg, text=f"Port: {self._port}").grid(row=0, column=0,
                                                       columnspan=4,
                                                       sticky="w")
        ttk.Label(cfg, text="Algorithms:").grid(row=1, column=0, sticky="nw",
                                                pady=(6, 0))
        # Treeview grouped by family.  Each leaf row is checkable via the
        # selection model (extended) so users can multi-select within and
        # across families; family rows act as quick "toggle whole group"
        # affordances via the side-bar buttons.
        tree_frame = ttk.Frame(cfg)
        tree_frame.grid(row=1, column=1, columnspan=2, sticky="ewns",
                        pady=(6, 0))
        cfg.columnconfigure(1, weight=1)
        # NOTE: do NOT set rowconfigure(1, weight=1) here - the tree must
        # take its natural height (height=12) so the rows below (limit,
        # CSV, summary, baud, preview-checkbox) and the Start/Stop/Close
        # buttons stay visible.  Letting the tree grow pushes the
        # control bar off the bottom of the dialog.
        self.tree_algos = ttk.Treeview(
            tree_frame, columns=("kind",), show="tree headings",
            selectmode="extended", height=12)
        self.tree_algos.heading("#0", text="algorithm")
        self.tree_algos.heading("kind", text="output")
        self.tree_algos.column("#0", width=200, anchor="w")
        self.tree_algos.column("kind", width=90, anchor="w")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                            command=self.tree_algos.yview)
        self.tree_algos.configure(yscrollcommand=vsb.set)
        self.tree_algos.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Map iid -> algo name for leaves; family iid -> list of leaf iids.
        self._tree_leaf_iid: dict = {}      # algo_name -> iid
        self._tree_iid_to_idx: dict = {}    # iid -> index in self._algo_choices
        self._tree_family_kids: dict = {}   # family iid -> list of leaf iids

        # Group choices by family, sort families in a sensible reading order.
        _family_order = ["basics", "edges", "ridge", "blobs", "keypoints",
                         "thresholding", "texture", "morphology", "other"]
        groups: dict = {}
        for idx, n in enumerate(self._algo_choices):
            groups.setdefault(family_of(n), []).append((idx, n))
        for fam in _family_order:
            entries = groups.get(fam)
            if not entries:
                continue
            fam_iid = self.tree_algos.insert(
                "", "end", text=f"▸ {fam}  ({len(entries)})",
                values=("",), open=True, tags=("family",))
            kids = []
            for idx, n in entries:
                leaf_iid = self.tree_algos.insert(
                    fam_iid, "end", text=n,
                    values=(stat_kind(n),), tags=("leaf",))
                self._tree_leaf_iid[n] = leaf_iid
                self._tree_iid_to_idx[leaf_iid] = idx
                kids.append(leaf_iid)
            self._tree_family_kids[fam_iid] = kids
        self.tree_algos.tag_configure("family",
                                      font=("Helvetica", 11, "bold"))
        # Clicking a family row selects all its kids.
        self.tree_algos.bind("<<TreeviewSelect>>", self._on_tree_select)

        bf = ttk.Frame(cfg); bf.grid(row=1, column=3, sticky="n",
                                    padx=6, pady=(6, 0))
        ttk.Button(bf, text="All",
                   command=self._tree_select_all).pack(fill="x")
        ttk.Button(bf, text="None",
                   command=self._tree_select_none).pack(fill="x",
                                                        pady=(4, 0))
        ttk.Separator(bf, orient="horizontal").pack(fill="x", pady=4)
        # Quick-pick by family (matches the colour groups in the preview).
        for fam in ["basics", "edges", "ridge", "blobs", "keypoints",
                    "thresholding", "texture", "morphology"]:
            if any(family_of(n) == fam for n in self._algo_choices):
                ttk.Button(bf, text=fam,
                           command=lambda f=fam:
                               self._tree_select_family(f)).pack(
                                   fill="x", pady=(2, 0))

        ttk.Label(cfg, text="Limit (images/algo):").grid(row=2, column=0,
                                                       sticky="w",
                                                       pady=(8, 0))
        self.var_limit = tk.IntVar(value=1)
        ttk.Spinbox(cfg, from_=1, to=5000, textvariable=self.var_limit,
                    width=10).grid(row=2, column=1, sticky="w",
                                   padx=(0, 8), pady=(8, 0))

        ttk.Label(cfg, text="CSV out:").grid(row=3, column=0, sticky="w",
                                            pady=(4, 0))
        self.var_csv = tk.StringVar(value="results.csv")
        ttk.Entry(cfg, textvariable=self.var_csv).grid(row=3, column=1,
                                                     sticky="ew",
                                                     pady=(4, 0))
        ttk.Button(cfg, text="…", width=3,
                   command=self._pick_csv).grid(row=3, column=2,
                                              sticky="w", pady=(4, 0))

        ttk.Label(cfg, text="Summary out:").grid(row=4, column=0, sticky="w",
                                                pady=(4, 0))
        self.var_md = tk.StringVar(value="summary.md")
        ttk.Entry(cfg, textvariable=self.var_md).grid(row=4, column=1,
                                                    sticky="ew",
                                                    pady=(4, 0))
        ttk.Button(cfg, text="…", width=3,
                   command=self._pick_md).grid(row=4, column=2,
                                             sticky="w", pady=(4, 0))

        ttk.Label(cfg, text="Link baud:").grid(row=5, column=0, sticky="w",
                                              pady=(4, 0))
        self.var_baud = tk.StringVar(value="1000000 (recommended)")
        baud_choices = ["115200 (safe / slow)", "460800", "921600",
                        "1000000 (recommended)", "2000000", "3000000 (max)"]
        ttk.Combobox(cfg, textvariable=self.var_baud, values=baud_choices,
                     width=18, state="readonly").grid(
            row=5, column=1, sticky="w", pady=(4, 0))
        ttk.Label(cfg, text="(higher = faster image push, KitProg3 max ~3 Mbaud)",
                  foreground="gray").grid(row=5, column=2, columnspan=2,
                                          sticky="w", pady=(4, 0))

        self.var_preview = tk.BooleanVar(value=True)
        ttk.Checkbutton(cfg, text="Show board post-algo preview "
                                  "(uncheck for faster bench)",
                        variable=self.var_preview).grid(
            row=6, column=0, columnspan=4, sticky="w", pady=(6, 0))

        # Optional dump of per-image (input, processed) PNG pairs.
        # Useful when running without an NN to inspect what each algo
        # produces; the firmware's post-algo preview becomes the "out"
        # image. Streaming is force-enabled by run_benchmark when this
        # is set, regardless of the checkbox above.
        self.var_save_proc = tk.BooleanVar(value=False)
        ttk.Checkbutton(cfg, text="Save processed images per algo to:",
                        variable=self.var_save_proc).grid(
            row=7, column=0, sticky="w", pady=(4, 0))
        self.var_save_dir = tk.StringVar(value="runs/processed")
        ttk.Entry(cfg, textvariable=self.var_save_dir).grid(
            row=7, column=1, sticky="ew", pady=(4, 0))
        ttk.Button(cfg, text="…", width=3,
                   command=self._pick_save_dir).grid(
            row=7, column=2, sticky="w", pady=(4, 0))

        # Benchmark mode: UART (upload images) vs Camera (fullscreen display)
        ttk.Separator(cfg, orient="horizontal").grid(
            row=8, column=0, columnspan=4, sticky="ew", pady=(8, 4))
        self.var_mode = tk.StringVar(value="uart")
        mode_frame = ttk.Frame(cfg)
        mode_frame.grid(row=9, column=0, columnspan=4, sticky="w", pady=(0, 4))
        ttk.Label(mode_frame, text="Mode:").pack(side="left")
        ttk.Radiobutton(mode_frame, text="UART (upload images)",
                        variable=self.var_mode, value="uart",
                        command=self._on_mode_change).pack(side="left", padx=(8, 4))
        ttk.Radiobutton(mode_frame, text="Camera (fullscreen display)",
                        variable=self.var_mode, value="camera",
                        command=self._on_mode_change).pack(side="left", padx=4)

        # Camera mode settings
        self._cam_frame = ttk.Frame(cfg)
        self._cam_frame.grid(row=10, column=0, columnspan=4, sticky="w")
        ttk.Label(self._cam_frame, text="Stabilize (ms):").grid(
            row=0, column=0, sticky="w")
        self.var_stabilize = tk.IntVar(value=500)
        ttk.Spinbox(self._cam_frame, from_=100, to=5000,
                    textvariable=self.var_stabilize, width=8).grid(
            row=0, column=1, sticky="w", padx=(4, 16))
        ttk.Label(self._cam_frame, text="Capture (ms):").grid(
            row=0, column=2, sticky="w")
        self.var_capture = tk.IntVar(value=2000)
        ttk.Spinbox(self._cam_frame, from_=500, to=10000,
                    textvariable=self.var_capture, width=8).grid(
            row=0, column=3, sticky="w", padx=(4, 16))
        ttk.Label(self._cam_frame, text="Conf threshold:").grid(
            row=0, column=4, sticky="w")
        self.var_conf_thresh = tk.DoubleVar(value=0.30)
        ttk.Spinbox(self._cam_frame, from_=0.05, to=0.99, increment=0.05,
                    textvariable=self.var_conf_thresh, width=6).grid(
            row=0, column=5, sticky="w", padx=4)
        # Initially hidden (UART mode is default)
        self._cam_frame.grid_remove()

        # Bottom widgets in the LEFT pane: log + Start/Stop/Close +
        # progress bar.  Packed with side="bottom" so Tk reserves their
        # space first; the algo tree (cfg, packed earlier with fill="x")
        # then takes the remaining vertical space above them.

        # log
        self.txt = tk.Text(self, height=10, state="disabled")
        self.txt.pack(side="bottom", fill="both", expand=True,
                      padx=8, pady=(0, 8))
        ttk.Label(self, text="Progress log",
                  font=("Helvetica", 10, "bold")).pack(
                      side="bottom", anchor="w", padx=8)

        # control buttons
        ctrl = ttk.Frame(self, padding=8)
        ctrl.pack(side="bottom", fill="x")
        self.btn_run = ttk.Button(ctrl, text="Start", command=self._start)
        self.btn_run.pack(side="left")
        self.btn_stop = ttk.Button(ctrl, text="Stop", command=self._stop,
                                   state="disabled")
        self.btn_stop.pack(side="left", padx=8)
        ttk.Button(ctrl, text="Close",
                   command=self._on_close).pack(side="right")

        # progress bar + status
        prog = ttk.Frame(self, padding=8); prog.pack(side="bottom", fill="x")
        self.lbl_status = ttk.Label(prog, text="idle")
        self.lbl_status.pack(anchor="w")
        self.pb = ttk.Progressbar(prog, mode="determinate")
        self.pb.pack(fill="x", pady=(4, 0))

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -------- selection helpers
    def _on_tree_select(self, _evt) -> None:
        # If the user clicked a family row, expand selection to its kids.
        sel = list(self.tree_algos.selection())
        added = False
        for iid in sel:
            if iid in self._tree_family_kids:
                kids = self._tree_family_kids[iid]
                # Replace family iid with all its leaves.
                self.tree_algos.selection_remove(iid)
                for k in kids:
                    self.tree_algos.selection_add(k)
                added = True
        if added:
            return

    def _tree_select_all(self) -> None:
        leaves = list(self._tree_iid_to_idx.keys())
        self.tree_algos.selection_set(leaves)

    def _tree_select_none(self) -> None:
        self.tree_algos.selection_set([])

    def _tree_select_family(self, fam: str) -> None:
        leaves = [self._tree_leaf_iid[n] for n in self._algo_choices
                  if family_of(n) == fam and n in self._tree_leaf_iid]
        self.tree_algos.selection_set(leaves)

    def _pick_csv(self) -> None:
        p = filedialog.asksaveasfilename(defaultextension=".csv",
                                         initialfile=self.var_csv.get())
        if p: self.var_csv.set(p)

    def _pick_md(self) -> None:
        p = filedialog.asksaveasfilename(defaultextension=".md",
                                         initialfile=self.var_md.get())
        if p: self.var_md.set(p)

    def _pick_save_dir(self) -> None:
        p = filedialog.askdirectory(initialdir=self.var_save_dir.get() or ".")
        if p:
            self.var_save_dir.set(p)
            self.var_save_proc.set(True)

    def _on_mode_change(self) -> None:
        if self.var_mode.get() == "camera":
            self._cam_frame.grid()
        else:
            self._cam_frame.grid_remove()

    # -------- log helpers
    def _log(self, msg: str) -> None:
        self.txt.configure(state="normal")
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    # -------- start / stop
    def _start(self) -> None:
        sel = self.tree_algos.selection()
        algo_ids: List[int] = []
        for iid in sel:
            if iid in self._tree_iid_to_idx:
                algo_ids.append(self._tree_iid_to_idx[iid])
        if not algo_ids:
            messagebox.showwarning("ModusMate",
                                   "Select at least one algorithm.")
            return
        algo_ids.sort()
        limit = max(1, int(self.var_limit.get()))
        csv_p = Path(self.var_csv.get()) if self.var_csv.get() else None
        md_p = Path(self.var_md.get()) if self.var_md.get() else None
        baud_str = self.var_baud.get().split()[0]
        try:
            baud_val = int(baud_str)
        except ValueError:
            baud_val = 115200
        self._upgrade_baud = baud_val if baud_val != 115200 else None
        self._enable_preview = bool(self.var_preview.get())
        self._save_proc_dir: Optional[Path] = None
        if self.var_save_proc.get():
            sd = self.var_save_dir.get().strip()
            if sd:
                self._save_proc_dir = Path(sd)
        if not self._enable_preview:
            # Clear any leftover preview from a previous run.
            self.cv_board.delete("all")
            self._photo_board = None
            self._item_board = None

        self._stop_event.clear()
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.pb.configure(value=0, maximum=len(algo_ids) * limit)
        self._log(f"starting: {len(algo_ids)} algos x {limit} images")
        self._log_main(f"benchmark: {len(algo_ids)} algos x {limit} images")
        self._algo_offset = 0
        self._algo_count = len(algo_ids)
        self._limit = limit
        self._bench_mode = self.var_mode.get()

        self._worker = threading.Thread(
            target=self._run, args=(algo_ids, limit, csv_p, md_p),
            daemon=True)
        self._worker.start()

    def _stop(self) -> None:
        self._stop_event.set()
        self.btn_stop.configure(state="disabled")
        self._log("stopping after current image…")

    def _on_close(self) -> None:
        if self._worker and self._worker.is_alive():
            if not messagebox.askyesno(
                "ModusMate",
                "Benchmark is still running. Stop and close?"):
                return
            self._stop_event.set()
            self._worker.join(timeout=5.0)
        self.destroy()

    # -------- worker thread
    def _run(self, algo_ids: List[int], limit: int,
             csv_p: Optional[Path], md_p: Optional[Path]) -> None:
        if self._bench_mode == "camera":
            self._run_camera(algo_ids, limit, csv_p, md_p)
        else:
            self._run_uart(algo_ids, limit, csv_p, md_p)

    def _run_camera(self, algo_ids: List[int], limit: int,
                    csv_p: Optional[Path], md_p: Optional[Path]) -> None:
        from .camera_bench import run_camera_benchmark, CamBenchProgress

        def on_progress(p: CamBenchProgress) -> None:
            # Reuse _handle_progress by wrapping into BenchProgress-like
            bp = bench.BenchProgress(
                kind=p.kind, message=p.message,
                algo_id=p.algo_id, algo_name=p.algo_name,
                image_index=p.image_index, image_total=p.image_total,
                algo_index=p.algo_index, algo_total=p.algo_total,
                correct=p.correct, seen=p.seen, no_detect=p.no_detect,
                accuracy=p.accuracy,
                mean_algo_us=p.mean_algo_us, mean_infer_us=p.mean_infer_us,
            )
            self.after(0, self._handle_progress, bp)

        try:
            cam_summary = run_camera_benchmark(
                port=self._port,
                algo_ids=algo_ids,
                limit=limit,
                stabilize_ms=self.var_stabilize.get(),
                capture_ms=self.var_capture.get(),
                confidence_threshold=self.var_conf_thresh.get(),
                progress=on_progress,
                stop_event=self._stop_event,
            )
            # Write CSV + summary
            if cam_summary.rows and csv_p:
                import csv as _csv
                fieldnames = ["algo", "algo_id", "image", "label", "label_id",
                              "pred_id", "pred_label", "conf", "algo_us", "infer_us"]
                with csv_p.open("w", newline="", encoding="utf-8") as f:
                    w = _csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(cam_summary.rows)
            if cam_summary.per_algo and md_p:
                lines = ["# Camera benchmark summary", ""]
                lines.append("| Algorithm | Accuracy | Correct | No-detect "
                             "| Mean algo µs | Mean infer µs |")
                lines.append("|---|---:|---:|---:|---:|---:|")
                for aid in algo_ids:
                    row = cam_summary.per_algo.get(aid)
                    if row is None:
                        continue
                    lines.append(
                        f"| {row['name']} | {row['accuracy']*100:5.1f}% | "
                        f"{row['correct']}/{row['seen']} | "
                        f"{row['no_detect']} | "
                        f"{row['mean_algo_us']} | {row['mean_infer_us']} |")
                md_p.write_text("\n".join(lines), encoding="utf-8")
        except Exception as e:
            self.after(0, self._log, f"camera benchmark failed: {e}")
        finally:
            self.after(0, self._on_done)

    def _run_uart(self, algo_ids: List[int], limit: int,
                  csv_p: Optional[Path], md_p: Optional[Path]) -> None:
        def on_progress(p: bench.BenchProgress) -> None:
            self.after(0, self._handle_progress, p)
        try:
            self._summary = bench.run_benchmark(
                port=self._port,
                algo_ids=algo_ids,
                limit=limit,
                csv_out=csv_p,
                summary_out=md_p,
                progress=on_progress,
                stop_event=self._stop_event,
                upgrade_baud=self._upgrade_baud,
                enable_preview=self._enable_preview,
                save_processed_dir=self._save_proc_dir,
            )
        except Exception as e:
            self.after(0, self._log, f"benchmark failed: {e}")
        finally:
            self.after(0, self._on_done)

    def _handle_progress(self, p: bench.BenchProgress) -> None:
        if p.kind == "start":
            self.lbl_status.config(text=p.message)
            self._log(p.message)
        elif p.kind == "algo_begin":
            self.lbl_status.config(
                text=f"{p.algo_index + 1}/{p.algo_total}: "
                     f"{p.algo_name}  (0/{p.image_total})")
            self._log(f"[{p.algo_name}] start")
        elif p.kind == "chunk":
            # sub-image streaming progress; reuse seen=sent and mean_algo_us=total
            sent, total = p.seen, p.mean_algo_us
            pct = (sent * 100 // total) if total else 0
            self.lbl_status.config(
                text=f"{p.algo_index + 1}/{p.algo_total}: {p.algo_name}  "
                     f"img {p.image_index}/{p.image_total}  "
                     f"↑ uploading {sent//1024}K/{total//1024}KB ({pct}%)")
            # Fractional within-image progress on the main bar so the user
            # can see the upload move even though we're still on image N.
            base = p.algo_index * self._limit + (p.image_index - 1)
            self.pb.configure(value=base + (sent / total if total else 0))
        elif p.kind == "image_loading":
            self.lbl_status.config(
                text=f"{p.algo_index + 1}/{p.algo_total}: {p.algo_name}  "
                     f"img {p.image_index}/{p.image_total}  "
                     f"↓ loading {p.message}…")
        elif p.kind == "image":
            done = p.algo_index * self._limit + p.image_index
            self.pb.configure(value=done)
            self.lbl_status.config(
                text=f"{p.algo_index + 1}/{p.algo_total}: {p.algo_name}  "
                     f"({p.image_index}/{p.image_total})  "
                     f"acc={p.accuracy * 100:5.1f}%  "
                     f"algoUs={p.mean_algo_us}  "
                     f"inferUs={p.mean_infer_us}")
        elif p.kind == "algo_end":
            self._log(f"[{p.algo_name}] {p.message}")
        elif p.kind == "image_begin" and p.image_rgb is not None:
            try:
                img = Image.frombytes("RGB", (320, 240), p.image_rgb)
                photo = ImageTk.PhotoImage(img)
            except Exception:
                return
            self._photo_input = photo
            if self._item_input is None:
                self._item_input = self.cv_input.create_image(
                    0, 0, anchor="nw", image=self._photo_input)
            else:
                self.cv_input.itemconfig(self._item_input,
                                         image=self._photo_input)
        elif p.kind == "image_processed" and p.preview_gray is not None:
            w, h = p.preview_w, p.preview_h
            if w <= 0 or h <= 0 or w * h != len(p.preview_gray):
                return
            try:
                img = _gray_preview_to_rgb(p.preview_gray, w, h,
                                           p.algo_name or "")
                photo = ImageTk.PhotoImage(img)
            except Exception:
                return
            self._photo_board = photo
            if self._item_board is None:
                self._item_board = self.cv_board.create_image(
                    0, 0, anchor="nw", image=self._photo_board)
            else:
                self.cv_board.itemconfig(self._item_board,
                                         image=self._photo_board)
        elif p.kind == "error":
            self._log(f"ERROR: {p.message}")
        elif p.kind == "done":
            self._log(f"finished: {p.message}")

    def _on_done(self) -> None:
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        if self._summary is not None:
            n_rows = len(self._summary.rows)
            self.lbl_status.config(text=f"done — {n_rows} samples processed")
            self._log_main(f"benchmark done: {n_rows} rows")


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
