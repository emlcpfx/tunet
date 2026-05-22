"""
comfy_ui.py — a tiny desktop front-end for comfy_launch.py.

The whole point of comfy_spark is "easier than ComfyUI": friendly flags instead
of a node graph. This is the same idea with a mouse. It is *preset-driven* — it
reads each presets/*.preset.json and renders ONLY the knobs that preset declares
in its `ui` metadata (with [?] tooltips), a file picker for the primary input,
and a LoRA-stack picker fed by the catalog. Hit Run and it shells out to
comfy_launch.py, streaming the job log into the pane at the bottom.

Pure stdlib (tkinter). Run:  python comfy_spark/comfy_ui.py

Auth + billing are unchanged — they live in comfy_launch.py / .env.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from tkinter import filedialog, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
PRESETS_DIR = os.path.join(HERE, "presets")
LORAS_DIR = os.path.join(HERE, "loras")
LAUNCH = os.path.join(HERE, "comfy_launch.py")
DEFAULT_CATALOG = "ltx2.loras.json"

GPU_CHOICES = ["t4", "a10", "l4", "l40s", "l40sx4", "rtxpro6000", "rtxpro6000x8"]
CUSTOM_LORA = "(custom URL…)"  # sentinel option in the LoRA picker


def _load_output_formats():
    """Output-format names from outputs/outputs.json (drives the format dropdown)."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "outputs.json")
    try:
        with open(p, encoding="utf-8") as f:
            return list((json.load(f).get("formats") or {}).keys()) or ["mp4"]
    except Exception:
        return ["mp4"]


OUTPUT_CHOICES = _load_output_formats()
# Friendly params map to dedicated flags (so e.g. prompt gets trigger injection);
# every other declared param is driven generically via --set node.path=value.
FRIENDLY = {"prompt": "--prompt", "negative": "--negative",
            "strength": "--strength", "fps": "--fps", "mask": "--mask"}


# ── tooltip ───────────────────────────────────────────────────────────────────

class Tooltip:
    """Hover text on a widget — used by the little [?] labels next to each knob."""

    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + 18
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#ffffe0",
                 relief="solid", borderwidth=1, wraplength=360,
                 font=("Segoe UI", 9)).pack(ipadx=4, ipady=2)

    def _hide(self, _):
        if self.tip:
            self.tip.destroy()
            self.tip = None


def _help(parent, text):
    lbl = tk.Label(parent, text=" ? ", fg="white", bg="#5b8def", cursor="question_arrow",
                   font=("Segoe UI", 8, "bold"))
    if text:
        Tooltip(lbl, text)
    return lbl


# ── data ────────────────────────────────────────────────────────────────────

def load_presets():
    out = {}
    if not os.path.isdir(PRESETS_DIR):
        return out
    for f in sorted(os.listdir(PRESETS_DIR)):
        if f.endswith(".preset.json"):
            try:
                with open(os.path.join(PRESETS_DIR, f), encoding="utf-8") as fh:
                    out[f[:-len(".preset.json")]] = json.load(fh)
            except (OSError, ValueError):
                pass
    return out


def load_catalog(name):
    path = name if os.path.isabs(name) or os.path.sep in name else os.path.join(LORAS_DIR, name)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("loras", {})
    except (OSError, ValueError):
        return {}


def _parse_rate(s):
    """ffprobe rates are 'num/den' (e.g. '24000/1001') or a bare float."""
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            return float(a) / float(b) if float(b) else None
        return float(s) if s else None
    except (ValueError, ZeroDivisionError):
        return None


def probe_video(path):
    """(nframes, fps) for a clip via ffprobe, or (None, None) if it can't be read
    (ffprobe missing, not a video, parse error). Reads container metadata only —
    duration × fps — so it's fast (no full-clip decode like -count_frames)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate,r_frame_rate,nb_frames",
             "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None, None
    if out.returncode != 0:
        return None, None
    try:
        data = json.loads(out.stdout or "{}")
    except ValueError:
        return None, None
    st = (data.get("streams") or [{}])[0]
    fps = _parse_rate(st.get("avg_frame_rate") or "") or _parse_rate(st.get("r_frame_rate") or "")
    nb = st.get("nb_frames")
    nframes = int(nb) if (nb and str(nb).isdigit() and int(nb) > 0) else None
    if nframes is None:                       # container didn't store a count → derive it
        dur = (data.get("format") or {}).get("duration")
        try:
            if dur and fps:
                nframes = round(float(dur) * fps)
        except (ValueError, TypeError):
            nframes = None
    return (nframes or None), fps


# ── app ───────────────────────────────────────────────────────────────────────

class App:
    def __init__(self, root):
        self.root = root
        self.presets = load_presets()
        self.proc = None
        self.logq = queue.Queue()
        self.jobq = {}              # id -> {id,label,argv,status}; serial job queue
        self._next_id = 1
        self._running = False       # main-thread gate so only one job runs at once
        self._viewing = None        # job id whose log is currently shown in the pane
        self.param_getters = {}     # param name -> () -> value (or None to skip)
        self.lora_rows = []         # list of (frame, name, strength)
        root.title("comfy_spark — LTX-2 on Spark Fuse")
        root.geometry("840x900")

        if not self.presets:
            tk.Label(root, text=f"No presets found in {PRESETS_DIR}", fg="red").pack(pady=20)
            return

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Preset:", font=("Segoe UI", 10, "bold")).pack(side="left")
        self.preset_var = tk.StringVar(value=next(iter(self.presets)))
        self.preset_box = ttk.Combobox(top, textvariable=self.preset_var, state="readonly",
                                       values=list(self.presets), width=30)
        self.preset_box.pack(side="left", padx=8)
        self.preset_box.bind("<<ComboboxSelected>>", lambda _e: self.build_form())
        self.title_lbl = ttk.Label(top, text="", foreground="#444", wraplength=460)
        self.title_lbl.pack(side="left", padx=12)

        # scrollable form area
        mid = ttk.Frame(root)
        mid.pack(fill="both", expand=True, padx=10)
        self.canvas = tk.Canvas(mid, highlightthickness=0)
        sb = ttk.Scrollbar(mid, orient="vertical", command=self.canvas.yview)
        self.form_host = ttk.Frame(self.canvas)
        self.form_host.bind("<Configure>",
                            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.form_host, anchor="nw", width=800)
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # action bar
        bar = ttk.Frame(root, padding=(10, 6))
        bar.pack(fill="x")
        self.no_trig = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Don't auto-add LoRA trigger words", variable=self.no_trig).pack(side="left")
        _help(bar, "Some LoRAs only activate when a trigger word is in the prompt (e.g. the "
                   "'transition' LoRA needs 'zhuanchang'). comfy_spark normally appends those "
                   "automatically. Check this to turn that off and place triggers yourself. Only "
                   "affects catalog LoRAs that declare a trigger.").pack(side="left", padx=(4, 0))
        self.convert_only = tk.BooleanVar(value=False)
        co = ttk.Checkbutton(bar, text="Convert only", variable=self.convert_only)
        co.pack(side="left", padx=(12, 0))
        _help(bar, "Convert the UI graph to API format on Spark and emit it (no render, no "
                   "weight download). The cheap way to validate a workflow / template preset.").pack(side="left", padx=4)
        self.add_btn = ttk.Button(bar, text="▶ Add to queue", command=lambda: self.enqueue(False))
        self.add_btn.pack(side="right")
        self.dry_btn = ttk.Button(bar, text="Dry run", command=lambda: self.enqueue(True))
        self.dry_btn.pack(side="right", padx=6)

        # queue — jobs run one at a time (serial), so a warm node carries across
        # the batch via idle-hold. Queue as many as you want; watch status here.
        qf = ttk.LabelFrame(root, text="Queue (serial)", padding=4)
        qf.pack(fill="x", padx=10, pady=(0, 6))
        self.tree = ttk.Treeview(qf, columns=("job", "status"), show="headings", height=5)
        self.tree.heading("job", text="Job")
        self.tree.heading("status", text="Status")
        self.tree.column("job", width=560, anchor="w")
        self.tree.column("status", width=150, anchor="w")
        self.tree.pack(side="left", fill="x", expand=True)
        # click a row -> show that job's own console output below
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        qb = ttk.Frame(qf)
        qb.pack(side="right", fill="y", padx=(6, 0))
        self.stop_btn = ttk.Button(qb, text="Stop current", command=self.stop, state="disabled")
        self.stop_btn.pack(fill="x", pady=1)
        ttk.Button(qb, text="Remove", command=self._remove_selected).pack(fill="x", pady=1)
        ttk.Button(qb, text="Clear finished", command=self._clear_finished).pack(fill="x", pady=1)

        # log
        logf = ttk.LabelFrame(root, text="Job log", padding=4)
        logf.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log = tk.Text(logf, height=12, bg="#11151c", fg="#cdd9e5",
                           font=("Consolas", 9), wrap="word")
        self.log.pack(side="left", fill="both", expand=True)
        ls = ttk.Scrollbar(logf, command=self.log.yview)
        ls.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=ls.set)

        self.build_form()
        self.root.after(100, self._drain_log)

    # ── form construction ─────────────────────────────────────────────────────

    def build_form(self):
        for w in self.form_host.winfo_children():
            w.destroy()
        self.param_getters = {}
        self.param_vars = {}        # param name -> tk var (for the resolution dropdown to drive)
        self.lora_rows = []
        self.detect_note = None     # label under the input row; filled by the clip probe
        preset = self.presets[self.preset_var.get()]
        ui = preset.get("ui", {})
        self.title_lbl.config(text=ui.get("title", preset.get("description", "")[:90]))

        # plain-language "what is this?" panel + canonical docs link(s)
        self._about_section(preset)

        # primary input
        prim = ui.get("primary_input")
        needs_input = preset.get("requires_input", True)
        self.primary_var = tk.StringVar()
        # a mask file picker, shown only for presets that declare a `mask` param
        # (e.g. wan_vace_inpaint). The launcher auto-picks the static vs mask-video
        # graph from the file extension, so the GUI just needs to pass the path.
        self.mask_var = tk.StringVar()
        # a face/identity reference image picker, shown for face-swap presets that
        # declare a `face` param (e.g. ltx_faceswap). Label/tooltip/filetypes come
        # from the preset's ui.secondary_input metadata when present.
        self.face_var = tk.StringVar()
        has_mask = "mask" in preset.get("params", {})
        has_face = "face" in preset.get("params", {})
        sec = ui.get("secondary_input") or {}
        if prim or needs_input or has_mask or has_face:
            f = ttk.LabelFrame(self.form_host, text="Input", padding=8)
            f.pack(fill="x", pady=6)
            if prim or needs_input:
                label = (prim or {}).get("label", "Primary input file")
                self._file_row(f, label, self.primary_var, (prim or {}).get("filetypes"),
                               (prim or {}).get("tooltip"), on_change=self._probe_and_fill)
                # auto-detect line: shows the picked clip's frame count + fps and
                # which Frames/fps values we filled in below.
                self.detect_note = ttk.Label(f, text="", wraplength=720)
                self.detect_note.pack(anchor="w", padx=(26, 0))
            if has_mask:
                self._file_row(f, "Mask (image or video)", self.mask_var,
                               [["Image", "*.png *.jpg *.jpeg *.webp"],
                                ["Mask video", "*.mp4 *.mov *.webm *.mkv *.avi *.m4v"],
                                ["All files", "*.*"]],
                               "White = the region to inpaint/remove. A video extension auto-"
                               "selects the per-frame mask-video graph.")
            if has_face:
                self._file_row(f, sec.get("label", "Face reference image"), self.face_var,
                               sec.get("filetypes",
                                       [["Image", "*.png *.jpg *.jpeg *.webp"], ["All files", "*.*"]]),
                               sec.get("tooltip", "The identity to swap in (a clear, well-lit "
                                       "frontal face works best)."))

        # declared knobs, grouped by section
        params = preset.get("params", {})
        sections = {}
        for name, spec in params.items():
            if name in ("mask", "face"):
                continue  # handled by the dedicated file rows above
            pui = spec.get("ui")
            # sensible fallback: render prompt/negative even without ui metadata
            if not pui and name in ("prompt", "negative"):
                pui = {"label": name.title(), "widget": "multiline", "section": "Main"}
            if not pui:
                continue
            sections.setdefault(pui.get("section", "Main"), []).append((name, spec, pui))

        for section in ("Main", "Advanced", *[s for s in sections if s not in ("Main", "Advanced")]):
            rows = sections.get(section)
            if not rows:
                continue
            frame = ttk.LabelFrame(self.form_host, text=section, padding=8)
            frame.pack(fill="x", pady=6)
            # a Resolution dropdown sits atop whichever section holds the size pair
            if ui.get("resolutions") and any(n in ("width", "short_edge")
                                             for n, _, _ in rows):
                self._resolution_row(frame, ui["resolutions"], params)
            for name, spec, pui in sorted(rows, key=lambda r: r[2].get("order", 99)):
                self._param_row(frame, name, spec, pui)

        # LoRA stack picker
        if preset.get("lora_chain"):
            self._lora_section(preset)

        # output dir + compute
        self._tail_section(preset)

    @staticmethod
    def _docs_links(preset):
        """Normalize a preset's `docs` field to [(label, url)]. Accepts a bare URL
        string, a list of URLs, or a list of {label,url} objects."""
        docs = preset.get("docs")
        if not docs:
            return []
        if isinstance(docs, str):
            return [("docs", docs)]
        out = []
        for d in docs:
            if isinstance(d, str):
                out.append(("docs", d))
            elif isinstance(d, dict) and d.get("url"):
                out.append((d.get("label", "docs"), d["url"]))
        return out

    def _about_section(self, preset):
        """A plain-English 'what is this workflow' card: what it does, what it
        takes, the knobs that matter, tags, and clickable links to the canonical
        docs. Driven entirely by the preset's `about` / `tags` / `docs` metadata."""
        about = preset.get("about") or {}
        tags = preset.get("tags") or []
        links = self._docs_links(preset)
        if not (about or tags or links):
            return
        frame = ttk.LabelFrame(self.form_host, text="About this workflow", padding=8)
        frame.pack(fill="x", pady=6)

        def field(label, text):
            row = ttk.Frame(frame)
            row.pack(fill="x", anchor="w", pady=1)
            ttk.Label(row, text=label, width=13, anchor="nw",
                      font=("Segoe UI", 9, "bold")).pack(side="left")
            ttk.Label(row, text=text, wraplength=620, justify="left").pack(
                side="left", fill="x", expand=True)

        if about.get("what"):
            field("What it does", about["what"])
        if about.get("inputs"):
            field("Inputs", about["inputs"])
        if about.get("key_knobs"):
            field("Key settings", about["key_knobs"])
        if tags:
            field("Tags", "   ".join(tags))
        if links:
            row = ttk.Frame(frame)
            row.pack(fill="x", anchor="w", pady=(3, 1))
            ttk.Label(row, text="Docs", width=13, anchor="nw",
                      font=("Segoe UI", 9, "bold")).pack(side="left")
            for label, url in links:
                link = tk.Label(row, text=f"{label} ↗", fg="#2a6fdb", cursor="hand2",
                                font=("Segoe UI", 9, "underline"))
                link.pack(side="left", padx=(0, 12))
                link.bind("<Button-1>", lambda _e, u=url: webbrowser.open(u))

    def _file_row(self, parent, label, var, filetypes, tip, on_change=None):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=18).pack(side="left")
        _help(row, tip).pack(side="left", padx=(0, 6))
        entry = ttk.Entry(row, textvariable=var)
        entry.pack(side="left", fill="x", expand=True)
        ft = [tuple(t) for t in filetypes] if filetypes else [("All files", "*.*")]

        def browse():
            chosen = filedialog.askopenfilename(filetypes=ft)
            if chosen:
                var.set(chosen)
                if on_change:
                    on_change()
        ttk.Button(row, text="Browse…", command=browse).pack(side="left", padx=4)
        if on_change:                    # also probe a hand-typed / pasted path
            entry.bind("<FocusOut>", lambda _e: on_change())
            entry.bind("<Return>", lambda _e: on_change())

    # ── input-clip probe → auto-fill Frames + fps ───────────────────────────────

    def _probe_and_fill(self):
        """Read the just-picked input clip and auto-fill the Frames + fps knobs.
        Probes off-thread (ffprobe metadata read) and marshals the result back to
        the UI thread; degrades quietly if ffprobe isn't installed."""
        path = self.primary_var.get().strip()
        if not path or not os.path.isfile(path) or not self.detect_note:
            return
        self.detect_note.config(text="reading clip…", foreground="#888")
        note = self.detect_note

        def work():
            nframes, fps = probe_video(path)
            self.root.after(0, lambda: self._apply_probe(note, nframes, fps))
        threading.Thread(target=work, daemon=True).start()

    def _apply_probe(self, note, nframes, fps):
        if not note.winfo_exists():        # form was rebuilt while probing
            return
        if not nframes:
            note.config(text="couldn't read the clip — install ffmpeg/ffprobe to auto-detect "
                             "frames; left at the preset default.", foreground="#c33")
            return
        msg = [f"{nframes} frames" + (f" @ {fps:.4g} fps" if fps else "")]
        params = self.presets[self.preset_var.get()].get("params", {})
        lui = (params.get("length") or {}).get("ui", {})
        if nframes > 1 and "length" in self.param_vars:
            lo, hi = float(lui.get("min", 9)), float(lui.get("max", 497))
            valid = self._round_to_grid(nframes, lo, hi, float(lui.get("step", 8)))
            self.param_vars["length"].set(float(valid))
            if nframes > hi and valid >= hi:
                msg.append(f"→ Frames {valid} (capped at max)")
            elif valid != nframes:
                msg.append(f"→ Frames {valid} (nearest 8n+1)")
            else:
                msg.append(f"→ Frames {valid}")
        if fps and "fps" in self.param_vars:
            fui = (params.get("fps") or {}).get("ui", {})
            flo, fhi = float(fui.get("min", 1)), float(fui.get("max", 1000))
            rfps = int(max(flo, min(fhi, round(fps))))
            self.param_vars["fps"].set(float(rfps))
            msg.append(f"fps {rfps}" + (" (capped)" if rfps != round(fps) else ""))
        note.config(text="clip: " + ", ".join(msg), foreground="#2a7a2a")

    @staticmethod
    def _round_to_grid(x, lo, hi, step):
        """Snap x to the nearest lo + k·step (LTX's 8n+1 grid), clamped to [lo, hi]."""
        k = round((x - lo) / step)
        return int(max(lo, min(hi, lo + k * step)))

    def _resolution_row(self, parent, resolutions, params):
        """A dropdown of common known-good sizes that fills the size fields below.
        Driven entirely by the preset's ui.resolutions, so it stays in CLI parity
        (it just sets the same size params the CLI exposes). Two schemas:
          - width/height: literal output dimensions (most presets).
          - short_edge/long_edge: edge lengths for graphs that auto-orient the
            output to the source clip's aspect (e.g. the Obscura cleanplate),
            where the dropdown entries carry "short"/"long" instead."""
        if "short_edge" in params and "long_edge" in params:
            pa, pb, fa, fb = "short_edge", "long_edge", "short", "long"
            hint = ("Common known-good sizes. Picking one fills the Short/Long edge "
                    "fields below. This graph auto-orients output to the SOURCE clip, "
                    "so these set the two edge lengths, not the aspect — a landscape "
                    "clip comes back landscape regardless. Custom… (or edit) for any size.")
        else:
            pa, pb, fa, fb = "width", "height", "width", "height"
            hint = ("Common known-good sizes for this model. Picking one fills the "
                    "Width/Height fields below; pick Custom… (or just edit them) for any size.")
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="Resolution", width=18, anchor="nw").pack(side="left")
        _help(row, hint).pack(side="left", padx=(0, 6))
        by_label = {r["label"]: r for r in resolutions}
        da = (params.get(pa) or {}).get("default")
        db = (params.get(pb) or {}).get("default")
        sel = next((r["label"] for r in resolutions
                    if r.get(fa) == da and r.get(fb) == db), "Custom…")
        var = tk.StringVar(value=sel)
        box = ttk.Combobox(row, textvariable=var, values=list(by_label) + ["Custom…"],
                           state="readonly", width=32)
        box.pack(side="left", fill="x", expand=True)

        def on_pick(_e=None):
            r = by_label.get(var.get())
            if r and pa in self.param_vars and pb in self.param_vars:
                self.param_vars[pa].set(float(r[fa]))
                self.param_vars[pb].set(float(r[fb]))
        box.bind("<<ComboboxSelected>>", on_pick)

    def _param_row(self, parent, name, spec, pui):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=pui.get("label", name), width=18, anchor="nw").pack(side="left")
        _help(row, pui.get("tooltip")).pack(side="left", padx=(0, 6), anchor="n")
        widget = pui.get("widget", "text")
        default = spec.get("default")

        if widget == "multiline":
            txt = tk.Text(row, height=3, wrap="word", font=("Segoe UI", 10))
            if default:
                txt.insert("1.0", str(default))
            txt.pack(side="left", fill="x", expand=True)
            self.param_getters[name] = lambda t=txt: (t.get("1.0", "end-1c").strip() or None)
        elif widget == "combo":
            var = tk.StringVar(value=str(default) if default is not None else "")
            ttk.Combobox(row, textvariable=var, values=[str(c) for c in pui.get("choices", [])],
                         state="readonly").pack(side="left", fill="x", expand=True)
            self.param_getters[name] = lambda v=var: (v.get() or None)
        elif widget in ("number", "slider"):
            var = tk.DoubleVar(value=float(default) if default is not None else 0.0)
            lo, hi = float(pui.get("min", 0)), float(pui.get("max", 100))
            step = float(pui.get("step", 1))
            integer = float(step).is_integer() and (default is None or float(default).is_integer())
            if widget == "slider":
                val_lbl = ttk.Label(row, width=6)
                scale = ttk.Scale(row, from_=lo, to=hi, variable=var,
                                  command=lambda _v, l=val_lbl, vr=var: l.config(text=f"{vr.get():.2f}"))
                scale.pack(side="left", fill="x", expand=True)
                val_lbl.config(text=f"{var.get():.2f}")
                val_lbl.pack(side="left", padx=4)
            else:
                ttk.Spinbox(row, from_=lo, to=hi, increment=step, textvariable=var,
                            width=10).pack(side="left")
            self.param_vars[name] = var      # let the Resolution dropdown drive width/height
            self.param_getters[name] = lambda v=var, i=integer: (int(round(v.get())) if i else round(v.get(), 3))
        else:  # text / entry
            var = tk.StringVar(value=str(default) if default is not None else "")
            ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
            self.param_getters[name] = lambda v=var: (v.get() or None)

    def _lora_section(self, preset):
        catalog = load_catalog(preset.get("lora_catalog", DEFAULT_CATALOG))
        frame = ttk.LabelFrame(self.form_host, text="LoRA stack  (drop-in, stackable — layered in order)",
                               padding=8)
        frame.pack(fill="x", pady=6)
        self.lora_list_frame = ttk.Frame(frame)
        self.lora_list_frame.pack(fill="x")

        add = ttk.Frame(frame)
        add.pack(fill="x", pady=(6, 0))
        ttk.Label(add, text="Add:").pack(side="left")
        names = (sorted(catalog) + [CUSTOM_LORA]) if catalog else [CUSTOM_LORA]
        pick = tk.StringVar(value=names[0])
        box = ttk.Combobox(add, textvariable=pick, values=names, state="readonly", width=22)
        box.pack(side="left", padx=4)

        ttk.Label(add, text="strength").pack(side="left", padx=(8, 2))
        strvar = tk.DoubleVar(value=1.0)
        ttk.Spinbox(add, from_=0.0, to=3.0, increment=0.1, textvariable=strvar, width=6).pack(side="left")
        urlvar = tk.StringVar()
        ttk.Entry(add, textvariable=urlvar, width=24).pack(side="left", padx=6)
        ttk.Button(add, text="+ Add",
                   command=lambda: self._add_lora(pick.get(), strvar.get(), urlvar, catalog)
                   ).pack(side="left")

        # a note line that reflects the picked catalog entry (trigger + tip)
        self.lora_note = ttk.Label(frame, text="pick a catalog LoRA, or “(custom URL…)” to paste a .safetensors URL",
                                   foreground="#888", wraplength=720)
        self.lora_note.pack(anchor="w", pady=(4, 0))

        def on_pick(_e=None):
            e = catalog.get(pick.get())
            if e:
                if e.get("strength") is not None:
                    strvar.set(float(e["strength"]))
                trig = f"  ·  trigger: {e['trigger']}" if e.get("trigger") else ""
                eg = f"\ne.g. prompt: “{e['prompt_example']}”" if e.get("prompt_example") else ""
                self.lora_note.config(text=((e.get("note", "") + trig) or "—") + eg)
            else:
                self.lora_note.config(text="paste a direct .safetensors URL on the right, then + Add")
        box.bind("<<ComboboxSelected>>", on_pick)

    def _add_lora(self, choice, strength, urlvar, catalog):
        is_custom = choice == CUSTOM_LORA
        ref = urlvar.get().strip() if is_custom else choice
        if not ref:
            return
        row = ttk.Frame(self.lora_list_frame)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=f"•  {ref}   @ {strength:g}", width=64, anchor="w").pack(side="left")
        ttk.Button(row, text="✕", width=3,
                   command=lambda: self._remove_lora(row)).pack(side="left")
        self.lora_rows.append((row, ref, float(strength)))
        if is_custom:
            urlvar.set("")

    def _remove_lora(self, row):
        self.lora_rows = [r for r in self.lora_rows if r[0] is not row]
        row.destroy()

    def _tail_section(self, preset):
        frame = ttk.LabelFrame(self.form_host, text="Output & compute", padding=8)
        frame.pack(fill="x", pady=6)

        r1 = ttk.Frame(frame)
        r1.pack(fill="x", pady=3)
        ttk.Label(r1, text="Download to", width=18).pack(side="left")
        _help(r1, "Local folder to pull the finished render (and converted graph) into when the job ends. Leave blank to fetch from ShareSync manually.").pack(side="left", padx=(0, 6))
        self.dl_var = tk.StringVar(value=os.path.join(HERE, "_dl"))
        ttk.Entry(r1, textvariable=self.dl_var).pack(side="left", fill="x", expand=True)
        ttk.Button(r1, text="Browse…",
                   command=lambda: self.dl_var.set(filedialog.askdirectory() or self.dl_var.get())
                   ).pack(side="left", padx=4)

        r2 = ttk.Frame(frame)
        r2.pack(fill="x", pady=3)
        ttk.Label(r2, text="GPU", width=18).pack(side="left")
        self.gpu_var = tk.StringVar(value=preset.get("gpu", "rtxpro6000"))
        ttk.Combobox(r2, textvariable=self.gpu_var, values=GPU_CHOICES, state="readonly",
                     width=14).pack(side="left")
        ttk.Label(r2, text="Mode").pack(side="left", padx=(16, 4))
        self.mode_var = tk.StringVar(value=preset.get("mode", "instant"))
        ttk.Combobox(r2, textvariable=self.mode_var, values=["instant", "smart"],
                     state="readonly", width=10).pack(side="left")
        ttk.Label(r2, text="Idle-hold (s)").pack(side="left", padx=(16, 4))
        self.idle_var = tk.IntVar(value=300)
        ttk.Spinbox(r2, from_=0, to=3600, increment=60, textvariable=self.idle_var,
                    width=8).pack(side="left")
        _help(r2, "Keep the warm node alive this long after the job so the next render skips the ~30 GB cold-start fetch.").pack(side="left", padx=6)
        ttk.Label(r2, text="Smart retries").pack(side="left", padx=(16, 4))
        self.retries_var = tk.IntVar(value=2)
        ttk.Spinbox(r2, from_=0, to=5, increment=1, textvariable=self.retries_var,
                    width=5).pack(side="left")
        _help(r2, "Only used when Mode = smart: re-launches on preemption [0–5]. Ignored in instant mode.").pack(side="left", padx=6)

        # Output format (high-bit EXR / ProRes). mp4 = the preset's own preview output.
        r2c = ttk.Frame(frame)
        r2c.pack(fill="x", pady=3)
        ttk.Label(r2c, text="Output format", width=18).pack(side="left")
        self.output_var = tk.StringVar(value="mp4")
        ttk.Combobox(r2c, textvariable=self.output_var, values=OUTPUT_CHOICES, state="readonly",
                     width=14).pack(side="left")
        _help(r2c, "mp4 = the preset's built-in preview. exr32/exr16 = scene-linear OpenEXR "
                   "sequence for Nuke/AE; prores_* = high-bit ProRes video. Any non-mp4 choice is "
                   "ADDED alongside the mp4 preview.").pack(side="left", padx=6)
        ttk.Label(r2c, text="Output fps").pack(side="left", padx=(16, 4))
        self.output_fps_var = tk.IntVar(value=24)
        ttk.Spinbox(r2c, from_=1, to=120, increment=1, textvariable=self.output_fps_var,
                    width=6).pack(side="left")
        _help(r2c, "Frame rate for video outputs (ProRes). Ignored for EXR/PNG sequences.").pack(side="left", padx=6)

        r3 = ttk.Frame(frame)
        r3.pack(fill="x", pady=3)
        ttk.Label(r3, text="Job name", width=18).pack(side="left")
        self.name_var = tk.StringVar()
        ttk.Entry(r3, textvariable=self.name_var, width=24).pack(side="left")
        ttk.Label(r3, text="Tags").pack(side="left", padx=(16, 4))
        self.tags_var = tk.StringVar()
        ttk.Entry(r3, textvariable=self.tags_var).pack(side="left", fill="x", expand=True)
        _help(r3, "Optional. Job name + extra grouping tags (comma-separated). cpfx_tunet/cpfx_comfy are always added.").pack(side="left", padx=6)

    # ── run ─────────────────────────────────────────────────────────────────--

    def build_argv(self, dry_run):
        preset_name = self.preset_var.get()
        preset = self.presets[preset_name]
        argv = [sys.executable, "-u", LAUNCH, "--preset", preset_name]

        primary = self.primary_var.get().strip()
        if primary:
            argv.append(primary)

        mask = self.mask_var.get().strip()
        if mask:
            argv += ["--mask", mask]

        face = self.face_var.get().strip()
        if face:
            argv += ["--face", face]

        for name, getter in self.param_getters.items():
            val = getter()
            if val is None or val == "":
                continue
            spec = preset["params"][name]
            if name in FRIENDLY:
                argv += [FRIENDLY[name], str(val)]
            else:
                argv += ["--set", f"{spec['node']}.{spec['path']}={json.dumps(val)}"]

        for _row, ref, strength in self.lora_rows:
            argv += ["--lora", f"{ref}:{strength:g}"]
        if self.no_trig.get():
            argv.append("--no-triggers")

        argv += ["--gpu", self.gpu_var.get(), "--mode", self.mode_var.get(),
                 "--idle-hold", str(self.idle_var.get())]
        if self.mode_var.get() == "smart":
            argv += ["--max-retries", str(self.retries_var.get())]
        if self.output_var.get() and self.output_var.get() != "mp4":
            argv += ["--output", self.output_var.get(),
                     "--output-fps", str(self.output_fps_var.get())]
        name = self.name_var.get().strip()
        if name:
            argv += ["--name", name]
        for tag in (t.strip() for t in self.tags_var.get().split(",")):
            if tag:
                argv += ["--tag", tag]
        dl = self.dl_var.get().strip()
        if dl:
            argv += ["--download", dl]
        if self.convert_only.get():
            argv.append("--convert-only")
        if dry_run:
            argv.append("--dry-run")
        return argv

    def enqueue(self, dry_run):
        """Snapshot the current form into a queued job. Jobs run serially; a warm
        node carries across the batch (idle-hold), so queue freely."""
        argv = self.build_argv(dry_run)
        prim = os.path.basename(self.primary_var.get().strip()) or "(no input)"
        label = ("[dry] " if dry_run else "") + f"{self.preset_var.get()} · {prim}"
        jid = self._next_id
        self._next_id += 1
        self.jobq[jid] = {"id": jid, "label": label, "argv": argv, "status": "queued", "log": []}
        self.tree.insert("", "end", iid=str(jid), values=(label, "queued"))
        self._job_log(jid, f"[ui] queued #{jid}: {label}\n"
                           + " ".join(self._q(a) for a in argv) + "\n")
        if self._viewing is None:           # show the very first job right away
            self._select_and_view(jid)
        self._pump()

    def _pump(self):
        """Start the next queued job if nothing is running (serial). Gated on a
        main-thread flag so two rapid enqueues can't launch concurrent jobs."""
        if self._running:
            return
        nxt = next((j for j in self.jobq.values() if j["status"] == "queued"), None)
        if not nxt:
            return
        self._running = True
        nxt["status"] = "running"
        self.tree.item(str(nxt["id"]), values=(nxt["label"], "running"))
        self.stop_btn.config(state="normal")
        self._job_log(nxt["id"], f"\n[ui] ▶ running #{nxt['id']}...\n\n")
        self._select_and_view(nxt["id"])    # follow the job that just started
        threading.Thread(target=self._run_job, args=(nxt,), daemon=True).start()

    def _run_job(self, job):
        try:
            self.proc = subprocess.Popen(job["argv"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, bufsize=1, cwd=os.path.dirname(HERE))
            for line in iter(self.proc.stdout.readline, ""):
                self.logq.put(("LOG", job["id"], line))
            self.proc.stdout.close()
            self.proc.wait()
            rc = self.proc.returncode
        except Exception as e:  # noqa: BLE001
            self.logq.put(("LOG", job["id"], f"\n[ui] ERROR launching: {e}\n"))
            rc = -1
        self.logq.put(("JOBEND", job["id"], rc))

    def stop(self):
        """Stop the locally-running job process. NOTE: the Spark job keeps running
        and its download won't complete — use the CLI --cancel to stop it on Spark."""
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            running = next((j for j in self.jobq.values() if j["status"] == "running"), None)
            if running:
                self._job_log(running["id"], "\n[ui] stop requested — the Spark job may keep "
                              "running and its download won't complete. Use the CLI --cancel.\n")

    def _remove_selected(self):
        for iid in self.tree.selection():
            j = self.jobq.get(int(iid))
            if j and j["status"] == "queued":      # only removable before it runs
                del self.jobq[int(iid)]
                self.tree.delete(iid)
                if self._viewing == int(iid):
                    self._viewing = None
                    self.log.delete("1.0", "end")

    def _clear_finished(self):
        for jid, j in list(self.jobq.items()):
            if j["status"] in ("done", "failed", "stopped"):
                del self.jobq[jid]
                self.tree.delete(str(jid))
                if self._viewing == jid:
                    self._viewing = None
                    self.log.delete("1.0", "end")

    def _drain_log(self):
        try:
            while True:
                item = self.logq.get_nowait()
                kind = item[0] if isinstance(item, tuple) else None
                if kind == "JOBEND":
                    _, jid, rc = item
                    j = self.jobq.get(jid)
                    if j:
                        j["status"] = "done" if rc == 0 else ("stopped" if rc < 0 else "failed")
                        self.tree.item(str(jid), values=(j["label"], j["status"]))
                    self._job_log(jid, f"\n[ui] #{jid} finished (exit {rc}).\n")
                    self.proc = None
                    self._running = False
                    self.stop_btn.config(state="disabled")
                    self._pump()                    # advance to the next queued job
                elif kind == "LOG":
                    self._job_log(item[1], item[2])
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log)

    # ── per-job log routing (each queued job keeps its own console output) ──────

    def _job_log(self, jid, text):
        """Append output to a job's own buffer; mirror to the pane only if that
        job is the one currently selected/viewed."""
        j = self.jobq.get(jid)
        if not j:
            return
        j.setdefault("log", []).append(text)
        if self._viewing == jid:
            self.log.insert("end", text)
            self.log.see("end")

    def _view_job(self, jid):
        """Repaint the log pane with the chosen job's full buffered output."""
        self._viewing = jid
        self.log.delete("1.0", "end")
        j = self.jobq.get(jid)
        if j:
            self.log.insert("end", "".join(j.get("log", [])))
            self.log.see("end")

    def _select_and_view(self, jid):
        self.tree.selection_set(str(jid))    # also fires _on_select
        self._view_job(jid)

    def _on_select(self, _e=None):
        sel = self.tree.selection()
        if sel:
            self._view_job(int(sel[0]))

    def _append(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    @staticmethod
    def _q(arg):
        return f'"{arg}"' if " " in arg else arg


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
