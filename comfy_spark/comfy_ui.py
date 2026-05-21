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
from tkinter import filedialog, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
PRESETS_DIR = os.path.join(HERE, "presets")
LORAS_DIR = os.path.join(HERE, "loras")
LAUNCH = os.path.join(HERE, "comfy_launch.py")
DEFAULT_CATALOG = "ltx2.loras.json"

GPU_CHOICES = ["t4", "a10", "l4", "l40s", "l40sx4", "rtxpro6000", "rtxpro6000x8"]
CUSTOM_LORA = "(custom URL…)"  # sentinel option in the LoRA picker
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


# ── app ───────────────────────────────────────────────────────────────────────

class App:
    def __init__(self, root):
        self.root = root
        self.presets = load_presets()
        self.proc = None
        self.logq = queue.Queue()
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
        ttk.Checkbutton(bar, text="No trigger words", variable=self.no_trig).pack(side="left")
        self.convert_only = tk.BooleanVar(value=False)
        co = ttk.Checkbutton(bar, text="Convert only", variable=self.convert_only)
        co.pack(side="left", padx=(12, 0))
        _help(bar, "Convert the UI graph to API format on Spark and emit it (no render, no "
                   "weight download). The cheap way to validate a workflow / template preset.").pack(side="left", padx=4)
        self.run_btn = ttk.Button(bar, text="▶ Run on Spark", command=lambda: self.run(False))
        self.run_btn.pack(side="right")
        self.dry_btn = ttk.Button(bar, text="Dry run", command=lambda: self.run(True))
        self.dry_btn.pack(side="right", padx=6)
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="right", padx=6)

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
        self.lora_rows = []
        preset = self.presets[self.preset_var.get()]
        ui = preset.get("ui", {})
        self.title_lbl.config(text=ui.get("title", preset.get("description", "")[:90]))

        # primary input
        prim = ui.get("primary_input")
        needs_input = preset.get("requires_input", True)
        self.primary_var = tk.StringVar()
        # a mask file picker, shown only for presets that declare a `mask` param
        # (e.g. wan_vace_inpaint). The launcher auto-picks the static vs mask-video
        # graph from the file extension, so the GUI just needs to pass the path.
        self.mask_var = tk.StringVar()
        has_mask = "mask" in preset.get("params", {})
        if prim or needs_input or has_mask:
            f = ttk.LabelFrame(self.form_host, text="Input", padding=8)
            f.pack(fill="x", pady=6)
            if prim or needs_input:
                label = (prim or {}).get("label", "Primary input file")
                self._file_row(f, label, self.primary_var, (prim or {}).get("filetypes"),
                               (prim or {}).get("tooltip"))
            if has_mask:
                self._file_row(f, "Mask (image or video)", self.mask_var,
                               [["Image", "*.png *.jpg *.jpeg *.webp"],
                                ["Mask video", "*.mp4 *.mov *.webm *.mkv *.avi *.m4v"],
                                ["All files", "*.*"]],
                               "White = the region to inpaint/remove. A video extension auto-"
                               "selects the per-frame mask-video graph.")

        # declared knobs, grouped by section
        params = preset.get("params", {})
        sections = {}
        for name, spec in params.items():
            if name == "mask":
                continue  # handled by the dedicated mask file row above
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
            for name, spec, pui in sorted(rows, key=lambda r: r[2].get("order", 99)):
                self._param_row(frame, name, spec, pui)

        # LoRA stack picker
        if preset.get("lora_chain"):
            self._lora_section(preset)

        # output dir + compute
        self._tail_section(preset)

    def _file_row(self, parent, label, var, filetypes, tip):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=18).pack(side="left")
        _help(row, tip).pack(side="left", padx=(0, 6))
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
        ft = [tuple(t) for t in filetypes] if filetypes else [("All files", "*.*")]
        ttk.Button(row, text="Browse…",
                   command=lambda: var.set(filedialog.askopenfilename(filetypes=ft) or var.get())
                   ).pack(side="left", padx=4)

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

    def run(self, dry_run):
        if self.proc and self.proc.poll() is None:
            self._append("[ui] a job is already running; Stop it first.\n")
            return
        argv = self.build_argv(dry_run)
        self.log.delete("1.0", "end")
        self._append("[ui] " + " ".join(self._q(a) for a in argv) + "\n\n")
        self.run_btn.config(state="disabled")
        self.dry_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        threading.Thread(target=self._run_thread, args=(argv,), daemon=True).start()

    def _run_thread(self, argv):
        try:
            self.proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, bufsize=1, cwd=os.path.dirname(HERE))
            for line in iter(self.proc.stdout.readline, ""):
                self.logq.put(line)
            self.proc.stdout.close()
            self.proc.wait()
            self.logq.put(f"\n[ui] finished (exit {self.proc.returncode}).\n")
        except Exception as e:  # noqa: BLE001
            self.logq.put(f"\n[ui] ERROR launching: {e}\n")
        finally:
            self.logq.put("__DONE__")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self._append("\n[ui] stop requested (the Spark job may keep running — use the CLI --cancel).\n")

    def _drain_log(self):
        try:
            while True:
                line = self.logq.get_nowait()
                if line == "__DONE__":
                    self.run_btn.config(state="normal")
                    self.dry_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                else:
                    self._append(line)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log)

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
