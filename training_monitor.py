"""
TuNet Training Monitor - Real-time loss graph visualization

Usage:
    python training_monitor.py --log_file path/to/training.log
    python training_monitor.py --output_dir path/to/output  (will find training.log there)
    python training_monitor.py --data_dir path/to/parent    (scan for all training runs)

The monitor reads the training log file and displays a live-updating loss graph.

Keyboard shortcuts:
    H / Home  - Fit all data in view
    1-4       - Zoom presets (Last 1, 5, 10, 20 epochs)
    L         - Toggle log scale Y-axis
    R         - Toggle raw data overlay
    S         - Cycle smoothing presets (0, 0.5, 0.9, 0.95, 0.99)
    G         - Toggle grid
"""

import os
import re
import argparse
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
from collections import deque
import time
import math

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


# Dark theme colors — matched to TuNet main app palette
COLORS = {
    'bg': '#1b1d23',
    'bg_light': '#22252c',
    'bg_panel': '#282b33',
    'border': '#3a3f4b',
    'grid': '#31353f',
    'text': '#c8ccd4',
    'text_bright': '#e2e5eb',
    'text_dim': '#8b919d',
    'l1': '#e86b6b',
    'l1_raw': '#e86b6b',
    'lpips': '#4ecdc4',
    'lpips_raw': '#4ecdc4',
    'val_l1': '#f0a050',
    'val_l1_raw': '#f0a050',
    'accent': '#5a9bf6',
    'best_marker': '#6bc77a',
    'crosshair': '#6a7080',
    'btn_active': '#3d6db5',
    'btn_normal': '#31353f',
    'btn_text': '#c8ccd4',
    'btn_border': '#4a5060',
    'good': '#6bc77a',
    'warn': '#e5c07b',
    'bad': '#e86b6b',
}

# Color palette for multiple runs
RUN_COLORS = [
    '#e86b6b', '#4ecdc4', '#f0a050', '#a78bfa',
    '#6bc77a', '#e5c07b', '#5a9bf6', '#e879f9',
    '#f97583', '#79c0ff', '#56d4dd', '#d2a8ff',
]


class RunData:
    """Encapsulates all data for a single training run."""

    def __init__(self, name, log_file, color_index=0):
        self.name = name
        self.log_file = log_file
        self.color_index = color_index
        self.color = RUN_COLORS[color_index % len(RUN_COLORS)]
        self.last_position = 0
        self.last_modified = 0
        self.max_points = 50000

        # Training data
        self.steps = deque(maxlen=self.max_points)
        self.l1_losses = deque(maxlen=self.max_points)
        self.lpips_losses = deque(maxlen=self.max_points)
        self.has_lpips = False
        self.loss_label = 'Loss'

        # Validation data
        self.val_steps = deque(maxlen=self.max_points)
        self.val_l1_losses = deque(maxlen=self.max_points)
        self.val_lpips_losses = deque(maxlen=self.max_points)
        self.val_psnr = deque(maxlen=self.max_points)
        self.val_ssim = deque(maxlen=self.max_points)
        self.has_val_data = False
        self.best_val_l1 = float('inf')
        self.best_val_l1_epoch = 0

        # Best tracking
        self.best_l1 = float('inf')
        self.best_l1_epoch = 0
        self.best_lpips = float('inf')
        self.best_lpips_epoch = 0

        # Timing
        self.epoch_start_times = {}
        self.time_per_step = deque(maxlen=1000)

        # Matplotlib line references (set when creating graph lines)
        self.l1_line = None
        self.l1_raw_line = None
        self.lpips_line = None
        self.lpips_raw_line = None
        self.val_l1_line = None
        self.val_l1_raw_line = None
        self.best_l1_marker = None
        self.best_lpips_marker = None
        self.best_l1_annot = None

    def clear(self):
        self.steps.clear()
        self.l1_losses.clear()
        self.lpips_losses.clear()
        self.has_lpips = False
        self.loss_label = 'Loss'
        self.last_position = 0
        self.best_l1 = float('inf')
        self.best_l1_epoch = 0
        self.best_lpips = float('inf')
        self.best_lpips_epoch = 0
        self.time_per_step.clear()
        self.val_steps.clear()
        self.val_l1_losses.clear()
        self.val_lpips_losses.clear()
        self.val_psnr.clear()
        self.val_ssim.clear()
        self.has_val_data = False
        self.best_val_l1 = float('inf')
        self.best_val_l1_epoch = 0


class TrainingMonitor:
    def __init__(self, root, log_file=None, data_dir=None):
        self.root = root
        self.root.title("TuNet Training Monitor")
        self.root.geometry("1280x750")
        self.root.minsize(900, 500)
        self.root.configure(bg=COLORS['bg'])

        # Run management
        self.runs = []          # List of RunData
        self.active_run_idx = 0  # Index of selected run for stats display

        # UI state
        self.smoothing = tk.DoubleVar(value=0.6)
        self.update_interval = 1000
        self.log_scale = False
        self.show_raw = True
        self.show_grid = True
        self.zoom_mode = 'all'
        self.crosshair_visible = False
        self._pan_start_px = None
        self.is_monitoring = False

        self.create_widgets()

        # Bind keyboard shortcuts
        self.root.bind('<Key>', self.on_key_press)

        # Load initial runs
        if data_dir:
            self._scan_directory(data_dir)
        if log_file and os.path.exists(log_file):
            self._add_run_from_log(log_file)

        # Auto-start if we have runs
        if self.runs:
            self.start_monitoring()

    def create_widgets(self):
        # ─── Main horizontal layout: run panel | graph + controls ───
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=COLORS['bg'],
                                   sashwidth=4, sashrelief='flat')
        main_pane.pack(fill=tk.BOTH, expand=True)

        # ─── Left: Run list panel ───
        run_panel = tk.Frame(main_pane, bg=COLORS['bg_panel'], width=200)

        tk.Label(run_panel, text="Runs", bg=COLORS['bg_panel'], fg=COLORS['text_bright'],
                 font=('Segoe UI', 10, 'bold')).pack(fill=tk.X, padx=6, pady=(6, 2))

        self.run_listbox = tk.Listbox(run_panel, bg=COLORS['bg_light'], fg=COLORS['text'],
                                       selectbackground=COLORS['btn_active'],
                                       selectforeground=COLORS['text_bright'],
                                       font=('Consolas', 9), relief='flat',
                                       highlightthickness=0, activestyle='none')
        self.run_listbox.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        self.run_listbox.bind('<<ListboxSelect>>', self._on_run_selected)
        self.run_listbox.bind('<Double-Button-1>', self._rename_run)

        btn_frame = tk.Frame(run_panel, bg=COLORS['bg_panel'])
        btn_frame.pack(fill=tk.X, padx=4, pady=4)

        self._make_button(btn_frame, "Add", self._add_run_dialog, small=True).pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        self._make_button(btn_frame, "Scan Dir", self._scan_dir_dialog, small=True).pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        self._make_button(btn_frame, "Remove", self._remove_selected_run, small=True).pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)

        main_pane.add(run_panel, width=200, minsize=150)

        # ─── Right: Graph and controls ───
        right_frame = tk.Frame(main_pane, bg=COLORS['bg'])

        # ─── Top control bar ───
        control_frame = tk.Frame(right_frame, bg=COLORS['bg'], pady=4)
        control_frame.pack(fill=tk.X, padx=8)

        self.monitor_btn = self._make_button(control_frame, "Start", self.toggle_monitoring)
        self.monitor_btn.pack(side=tk.LEFT, padx=2)

        # ─── Toolbar row: zoom presets + toggles + smoothing ───
        toolbar_frame = tk.Frame(right_frame, bg=COLORS['bg_panel'], pady=3)
        toolbar_frame.pack(fill=tk.X, padx=8, pady=(0, 2))

        # Zoom presets
        zoom_group = tk.Frame(toolbar_frame, bg=COLORS['bg_panel'])
        zoom_group.pack(side=tk.LEFT, padx=4)
        tk.Label(zoom_group, text="View:", bg=COLORS['bg_panel'], fg=COLORS['text_dim'],
                 font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=(0, 4))

        self.zoom_buttons = {}
        for label, mode in [("All [H]", 'all'), ("1 ep [1]", 'last_1'),
                            ("5 ep [2]", 'last_5'), ("10 ep [3]", 'last_10'),
                            ("20 ep [4]", 'last_20')]:
            btn = self._make_button(zoom_group, label,
                                    lambda m=mode: self.set_zoom(m), small=True)
            btn.pack(side=tk.LEFT, padx=1)
            self.zoom_buttons[mode] = btn
        self._highlight_zoom_button()

        # Separator
        tk.Frame(toolbar_frame, bg=COLORS['border'], width=1).pack(
            side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        # Toggles
        toggle_group = tk.Frame(toolbar_frame, bg=COLORS['bg_panel'])
        toggle_group.pack(side=tk.LEFT, padx=4)

        self.log_btn = self._make_button(toggle_group, "Log Y [L]",
                                         self.toggle_log_scale, small=True)
        self.log_btn.pack(side=tk.LEFT, padx=1)

        self.raw_btn = self._make_button(toggle_group, "Raw [R]",
                                         self.toggle_raw, small=True)
        self.raw_btn.pack(side=tk.LEFT, padx=1)
        self._update_toggle_btn(self.raw_btn, self.show_raw)

        self.grid_btn = self._make_button(toggle_group, "Grid [G]",
                                          self.toggle_grid, small=True)
        self.grid_btn.pack(side=tk.LEFT, padx=1)
        self._update_toggle_btn(self.grid_btn, self.show_grid)

        # Separator
        tk.Frame(toolbar_frame, bg=COLORS['border'], width=1).pack(
            side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        # Smoothing
        smooth_group = tk.Frame(toolbar_frame, bg=COLORS['bg_panel'])
        smooth_group.pack(side=tk.LEFT, padx=4)
        tk.Label(smooth_group, text="Smooth [S]:", bg=COLORS['bg_panel'], fg=COLORS['text_dim'],
                 font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=(0, 4))

        self.smooth_label = tk.Label(smooth_group, text="0.60", bg=COLORS['bg_panel'],
                                     fg=COLORS['accent'], font=('Consolas', 9, 'bold'), width=4)
        self.smooth_label.pack(side=tk.LEFT, padx=(0, 4))

        smooth_scale = tk.Scale(smooth_group, from_=0.0, to=0.99, resolution=0.01,
                                variable=self.smoothing, orient=tk.HORIZONTAL, length=120,
                                bg=COLORS['bg_panel'], fg=COLORS['text'], troughcolor=COLORS['bg_light'],
                                highlightthickness=0, showvalue=False,
                                command=self._on_smooth_change)
        smooth_scale.pack(side=tk.LEFT, padx=2)

        # Clear button on right side
        self._make_button(toolbar_frame, "Clear All", self.clear_all_data, small=True).pack(
            side=tk.RIGHT, padx=4)

        # ─── Graph area ───
        if HAS_MATPLOTLIB:
            self.create_matplotlib_graph(right_frame)
        else:
            self.create_fallback_display(right_frame)

        # ─── Analysis panel ───
        self.create_analysis_panel(right_frame)

        # ─── Stats panel at bottom ───
        self.create_stats_panel(right_frame)

        main_pane.add(right_frame, minsize=400)

    # ─── Run management ───

    def _add_run_from_log(self, log_file, name=None):
        """Add a run from a log file path. Returns the RunData or None if duplicate."""
        log_file = os.path.abspath(log_file)
        # Skip duplicates
        for r in self.runs:
            if os.path.abspath(r.log_file) == log_file:
                return None
        if name is None:
            name = os.path.basename(os.path.dirname(log_file))
            if not name or name == '.':
                name = os.path.basename(log_file)
        run = RunData(name, log_file, color_index=len(self.runs))
        self.runs.append(run)
        if HAS_MATPLOTLIB:
            self._create_run_lines(run)
        self._refresh_run_listbox()
        return run

    def _scan_directory(self, parent_dir):
        """Scan a parent directory for subdirectories containing training.log."""
        parent_dir = os.path.abspath(parent_dir)
        found = 0
        # Check the directory itself
        log_here = os.path.join(parent_dir, 'training.log')
        if os.path.exists(log_here):
            if self._add_run_from_log(log_here):
                found += 1
        # Check subdirectories
        try:
            for entry in sorted(os.scandir(parent_dir), key=lambda e: e.name):
                if entry.is_dir():
                    log_path = os.path.join(entry.path, 'training.log')
                    if os.path.exists(log_path):
                        if self._add_run_from_log(log_path):
                            found += 1
        except PermissionError:
            pass
        return found

    def _refresh_run_listbox(self):
        self.run_listbox.delete(0, tk.END)
        for i, run in enumerate(self.runs):
            color_marker = '\u25cf'  # filled circle
            self.run_listbox.insert(tk.END, f" {color_marker} {run.name}")
            self.run_listbox.itemconfigure(i, fg=run.color)
        if self.runs and self.active_run_idx < len(self.runs):
            self.run_listbox.selection_set(self.active_run_idx)

    def _on_run_selected(self, event):
        sel = self.run_listbox.curselection()
        if sel:
            self.active_run_idx = sel[0]
            self.update_stats()

    def _rename_run(self, event):
        sel = self.run_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        run = self.runs[idx]
        new_name = simpledialog.askstring("Rename Run", "Enter new name:",
                                          initialvalue=run.name, parent=self.root)
        if new_name and new_name.strip():
            run.name = new_name.strip()
            self._refresh_run_listbox()
            if HAS_MATPLOTLIB:
                self._update_legend()
                self.canvas.draw_idle()

    def _add_run_dialog(self):
        path = filedialog.askopenfilename(
            title="Select Training Log",
            filetypes=[("Log files", "*.log"), ("All files", "*.*")]
        )
        if path:
            run = self._add_run_from_log(path)
            if run:
                self.read_log_file(run, full_read=True)
                self.update_graph()
                self.update_stats()
                if not self.is_monitoring:
                    self.start_monitoring()

    def _scan_dir_dialog(self):
        path = filedialog.askdirectory(title="Select Parent Directory to Scan")
        if path:
            found = self._scan_directory(path)
            if found > 0:
                for run in self.runs:
                    self.read_log_file(run, full_read=True)
                self.update_graph()
                self.update_stats()
                if not self.is_monitoring:
                    self.start_monitoring()

    def _remove_selected_run(self):
        sel = self.run_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        run = self.runs[idx]
        # Remove matplotlib lines
        if HAS_MATPLOTLIB:
            for attr in ('l1_line', 'l1_raw_line', 'lpips_line', 'lpips_raw_line',
                          'val_l1_line', 'val_l1_raw_line', 'best_l1_marker',
                          'best_lpips_marker'):
                line = getattr(run, attr, None)
                if line is not None:
                    line.remove()
            if run.best_l1_annot is not None:
                run.best_l1_annot.remove()
        self.runs.pop(idx)
        if self.active_run_idx >= len(self.runs):
            self.active_run_idx = max(0, len(self.runs) - 1)
        self._refresh_run_listbox()
        if HAS_MATPLOTLIB:
            self._update_legend()
            self.canvas.draw_idle()
        self.update_stats()

    # ─── Matplotlib graph lines per run ───

    def _create_run_lines(self, run):
        """Create matplotlib line objects for a run."""
        c = run.color
        raw_alpha = 0.2
        # Primary axis lines
        run.l1_raw_line, = self.ax.plot([], [], '-', color=c,
                                         linewidth=0.5, alpha=raw_alpha, zorder=1)
        run.l1_line, = self.ax.plot([], [], '-', color=c,
                                     linewidth=1.8, alpha=0.95, label=run.name, zorder=3)
        # Validation
        val_c = c  # same color, dashed
        run.val_l1_raw_line, = self.ax.plot([], [], '--', color=val_c,
                                             linewidth=0.5, alpha=raw_alpha, zorder=1)
        run.val_l1_line, = self.ax.plot([], [], '--o', color=val_c,
                                         linewidth=1.4, alpha=0.8, markersize=3, zorder=4)
        # LPIPS on secondary axis
        lpips_c = c
        run.lpips_raw_line, = self.ax2.plot([], [], '-', color=lpips_c,
                                             linewidth=0.5, alpha=raw_alpha, zorder=1)
        run.lpips_line, = self.ax2.plot([], [], '-', color=lpips_c,
                                         linewidth=1.4, alpha=0.7, linestyle='dotted', zorder=3)
        # Best markers
        run.best_l1_marker, = self.ax.plot([], [], '*', color=c,
                                            markersize=12, zorder=5)
        run.best_lpips_marker, = self.ax2.plot([], [], '*', color=c,
                                                markersize=12, zorder=5)
        run.best_l1_annot = self.ax.annotate('', xy=(0, 0), fontsize=7,
                                              color=c,
                                              bbox=dict(boxstyle='round,pad=0.2',
                                                        facecolor=COLORS['bg_light'],
                                                        edgecolor=c, alpha=0.85),
                                              zorder=6)

    def _make_button(self, parent, text, command, small=False):
        font = ('Segoe UI', 8) if small else ('Segoe UI', 9)
        padx = 6 if small else 10
        pady = 2 if small else 4
        btn = tk.Label(parent, text=text, bg=COLORS['btn_normal'], fg=COLORS['btn_text'],
                       font=font, padx=padx, pady=pady, cursor='hand2',
                       relief='raised', borderwidth=1)
        btn.bind('<Button-1>', lambda e: command())
        btn.bind('<Enter>', lambda e: btn.configure(bg=COLORS['btn_active']))
        btn.bind('<Leave>', lambda e: self._btn_leave(btn))
        btn._is_active = False
        return btn

    def _btn_leave(self, btn):
        if getattr(btn, '_is_active', False):
            btn.configure(bg=COLORS['btn_active'])
        else:
            btn.configure(bg=COLORS['btn_normal'])

    def _update_toggle_btn(self, btn, active):
        btn._is_active = active
        btn.configure(bg=COLORS['btn_active'] if active else COLORS['btn_normal'])

    def _highlight_zoom_button(self):
        for mode, btn in self.zoom_buttons.items():
            self._update_toggle_btn(btn, mode == self.zoom_mode)

    def _on_smooth_change(self, val):
        self.smooth_label.configure(text=f"{float(val):.2f}")
        if any(r.steps for r in self.runs):
            self.update_graph()

    def create_matplotlib_graph(self, parent):
        """Create matplotlib figure with dual Y-axes."""
        self.fig = Figure(figsize=(10, 5), dpi=100, facecolor=COLORS['bg'])

        # Primary axis (L1)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(COLORS['bg_light'])
        self.ax.set_xlabel('Epoch', color=COLORS['text'], fontsize=10)
        self.ax.set_ylabel('Loss', color=COLORS['text'], fontsize=10)
        self.ax.tick_params(axis='x', colors=COLORS['text'], labelsize=9)
        self.ax.tick_params(axis='y', colors=COLORS['text'], labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color(COLORS['border'])
        self.ax.grid(True, color=COLORS['grid'], linestyle='-', linewidth=0.5, alpha=0.5)

        # Secondary axis (LPIPS) - created but hidden until needed
        self.ax2 = self.ax.twinx()
        self.ax2.set_ylabel('LPIPS Loss', color=COLORS['lpips'], fontsize=10)
        self.ax2.tick_params(axis='y', colors=COLORS['lpips'], labelsize=9)
        for spine in self.ax2.spines.values():
            spine.set_color(COLORS['border'])
        self.ax2.set_visible(False)

        # Crosshair elements
        self.crosshair_v = self.ax.axvline(x=0, color=COLORS['crosshair'],
                                            linewidth=0.5, linestyle='--', visible=False)
        self.crosshair_h = self.ax.axhline(y=0, color=COLORS['crosshair'],
                                            linewidth=0.5, linestyle='--', visible=False)
        self.crosshair_text = self.ax.text(0, 0, '', fontsize=8, color=COLORS['text_bright'],
                                            bbox=dict(boxstyle='round,pad=0.3',
                                                      facecolor=COLORS['bg'],
                                                      edgecolor=COLORS['border'],
                                                      alpha=0.9),
                                            visible=False, zorder=10)

        # Legend
        self._update_legend()

        # Embed in tkinter
        graph_frame = tk.Frame(parent, bg=COLORS['bg'])
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 0))

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()

        # Navigation toolbar
        toolbar_bg = '#b0b0b0'
        toolbar_container = tk.Frame(graph_frame, bg=toolbar_bg)
        toolbar_container.pack(fill=tk.X, side=tk.BOTTOM)
        self.nav_toolbar = NavigationToolbar2Tk(self.canvas, toolbar_container)
        self.nav_toolbar.configure(bg=toolbar_bg)
        self.nav_toolbar.update()
        for child in self.nav_toolbar.winfo_children():
            try:
                child.configure(bg=toolbar_bg, fg='#1e1e1e')
            except tk.TclError:
                try:
                    child.configure(bg=toolbar_bg)
                except tk.TclError:
                    pass
        for attr in ('_message_label', 'message'):
            label = getattr(self.nav_toolbar, attr, None)
            if label is not None:
                try:
                    label.configure(fg='#1e1e1e', bg=toolbar_bg)
                except (tk.TclError, AttributeError):
                    pass
                break

        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig.tight_layout()
        self.fig.subplots_adjust(right=0.88)

        # Connect mouse events
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('axes_leave_event', self.on_mouse_leave)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self._pan_start_px = None
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)

    def create_stats_panel(self, parent):
        """Create bottom stats panel with key metrics."""
        stats_outer = tk.Frame(parent, bg=COLORS['border'], pady=1)
        stats_outer.pack(fill=tk.X, padx=8, pady=(2, 6))

        stats_frame = tk.Frame(stats_outer, bg=COLORS['bg_panel'], pady=4)
        stats_frame.pack(fill=tk.X, padx=1)

        self.stat_labels = {}

        stats_config = [
            ('run_name', 'Run', '--'),
            ('epoch', 'Epoch', '--'),
            ('l1_cur', 'Loss Current', '--'),
            ('l1_best', 'Loss Best', '--'),
            ('l1_best_ep', 'Best @ Epoch', '--'),
            ('val_l1_cur', 'Val Loss', '--'),
            ('val_l1_best', 'Val Best', '--'),
            ('lpips_cur', 'LPIPS Current', '--'),
            ('lpips_best', 'LPIPS Best', '--'),
            ('psnr', 'PSNR (dB)', '--'),
            ('ssim', 'SSIM', '--'),
            ('points', 'Data Points', '0'),
            ('rate', 'Step Time', '--'),
        ]

        for key, label_text, default in stats_config:
            box = tk.Frame(stats_frame, bg=COLORS['bg_panel'])
            box.pack(side=tk.LEFT, padx=8, expand=True)

            tk.Label(box, text=label_text, bg=COLORS['bg_panel'], fg=COLORS['text_dim'],
                     font=('Segoe UI', 7)).pack()
            val_label = tk.Label(box, text=default, bg=COLORS['bg_panel'],
                                 fg=COLORS['text_bright'], font=('Consolas', 10, 'bold'))
            val_label.pack()
            self.stat_labels[key] = val_label

    def create_analysis_panel(self, parent):
        """Create training analysis panel with trend indicators."""
        analysis_outer = tk.Frame(parent, bg=COLORS['border'], pady=1)
        analysis_outer.pack(fill=tk.X, padx=8, pady=(2, 0))

        analysis_frame = tk.Frame(analysis_outer, bg=COLORS['bg_panel'], pady=4)
        analysis_frame.pack(fill=tk.X, padx=1)

        tk.Label(analysis_frame, text="Analysis:", bg=COLORS['bg_panel'],
                 fg=COLORS['text_dim'], font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=(8, 12))

        self.analysis_labels = {}

        analysis_items = [
            ('trend', 'Trend'),
            ('improvement', 'Recent Change'),
            ('plateau', 'Plateau Check'),
            ('recommendation', 'Status'),
        ]

        for key, label_text in analysis_items:
            box = tk.Frame(analysis_frame, bg=COLORS['bg_panel'])
            box.pack(side=tk.LEFT, padx=12, expand=True)

            tk.Label(box, text=label_text, bg=COLORS['bg_panel'], fg=COLORS['text_dim'],
                     font=('Segoe UI', 7)).pack(side=tk.LEFT, padx=(0, 4))
            val_label = tk.Label(box, text="--", bg=COLORS['bg_panel'],
                                 fg=COLORS['text'], font=('Consolas', 9, 'bold'))
            val_label.pack(side=tk.LEFT)
            self.analysis_labels[key] = val_label

    def analyze_training(self):
        """Analyze training progress for the active run."""
        if not hasattr(self, 'analysis_labels'):
            return
        run = self.runs[self.active_run_idx] if self.active_run_idx < len(self.runs) else None
        if not run or not run.steps:
            return

        steps = list(run.steps)
        losses = list(run.l1_losses)
        current_epoch = steps[-1]

        if current_epoch < 5:
            for key in self.analysis_labels:
                self.analysis_labels[key].configure(text="Collecting...", fg=COLORS['text_dim'])
            return

        window_epochs = min(20, current_epoch * 0.5)
        cutoff = current_epoch - window_epochs
        window_steps = []
        window_losses = []
        for s, l in zip(steps, losses):
            if s >= cutoff:
                window_steps.append(s)
                window_losses.append(l)

        if len(window_steps) < 10:
            for key in self.analysis_labels:
                self.analysis_labels[key].configure(text="Collecting...", fg=COLORS['text_dim'])
            return

        smoothed = []
        alpha = 0.95
        last = window_losses[0]
        for v in window_losses:
            last = alpha * last + (1 - alpha) * v
            smoothed.append(last)

        n = len(smoothed)
        x_vals = window_steps
        y_vals = smoothed
        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n
        ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
        ss_xx = sum((x - x_mean) ** 2 for x in x_vals)

        slope = ss_xy / ss_xx if ss_xx > 0 else 0

        current_smooth = smoothed[-1]
        relative_slope = slope / current_smooth if current_smooth > 0 else 0

        if relative_slope < -0.005:
            self.analysis_labels['trend'].configure(text="Improving", fg=COLORS['good'])
        elif relative_slope > 0.005:
            self.analysis_labels['trend'].configure(text="Diverging", fg=COLORS['bad'])
        else:
            self.analysis_labels['trend'].configure(text="Flat", fg=COLORS['warn'])

        mid = len(smoothed) // 2
        first_half_avg = sum(smoothed[:mid]) / mid if mid > 0 else 0
        second_half_avg = sum(smoothed[mid:]) / (len(smoothed) - mid) if (len(smoothed) - mid) > 0 else 0

        pct_change = ((second_half_avg - first_half_avg) / first_half_avg) * 100 if first_half_avg > 0 else 0

        if pct_change < -1:
            change_color = COLORS['good']
        elif pct_change > 1:
            change_color = COLORS['bad']
        else:
            change_color = COLORS['warn']
        self.analysis_labels['improvement'].configure(text=f"{pct_change:+.1f}%", fg=change_color)

        epochs_since_best = current_epoch - run.best_l1_epoch
        if epochs_since_best < 10:
            self.analysis_labels['plateau'].configure(
                text=f"Best {epochs_since_best:.0f}ep ago", fg=COLORS['good'])
        elif epochs_since_best < 30:
            self.analysis_labels['plateau'].configure(
                text=f"Best {epochs_since_best:.0f}ep ago", fg=COLORS['warn'])
        else:
            self.analysis_labels['plateau'].configure(
                text=f"Best {epochs_since_best:.0f}ep ago", fg=COLORS['bad'])

        if relative_slope < -0.005 and epochs_since_best < 20:
            self.analysis_labels['recommendation'].configure(text="Training well", fg=COLORS['good'])
        elif relative_slope > 0.01:
            self.analysis_labels['recommendation'].configure(text="Diverging - check LR", fg=COLORS['bad'])
        elif epochs_since_best > 50:
            self.analysis_labels['recommendation'].configure(text="Consider stopping", fg=COLORS['bad'])
        elif epochs_since_best > 30 or abs(relative_slope) < 0.001:
            self.analysis_labels['recommendation'].configure(text="Plateau - may stop", fg=COLORS['warn'])
        elif relative_slope < -0.001:
            self.analysis_labels['recommendation'].configure(text="Slow progress", fg=COLORS['warn'])
        else:
            self.analysis_labels['recommendation'].configure(text="Stable", fg=COLORS['text'])

    def create_fallback_display(self, parent):
        """Create text-based display if matplotlib not available."""
        fallback_frame = tk.Frame(parent, bg=COLORS['bg'])
        fallback_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(fallback_frame, text="matplotlib not installed - showing text output",
                 bg=COLORS['bg'], fg=COLORS['text'], font=('Segoe UI', 12)).pack(pady=10)

        self.text_display = tk.Text(fallback_frame, bg=COLORS['bg'], fg=COLORS['text'],
                                     font=('Consolas', 10), state=tk.DISABLED)
        self.text_display.pack(fill=tk.BOTH, expand=True)

    # ─── Keyboard shortcuts ───

    def on_key_press(self, event):
        key = event.keysym.lower()
        if key in ('h', 'home'):
            self.set_zoom('all')
        elif key == '1':
            self.set_zoom('last_1')
        elif key == '2':
            self.set_zoom('last_5')
        elif key == '3':
            self.set_zoom('last_10')
        elif key == '4':
            self.set_zoom('last_20')
        elif key == 'l':
            self.toggle_log_scale()
        elif key == 'r':
            self.toggle_raw()
        elif key == 'g':
            self.toggle_grid()
        elif key == 's':
            self.cycle_smoothing()

    # ─── Zoom / view controls ───

    def set_zoom(self, mode):
        self.zoom_mode = mode
        self._highlight_zoom_button()
        if any(r.steps for r in self.runs):
            self.update_graph()

    def toggle_log_scale(self):
        self.log_scale = not self.log_scale
        self._update_toggle_btn(self.log_btn, self.log_scale)
        if HAS_MATPLOTLIB:
            scale = 'log' if self.log_scale else 'linear'
            self.ax.set_yscale(scale)
            any_lpips = any(r.has_lpips for r in self.runs)
            if any_lpips:
                self.ax2.set_yscale(scale)
            if any(r.steps for r in self.runs):
                self.update_graph()

    def toggle_raw(self):
        self.show_raw = not self.show_raw
        self._update_toggle_btn(self.raw_btn, self.show_raw)
        if HAS_MATPLOTLIB:
            for run in self.runs:
                run.l1_raw_line.set_visible(self.show_raw)
                run.lpips_raw_line.set_visible(self.show_raw and run.has_lpips)
                run.val_l1_raw_line.set_visible(self.show_raw and run.has_val_data)
            self.canvas.draw_idle()

    def toggle_grid(self):
        self.show_grid = not self.show_grid
        self._update_toggle_btn(self.grid_btn, self.show_grid)
        if HAS_MATPLOTLIB:
            self.ax.grid(self.show_grid, color=COLORS['grid'], linestyle='-',
                         linewidth=0.5, alpha=0.5)
            self.canvas.draw_idle()

    def cycle_smoothing(self):
        presets = [0.0, 0.5, 0.9, 0.95, 0.99]
        current = self.smoothing.get()
        for p in presets:
            if p > current + 0.005:
                self.smoothing.set(p)
                self.smooth_label.configure(text=f"{p:.2f}")
                if any(r.steps for r in self.runs):
                    self.update_graph()
                return
        self.smoothing.set(0.0)
        self.smooth_label.configure(text="0.00")
        if any(r.steps for r in self.runs):
            self.update_graph()

    # ─── Mouse interaction ───

    def on_mouse_move(self, event):
        if not HAS_MATPLOTLIB or not event.inaxes:
            self.on_mouse_leave(event)
            return

        if getattr(self, '_pan_start_px', None) is not None:
            self.on_mmb_drag(event)
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Find nearest data point across the active run
        run = self.runs[self.active_run_idx] if self.active_run_idx < len(self.runs) else None
        if not run or not run.steps:
            return

        steps_list = list(run.steps)
        l1_list = list(run.l1_losses)
        idx = self._find_nearest_idx(steps_list, x)
        if idx is None:
            return

        ep = steps_list[idx]
        l1_val = l1_list[idx]
        lpips_val = run.lpips_losses[idx] if run.has_lpips and idx < len(run.lpips_losses) else None

        self.crosshair_v.set_xdata([ep])
        self.crosshair_h.set_ydata([l1_val])
        self.crosshair_v.set_visible(True)
        self.crosshair_h.set_visible(True)

        text = f"Ep {ep:.2f}  {run.loss_label}: {l1_val:.5f}"
        if lpips_val is not None:
            text += f"  LPIPS: {lpips_val:.5f}"

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        tx = ep + x_range * 0.02
        ty = l1_val + y_range * 0.05
        ha = 'left'
        if ep > xlim[0] + x_range * 0.7:
            tx = ep - x_range * 0.02
            ha = 'right'

        self.crosshair_text.set_position((tx, ty))
        self.crosshair_text.set_text(text)
        self.crosshair_text.set_ha(ha)
        self.crosshair_text.set_visible(True)

        self.canvas.draw_idle()

    def on_mouse_leave(self, event):
        if not HAS_MATPLOTLIB:
            return
        self.crosshair_v.set_visible(False)
        self.crosshair_h.set_visible(False)
        self.crosshair_text.set_visible(False)
        self.canvas.draw_idle()

    def on_scroll(self, event):
        if not event.inaxes or not any(r.steps for r in self.runs):
            return

        scale_factor = 0.8 if event.button == 'up' else 1.25

        px, py = event.x, event.y
        xdata, _ = self.ax.transData.inverted().transform((px, py))

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        new_width = (xlim[1] - xlim[0]) * scale_factor
        rel_x = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        self.ax.set_xlim(xdata - new_width * rel_x, xdata + new_width * (1 - rel_x))

        if not self.log_scale:
            _, ydata_ax = self.ax.transData.inverted().transform((px, py))
            new_height = (ylim[1] - ylim[0]) * scale_factor
            rel_y = (ydata_ax - ylim[0]) / (ylim[1] - ylim[0])
            self.ax.set_ylim(ydata_ax - new_height * rel_y, ydata_ax + new_height * (1 - rel_y))

        if self.ax2.get_visible() and not self.log_scale:
            y2lim = self.ax2.get_ylim()
            _, ydata_ax2 = self.ax2.transData.inverted().transform((px, py))
            new_h2 = (y2lim[1] - y2lim[0]) * scale_factor
            rel_y2 = (ydata_ax2 - y2lim[0]) / (y2lim[1] - y2lim[0])
            self.ax2.set_ylim(ydata_ax2 - new_h2 * rel_y2, ydata_ax2 + new_h2 * (1 - rel_y2))

        self.zoom_mode = 'custom'
        self._highlight_zoom_button()
        self.canvas.draw_idle()

    def on_button_press(self, event):
        if event.button == 2 and event.inaxes and any(r.steps for r in self.runs):
            self._pan_start_px = (event.x, event.y)
            self._pan_start_xlim = self.ax.get_xlim()
            self._pan_start_ylim = self.ax.get_ylim()
            self._pan_start_y2lim = self.ax2.get_ylim() if self.ax2.get_visible() else None

    def on_button_release(self, event):
        if event.button == 2:
            self._pan_start_px = None

    def on_mmb_drag(self, event):
        if self._pan_start_px is None or event.x is None or event.y is None:
            return
        dpx = self._pan_start_px[0] - event.x
        dpy = self._pan_start_px[1] - event.y

        inv = self.ax.transData.inverted()
        origin = inv.transform((0, 0))
        delta = inv.transform((dpx, dpy))
        dx = delta[0] - origin[0]
        dy = -(delta[1] - origin[1])

        x0, x1 = self._pan_start_xlim
        self.ax.set_xlim(x0 + dx, x1 + dx)
        if not self.log_scale:
            y0, y1 = self._pan_start_ylim
            self.ax.set_ylim(y0 + dy, y1 + dy)

        if self._pan_start_y2lim is not None and not self.log_scale:
            inv2 = self.ax2.transData.inverted()
            origin2 = inv2.transform((0, 0))
            delta2 = inv2.transform((0, dpy))
            dy2 = -(delta2[1] - origin2[1])
            y2_0, y2_1 = self._pan_start_y2lim
            self.ax2.set_ylim(y2_0 + dy2, y2_1 + dy2)

        self.zoom_mode = 'custom'
        self._highlight_zoom_button()
        self.canvas.draw_idle()

    def _find_nearest_idx(self, steps, target):
        if not steps:
            return None
        lo, hi = 0, len(steps) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if steps[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        if lo > 0 and abs(steps[lo - 1] - target) < abs(steps[lo] - target):
            return lo - 1
        return lo

    # ─── File monitoring ───

    def toggle_monitoring(self):
        if self.is_monitoring:
            self.stop_monitoring()
        else:
            self.start_monitoring()

    def start_monitoring(self):
        if not self.runs:
            return
        self.is_monitoring = True
        self.monitor_btn.configure(text="Stop", bg=COLORS['btn_active'])
        self.monitor_btn._is_active = True
        # Read existing content first
        for run in self.runs:
            self.read_log_file(run, full_read=True)
        self.update_graph()
        self.update_stats()
        self.update_loop()

    def stop_monitoring(self):
        self.is_monitoring = False
        self.monitor_btn.configure(text="Start", bg=COLORS['btn_normal'])
        self.monitor_btn._is_active = False

    def read_log_file(self, run, full_read=False):
        log_path = run.log_file
        if not log_path or not os.path.exists(log_path):
            return False

        try:
            current_modified = os.path.getmtime(log_path)
            if not full_read and current_modified == run.last_modified:
                return False
            run.last_modified = current_modified

            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                if full_read:
                    run.last_position = 0
                    run.clear()

                f.seek(run.last_position)
                new_content = f.read()
                run.last_position = f.tell()

            if new_content:
                return self.parse_log_content(run, new_content)

        except Exception:
            pass
        return False

    def parse_log_content(self, run, content):
        pattern = r'Epoch\[(\d+)\]\s*Step\[(\d+)\].*?\b(L1|L2|BCE\+Dice):([\d.]+)'
        lpips_pattern = r'LPIPS:([\d.]+)'
        time_pattern = r'T/Step:([\d.]+)s'
        val_pattern = r'Val Epoch\[(\d+)\]\s*Step\[(\d+)\].*?Val_(L1|L2|BCE\+Dice):([\d.]+)'
        val_lpips_pattern = r'Val_LPIPS:([\d.]+)'
        val_psnr_pattern = r'PSNR:([\d.]+)dB'
        val_ssim_pattern = r'SSIM:([\d.]+)'

        lines = content.split('\n')
        new_data = False

        for line in lines:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                detected_label = match.group(3)
                l1_loss = float(match.group(4))

                if not run.steps and detected_label != run.loss_label:
                    run.loss_label = detected_label

                lpips_match = re.search(lpips_pattern, line)
                lpips_loss = float(lpips_match.group(1)) if lpips_match else None

                time_match = re.search(time_pattern, line)
                if time_match:
                    run.time_per_step.append(float(time_match.group(1)))

                iter_match = re.search(r'\((\d+)/(\d+)\)', line)
                if iter_match:
                    step_in_epoch = int(iter_match.group(1))
                    total_steps = int(iter_match.group(2))
                    x_value = epoch - 1 + (step_in_epoch / total_steps)
                else:
                    x_value = epoch

                run.steps.append(x_value)
                run.l1_losses.append(l1_loss)

                if l1_loss < run.best_l1:
                    run.best_l1 = l1_loss
                    run.best_l1_epoch = x_value

                if lpips_loss is not None:
                    run.lpips_losses.append(lpips_loss)
                    run.has_lpips = True
                    if lpips_loss < run.best_lpips:
                        run.best_lpips = lpips_loss
                        run.best_lpips_epoch = x_value
                elif run.has_lpips:
                    run.lpips_losses.append(run.lpips_losses[-1] if run.lpips_losses else 0)

                new_data = True
                continue

            val_match = re.search(val_pattern, line)
            if val_match:
                val_epoch = int(val_match.group(1))
                val_l1_loss = float(val_match.group(4))
                val_x_value = val_epoch
                run.val_steps.append(val_x_value)
                run.val_l1_losses.append(val_l1_loss)
                run.has_val_data = True
                if val_l1_loss < run.best_val_l1:
                    run.best_val_l1 = val_l1_loss
                    run.best_val_l1_epoch = val_x_value
                val_lp_match = re.search(val_lpips_pattern, line)
                if val_lp_match:
                    run.val_lpips_losses.append(float(val_lp_match.group(1)))
                psnr_match = re.search(val_psnr_pattern, line)
                if psnr_match:
                    run.val_psnr.append(float(psnr_match.group(1)))
                ssim_match = re.search(val_ssim_pattern, line)
                if ssim_match:
                    run.val_ssim.append(float(ssim_match.group(1)))
                new_data = True

        return new_data

    # ─── Smoothing ───

    def apply_smoothing(self, values):
        if not values:
            return []
        alpha = self.smoothing.get()
        if alpha <= 0:
            return list(values)

        smoothed = []
        last = values[0]
        for v in values:
            last = alpha * last + (1 - alpha) * v
            smoothed.append(last)
        return smoothed

    # ─── Graph update ───

    def _update_legend(self):
        lines = []
        labels = []
        for run in self.runs:
            if run.l1_line is not None:
                lines.append(run.l1_line)
                labels.append(run.name)

        if lines:
            self.ax.legend(lines, labels, loc='upper right',
                           facecolor=COLORS['bg_light'], edgecolor=COLORS['border'],
                           labelcolor=COLORS['text'], fontsize=8)
        elif hasattr(self.ax, 'get_legend') and self.ax.get_legend():
            self.ax.get_legend().remove()

    def update_graph(self):
        if not HAS_MATPLOTLIB:
            self.update_text_display()
            return

        if not any(r.steps for r in self.runs):
            return

        # Compute global view range across all runs
        all_steps = []
        for run in self.runs:
            if run.steps:
                all_steps.extend([run.steps[0], run.steps[-1]])
        if not all_steps:
            return

        global_min = min(all_steps)
        global_max = max(all_steps)
        x_min, x_max = self._get_view_range(global_min, global_max)

        any_lpips = False
        all_y_vals = []

        for run in self.runs:
            if not run.steps:
                # Hide lines for empty runs
                if run.l1_line:
                    run.l1_line.set_data([], [])
                    run.l1_raw_line.set_data([], [])
                continue

            steps = list(run.steps)
            l1_raw = list(run.l1_losses)
            l1_smooth = self.apply_smoothing(run.l1_losses)

            view_mask = [x_min <= s <= x_max for s in steps]
            view_l1_smooth = [v for v, m in zip(l1_smooth, view_mask) if m]
            view_l1_raw = [v for v, m in zip(l1_raw, view_mask) if m]

            # Update L1 lines
            run.l1_raw_line.set_data(steps, l1_raw)
            run.l1_raw_line.set_visible(self.show_raw)
            run.l1_line.set_data(steps, l1_smooth)

            # Collect Y values for axis scaling
            if view_l1_smooth:
                all_y_vals.extend(view_l1_smooth)
            if self.show_raw and view_l1_raw:
                all_y_vals.extend(view_l1_raw)

            # LPIPS
            if run.has_lpips and run.lpips_losses:
                any_lpips = True
                lpips_raw = list(run.lpips_losses)
                lpips_smooth = self.apply_smoothing(run.lpips_losses)
                run.lpips_raw_line.set_data(steps, lpips_raw)
                run.lpips_raw_line.set_visible(self.show_raw)
                run.lpips_line.set_data(steps, lpips_smooth)
                run.lpips_line.set_visible(True)
            else:
                run.lpips_line.set_visible(False)
                run.lpips_raw_line.set_visible(False)

            # Validation lines
            if run.has_val_data and run.val_steps:
                val_steps = list(run.val_steps)
                val_l1_raw = list(run.val_l1_losses)
                val_l1_smooth = self.apply_smoothing(run.val_l1_losses)
                run.val_l1_raw_line.set_data(val_steps, val_l1_raw)
                run.val_l1_raw_line.set_visible(self.show_raw)
                run.val_l1_line.set_data(val_steps, val_l1_smooth)
                run.val_l1_line.set_visible(True)
                # Include in Y range
                val_in_view = [v for s, v in zip(val_steps, val_l1_raw) if x_min <= s <= x_max]
                if val_in_view:
                    all_y_vals.extend(val_in_view)
            else:
                run.val_l1_line.set_visible(False)
                run.val_l1_raw_line.set_visible(False)

            # Best marker — only show for active run to avoid clutter
            is_active = (self.runs.index(run) == self.active_run_idx)
            if is_active and run.best_l1 < float('inf'):
                run.best_l1_marker.set_data([run.best_l1_epoch], [run.best_l1])
                run.best_l1_annot.set_text(f" Best: {run.best_l1:.5f}")
                run.best_l1_annot.xy = (run.best_l1_epoch, run.best_l1)
                run.best_l1_annot.set_visible(x_min <= run.best_l1_epoch <= x_max)
            else:
                run.best_l1_marker.set_data([], [])
                run.best_l1_annot.set_visible(False)

            if run.has_lpips and run.best_lpips < float('inf'):
                run.best_lpips_marker.set_data([run.best_lpips_epoch], [run.best_lpips])
            else:
                run.best_lpips_marker.set_data([], [])

        # Show/hide LPIPS axis
        self.ax2.set_visible(any_lpips)

        # Set axis limits
        if self.zoom_mode != 'custom':
            self.ax.set_xlim(x_min - (x_max - x_min) * 0.01,
                             x_max + (x_max - x_min) * 0.01)

            if all_y_vals and not self.log_scale:
                y_min, y_max = min(all_y_vals), max(all_y_vals)
                margin = (y_max - y_min) * 0.1 or 0.01
                self.ax.set_ylim(max(0, y_min - margin), y_max + margin)
            elif self.log_scale:
                self.ax.relim()
                self.ax.autoscale_view()

        self._update_legend()
        self.canvas.draw_idle()

    def _get_view_range(self, total_min, total_max):
        if self.zoom_mode == 'all':
            return total_min, total_max
        elif self.zoom_mode == 'custom':
            xlim = self.ax.get_xlim()
            return xlim[0], xlim[1]

        epoch_counts = {'last_1': 1, 'last_5': 5, 'last_10': 10, 'last_20': 20}
        n_epochs = epoch_counts.get(self.zoom_mode, 10)
        return max(total_min, total_max - n_epochs), total_max

    def update_stats(self):
        run = self.runs[self.active_run_idx] if self.active_run_idx < len(self.runs) else None
        if not run or not run.steps:
            # Reset all
            for key in self.stat_labels:
                self.stat_labels[key].configure(text='--')
            self.stat_labels['points'].configure(text='0')
            return

        self.stat_labels['run_name'].configure(text=run.name[:15], fg=run.color)

        ep = run.steps[-1]
        l1 = run.l1_losses[-1]

        self.stat_labels['epoch'].configure(text=f"{ep:.2f}")
        self.stat_labels['l1_cur'].configure(text=f"{l1:.5f}")
        self.stat_labels['points'].configure(text=f"{len(run.steps):,}")

        if run.best_l1 < float('inf'):
            self.stat_labels['l1_best'].configure(text=f"{run.best_l1:.5f}")
            self.stat_labels['l1_best_ep'].configure(text=f"{run.best_l1_epoch:.1f}")

        if run.has_val_data and run.val_l1_losses:
            self.stat_labels['val_l1_cur'].configure(text=f"{run.val_l1_losses[-1]:.5f}")
            if run.best_val_l1 < float('inf'):
                self.stat_labels['val_l1_best'].configure(text=f"{run.best_val_l1:.5f}")
        else:
            self.stat_labels['val_l1_cur'].configure(text='--')
            self.stat_labels['val_l1_best'].configure(text='--')

        if run.has_lpips and run.lpips_losses:
            self.stat_labels['lpips_cur'].configure(text=f"{run.lpips_losses[-1]:.5f}")
            if run.best_lpips < float('inf'):
                self.stat_labels['lpips_best'].configure(text=f"{run.best_lpips:.5f}")
        else:
            self.stat_labels['lpips_cur'].configure(text='--')
            self.stat_labels['lpips_best'].configure(text='--')

        if run.val_psnr:
            self.stat_labels['psnr'].configure(text=f"{run.val_psnr[-1]:.2f}")
        else:
            self.stat_labels['psnr'].configure(text='--')
        if run.val_ssim:
            self.stat_labels['ssim'].configure(text=f"{run.val_ssim[-1]:.4f}")
        else:
            self.stat_labels['ssim'].configure(text='--')

        if run.time_per_step:
            avg_time = sum(run.time_per_step) / len(run.time_per_step)
            self.stat_labels['rate'].configure(text=f"{avg_time:.3f}s")
        else:
            self.stat_labels['rate'].configure(text='--')

        self.analyze_training()

    def update_text_display(self):
        if not hasattr(self, 'text_display'):
            return
        run = self.runs[self.active_run_idx] if self.active_run_idx < len(self.runs) else None
        if not run or not run.steps:
            return

        self.text_display.configure(state=tk.NORMAL)
        self.text_display.delete(1.0, tk.END)

        n = min(50, len(run.steps))
        self.text_display.insert(tk.END, f"Run: {run.name} — Last {n} data points:\n\n")
        self.text_display.insert(tk.END, f"{'Epoch':<10} {'Loss':<14} {'LPIPS':<14}\n")
        self.text_display.insert(tk.END, "-" * 40 + "\n")

        for i in range(-n, 0):
            epoch = run.steps[i]
            l1 = run.l1_losses[i]
            lpips = run.lpips_losses[i] if run.has_lpips and len(run.lpips_losses) > abs(i) else None
            line = f"{epoch:<10.2f} {l1:<14.6f}"
            if lpips is not None:
                line += f" {lpips:<14.6f}"
            self.text_display.insert(tk.END, line + "\n")

        self.text_display.configure(state=tk.DISABLED)

    def clear_all_data(self):
        for run in self.runs:
            run.clear()
            if HAS_MATPLOTLIB:
                run.l1_line.set_data([], [])
                run.l1_raw_line.set_data([], [])
                run.lpips_line.set_data([], [])
                run.lpips_raw_line.set_data([], [])
                run.val_l1_line.set_data([], [])
                run.val_l1_raw_line.set_data([], [])
                run.best_l1_marker.set_data([], [])
                run.best_lpips_marker.set_data([], [])
                run.best_l1_annot.set_visible(False)

        if HAS_MATPLOTLIB:
            self.canvas.draw_idle()

        if hasattr(self, 'stat_labels'):
            for key in self.stat_labels:
                self.stat_labels[key].configure(text='--')
            self.stat_labels['points'].configure(text='0')

        if hasattr(self, 'analysis_labels'):
            for key in self.analysis_labels:
                self.analysis_labels[key].configure(text='--', fg=COLORS['text'])

    def update_loop(self):
        if not self.is_monitoring:
            return
        any_new = False
        for run in self.runs:
            if self.read_log_file(run):
                any_new = True
        if any_new:
            self.update_graph()
            self.update_stats()
        self.root.after(self.update_interval, self.update_loop)


def main():
    parser = argparse.ArgumentParser(description='TuNet Training Monitor')
    parser.add_argument('--log_file', type=str, help='Path to training.log file')
    parser.add_argument('--output_dir', type=str, help='Path to output directory containing training.log')
    parser.add_argument('--data_dir', type=str, help='Parent directory to scan for multiple training runs')
    args = parser.parse_args()

    log_file = None
    data_dir = None

    if args.log_file:
        log_file = args.log_file
    elif args.output_dir:
        log_file = os.path.join(args.output_dir, 'training.log')

    if args.data_dir:
        data_dir = args.data_dir

    if not HAS_MATPLOTLIB:
        print("\n" + "=" * 60)
        print("WARNING: matplotlib is not installed!")
        print("For the best experience, install it with:")
        print("  pip install matplotlib")
        print("=" * 60 + "\n")

    root = tk.Tk()

    # Set dark theme
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('.', background=COLORS['bg'], foreground=COLORS['text'])
    style.configure('TFrame', background=COLORS['bg'])
    style.configure('TLabel', background=COLORS['bg'], foreground=COLORS['text'])
    style.configure('TButton', background=COLORS['btn_normal'], foreground=COLORS['text'])
    style.configure('TEntry', fieldbackground=COLORS['bg_light'], foreground=COLORS['text'])
    style.configure('TScale', background=COLORS['bg'], troughcolor=COLORS['bg_light'])

    app = TrainingMonitor(root, log_file, data_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
