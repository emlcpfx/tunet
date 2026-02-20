"""
TuNet Training Monitor - Real-time loss graph visualization

Usage:
    python training_monitor.py --log_file path/to/training.log
    python training_monitor.py --output_dir path/to/output  (will find training.log there)

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
from tkinter import ttk, filedialog
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


# Dark theme colors
COLORS = {
    'bg': '#1e1e1e',
    'bg_light': '#2d2d2d',
    'bg_panel': '#252525',
    'border': '#555555',
    'grid': '#404040',
    'text': '#cccccc',
    'text_bright': '#ffffff',
    'text_dim': '#888888',
    'l1': '#ff6b6b',
    'l1_raw': '#ff6b6b',
    'lpips': '#4ecdc4',
    'lpips_raw': '#4ecdc4',
    'accent': '#61afef',
    'best_marker': '#98c379',
    'crosshair': '#888888',
    'btn_active': '#3a6ea5',
    'btn_normal': '#3d3d3d',
}


class TrainingMonitor:
    def __init__(self, root, log_file=None):
        self.root = root
        self.root.title("TuNet Training Monitor")
        self.root.geometry("1100x750")
        self.root.minsize(800, 500)
        self.root.configure(bg=COLORS['bg'])

        # Data storage
        self.max_points = 50000
        self.steps = deque(maxlen=self.max_points)
        self.l1_losses = deque(maxlen=self.max_points)
        self.lpips_losses = deque(maxlen=self.max_points)
        self.has_lpips = False

        # Timing data
        self.epoch_start_times = {}  # epoch_num -> first_seen_time
        self.time_per_step = deque(maxlen=1000)

        # File monitoring
        self.log_file = tk.StringVar(value=log_file or "")
        self.last_position = 0
        self.last_modified = 0
        self.is_monitoring = False

        # UI state
        self.smoothing = tk.DoubleVar(value=0.6)
        self.update_interval = 1000
        self.log_scale = False
        self.show_raw = True
        self.show_grid = True
        self.zoom_mode = 'all'  # 'all', 'last_1', 'last_5', 'last_10', 'last_20'
        self.crosshair_visible = False

        # Best loss tracking
        self.best_l1 = float('inf')
        self.best_l1_epoch = 0
        self.best_lpips = float('inf')
        self.best_lpips_epoch = 0

        self.create_widgets()

        # Bind keyboard shortcuts
        self.root.bind('<Key>', self.on_key_press)

        # Start monitoring if log file provided
        if log_file and os.path.exists(log_file):
            self.start_monitoring()

    def create_widgets(self):
        # ─── Top control bar ───
        control_frame = tk.Frame(self.root, bg=COLORS['bg'], pady=4)
        control_frame.pack(fill=tk.X, padx=8)

        tk.Label(control_frame, text="Log:", bg=COLORS['bg'], fg=COLORS['text'],
                 font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(4, 2))
        log_entry = tk.Entry(control_frame, textvariable=self.log_file, width=50,
                             bg=COLORS['bg_light'], fg=COLORS['text'], insertbackground='white',
                             relief='flat', font=('Consolas', 9))
        log_entry.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        self._make_button(control_frame, "Browse", self.browse_log).pack(side=tk.LEFT, padx=2)
        self.monitor_btn = self._make_button(control_frame, "Start", self.toggle_monitoring)
        self.monitor_btn.pack(side=tk.LEFT, padx=2)

        # ─── Toolbar row: zoom presets + toggles + smoothing ───
        toolbar_frame = tk.Frame(self.root, bg=COLORS['bg_panel'], pady=3)
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
        self._make_button(toolbar_frame, "Clear", self.clear_data, small=True).pack(
            side=tk.RIGHT, padx=4)

        # ─── Graph area ───
        if HAS_MATPLOTLIB:
            self.create_matplotlib_graph()
        else:
            self.create_fallback_display()

        # ─── Stats panel at bottom ───
        self.create_stats_panel()

    def _make_button(self, parent, text, command, small=False):
        font = ('Segoe UI', 8) if small else ('Segoe UI', 9)
        padx = 6 if small else 10
        pady = 1 if small else 3
        btn = tk.Label(parent, text=text, bg=COLORS['btn_normal'], fg=COLORS['text'],
                       font=font, padx=padx, pady=pady, cursor='hand2', relief='flat')
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
        if self.steps:
            self.update_graph()

    def create_matplotlib_graph(self):
        """Create matplotlib figure with dual Y-axes."""
        self.fig = Figure(figsize=(10, 5), dpi=100, facecolor=COLORS['bg'])

        # Primary axis (L1)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(COLORS['bg_light'])
        self.ax.set_xlabel('Epoch', color=COLORS['text'], fontsize=10)
        self.ax.set_ylabel('L1 Loss', color=COLORS['l1'], fontsize=10)
        self.ax.tick_params(axis='x', colors=COLORS['text'], labelsize=9)
        self.ax.tick_params(axis='y', colors=COLORS['l1'], labelsize=9)
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

        # Lines: raw (faint) + smoothed (bold)
        self.l1_raw_line, = self.ax.plot([], [], '-', color=COLORS['l1_raw'],
                                          linewidth=0.5, alpha=0.25, zorder=1)
        self.l1_line, = self.ax.plot([], [], '-', color=COLORS['l1'],
                                      linewidth=1.8, alpha=0.95, label='L1 Loss', zorder=3)

        self.lpips_raw_line, = self.ax2.plot([], [], '-', color=COLORS['lpips_raw'],
                                              linewidth=0.5, alpha=0.25, zorder=1)
        self.lpips_line, = self.ax2.plot([], [], '-', color=COLORS['lpips'],
                                          linewidth=1.8, alpha=0.95, label='LPIPS Loss', zorder=3)

        # Best loss markers (initialized empty)
        self.best_l1_marker, = self.ax.plot([], [], '*', color=COLORS['best_marker'],
                                             markersize=14, zorder=5, label='Best L1')
        self.best_lpips_marker, = self.ax2.plot([], [], '*', color=COLORS['best_marker'],
                                                 markersize=14, zorder=5)

        # Best loss annotation
        self.best_l1_annot = self.ax.annotate('', xy=(0, 0), fontsize=8,
                                               color=COLORS['best_marker'],
                                               bbox=dict(boxstyle='round,pad=0.3',
                                                         facecolor=COLORS['bg_light'],
                                                         edgecolor=COLORS['best_marker'],
                                                         alpha=0.9),
                                               zorder=6)

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
        graph_frame = tk.Frame(self.root, bg=COLORS['bg'])
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 0))

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()

        # Navigation toolbar (zoom, pan, save built-in)
        toolbar_container = tk.Frame(graph_frame, bg=COLORS['bg'])
        toolbar_container.pack(fill=tk.X, side=tk.BOTTOM)
        self.nav_toolbar = NavigationToolbar2Tk(self.canvas, toolbar_container)
        self.nav_toolbar.configure(bg=COLORS['bg'])
        self.nav_toolbar.update()
        # Style toolbar children
        for child in self.nav_toolbar.winfo_children():
            try:
                child.configure(bg=COLORS['bg'])
            except tk.TclError:
                pass

        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig.tight_layout()
        self.fig.subplots_adjust(right=0.88)

        # Connect mouse events for crosshair
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('axes_leave_event', self.on_mouse_leave)
        # Scroll zoom
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

    def create_stats_panel(self):
        """Create bottom stats panel with key metrics."""
        stats_outer = tk.Frame(self.root, bg=COLORS['border'], pady=1)
        stats_outer.pack(fill=tk.X, padx=8, pady=(2, 6))

        stats_frame = tk.Frame(stats_outer, bg=COLORS['bg_panel'], pady=4)
        stats_frame.pack(fill=tk.X, padx=1)

        # Row of stat boxes
        self.stat_labels = {}

        stats_config = [
            ('epoch', 'Epoch', '--'),
            ('l1_cur', 'L1 Current', '--'),
            ('l1_best', 'L1 Best', '--'),
            ('l1_best_ep', 'Best @ Epoch', '--'),
            ('lpips_cur', 'LPIPS Current', '--'),
            ('lpips_best', 'LPIPS Best', '--'),
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

    def create_fallback_display(self):
        """Create text-based display if matplotlib not available."""
        fallback_frame = tk.Frame(self.root, bg=COLORS['bg'])
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
        if self.steps:
            self.update_graph()

    def toggle_log_scale(self):
        self.log_scale = not self.log_scale
        self._update_toggle_btn(self.log_btn, self.log_scale)
        if HAS_MATPLOTLIB:
            scale = 'log' if self.log_scale else 'linear'
            self.ax.set_yscale(scale)
            if self.has_lpips:
                self.ax2.set_yscale(scale)
            if self.steps:
                self.update_graph()

    def toggle_raw(self):
        self.show_raw = not self.show_raw
        self._update_toggle_btn(self.raw_btn, self.show_raw)
        if HAS_MATPLOTLIB:
            self.l1_raw_line.set_visible(self.show_raw)
            self.lpips_raw_line.set_visible(self.show_raw and self.has_lpips)
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
        # Find next preset
        for p in presets:
            if p > current + 0.005:
                self.smoothing.set(p)
                self.smooth_label.configure(text=f"{p:.2f}")
                if self.steps:
                    self.update_graph()
                return
        # Wrap around
        self.smoothing.set(0.0)
        self.smooth_label.configure(text="0.00")
        if self.steps:
            self.update_graph()

    # ─── Mouse interaction ───

    def on_mouse_move(self, event):
        if not HAS_MATPLOTLIB or not event.inaxes or not self.steps:
            self.on_mouse_leave(event)
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Find nearest data point
        steps_list = list(self.steps)
        l1_list = list(self.l1_losses)
        idx = self._find_nearest_idx(steps_list, x)
        if idx is None:
            return

        ep = steps_list[idx]
        l1_val = l1_list[idx]
        lpips_val = self.lpips_losses[idx] if self.has_lpips and idx < len(self.lpips_losses) else None

        # Update crosshair
        self.crosshair_v.set_xdata([ep])
        self.crosshair_h.set_ydata([l1_val])
        self.crosshair_v.set_visible(True)
        self.crosshair_h.set_visible(True)

        # Build tooltip text
        text = f"Ep {ep:.2f}  L1: {l1_val:.5f}"
        if lpips_val is not None:
            text += f"  LPIPS: {lpips_val:.5f}"

        # Position tooltip
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        # Offset tooltip from crosshair - flip side if near edge
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
        """Scroll to zoom in/out centered on cursor position."""
        if not event.inaxes or not self.steps:
            return

        scale_factor = 0.8 if event.button == 'up' else 1.25
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata

        # Zoom X
        new_width = (xlim[1] - xlim[0]) * scale_factor
        rel_x = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        self.ax.set_xlim(xdata - new_width * rel_x, xdata + new_width * (1 - rel_x))

        # Zoom Y (primary axis only, secondary follows on redraw)
        if not self.log_scale:
            new_height = (ylim[1] - ylim[0]) * scale_factor
            rel_y = (ydata - ylim[0]) / (ylim[1] - ylim[0])
            self.ax.set_ylim(ydata - new_height * rel_y, ydata + new_height * (1 - rel_y))

        # Switch to custom zoom mode
        self.zoom_mode = 'custom'
        self._highlight_zoom_button()
        self.canvas.draw_idle()

    def _find_nearest_idx(self, steps, target):
        if not steps:
            return None
        # Binary search for nearest
        lo, hi = 0, len(steps) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if steps[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        # Check lo and lo-1
        if lo > 0 and abs(steps[lo - 1] - target) < abs(steps[lo] - target):
            return lo - 1
        return lo

    # ─── File monitoring ───

    def browse_log(self):
        initial_dir = os.path.dirname(self.log_file.get()) if self.log_file.get() else os.getcwd()
        path = filedialog.askopenfilename(
            title="Select Training Log",
            initialdir=initial_dir,
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if path:
            self.log_file.set(path)
            self.clear_data()
            if self.is_monitoring:
                self.stop_monitoring()
            self.start_monitoring()

    def toggle_monitoring(self):
        if self.is_monitoring:
            self.stop_monitoring()
        else:
            self.start_monitoring()

    def start_monitoring(self):
        log_path = self.log_file.get()
        if not log_path or not os.path.exists(log_path):
            return

        self.is_monitoring = True
        self.monitor_btn.configure(text="Stop", bg=COLORS['btn_active'])
        self.monitor_btn._is_active = True

        # Read existing content first
        self.read_log_file(full_read=True)
        self.update_loop()

    def stop_monitoring(self):
        self.is_monitoring = False
        self.monitor_btn.configure(text="Start", bg=COLORS['btn_normal'])
        self.monitor_btn._is_active = False

    def read_log_file(self, full_read=False):
        log_path = self.log_file.get()
        if not log_path or not os.path.exists(log_path):
            return

        try:
            current_modified = os.path.getmtime(log_path)
            if not full_read and current_modified == self.last_modified:
                return
            self.last_modified = current_modified

            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                if full_read:
                    self.last_position = 0
                    self.clear_data()

                f.seek(self.last_position)
                new_content = f.read()
                self.last_position = f.tell()

            if new_content:
                self.parse_log_content(new_content)

        except Exception as e:
            pass

    def parse_log_content(self, content):
        pattern = r'Epoch\[(\d+)\]\s*Step\[(\d+)\].*?L1:([\d.]+)'
        lpips_pattern = r'LPIPS:([\d.]+)'
        time_pattern = r'T/Step:([\d.]+)s'

        lines = content.split('\n')
        new_data = False

        for line in lines:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                l1_loss = float(match.group(3))

                lpips_match = re.search(lpips_pattern, line)
                lpips_loss = float(lpips_match.group(1)) if lpips_match else None

                # Parse step time
                time_match = re.search(time_pattern, line)
                if time_match:
                    self.time_per_step.append(float(time_match.group(1)))

                # X-axis: fractional epoch
                iter_match = re.search(r'\((\d+)/(\d+)\)', line)
                if iter_match:
                    step_in_epoch = int(iter_match.group(1))
                    total_steps = int(iter_match.group(2))
                    x_value = epoch - 1 + (step_in_epoch / total_steps)
                else:
                    x_value = epoch

                self.steps.append(x_value)
                self.l1_losses.append(l1_loss)

                # Track best
                if l1_loss < self.best_l1:
                    self.best_l1 = l1_loss
                    self.best_l1_epoch = x_value

                if lpips_loss is not None:
                    self.lpips_losses.append(lpips_loss)
                    self.has_lpips = True
                    if lpips_loss < self.best_lpips:
                        self.best_lpips = lpips_loss
                        self.best_lpips_epoch = x_value
                elif self.has_lpips:
                    self.lpips_losses.append(self.lpips_losses[-1] if self.lpips_losses else 0)

                new_data = True

        if new_data:
            self.update_graph()
            self.update_stats()

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
        lines = [self.l1_line]
        labels = ['L1 Loss']
        if self.has_lpips:
            lines.append(self.lpips_line)
            labels.append('LPIPS Loss')
        lines.append(self.best_l1_marker)
        labels.append('Best')

        self.ax.legend(lines, labels, loc='upper right',
                       facecolor=COLORS['bg_light'], edgecolor=COLORS['border'],
                       labelcolor=COLORS['text'], fontsize=9)

    def update_graph(self):
        if not HAS_MATPLOTLIB:
            self.update_text_display()
            return

        if not self.steps:
            return

        steps = list(self.steps)
        l1_raw = list(self.l1_losses)
        l1_smooth = self.apply_smoothing(self.l1_losses)

        # Determine view range
        x_min, x_max = self._get_view_range(steps)
        view_mask = [x_min <= s <= x_max for s in steps]

        # Filter to view range for axis scaling
        view_steps = [s for s, m in zip(steps, view_mask) if m]
        view_l1_smooth = [v for v, m in zip(l1_smooth, view_mask) if m]
        view_l1_raw = [v for v, m in zip(l1_raw, view_mask) if m]

        # Update L1 lines
        self.l1_raw_line.set_data(steps, l1_raw)
        self.l1_raw_line.set_visible(self.show_raw)
        self.l1_line.set_data(steps, l1_smooth)

        # Update LPIPS
        if self.has_lpips and self.lpips_losses:
            lpips_raw = list(self.lpips_losses)
            lpips_smooth = self.apply_smoothing(self.lpips_losses)
            self.lpips_raw_line.set_data(steps, lpips_raw)
            self.lpips_raw_line.set_visible(self.show_raw)
            self.lpips_line.set_data(steps, lpips_smooth)
            self.lpips_line.set_visible(True)
            self.ax2.set_visible(True)

            # Scale secondary axis to view range
            view_lpips_smooth = [v for v, m in zip(lpips_smooth, view_mask) if m]
            if view_lpips_smooth:
                lp_min, lp_max = min(view_lpips_smooth), max(view_lpips_smooth)
                margin = (lp_max - lp_min) * 0.1 or 0.01
                if not self.log_scale:
                    self.ax2.set_ylim(max(0, lp_min - margin), lp_max + margin)
        else:
            self.lpips_line.set_visible(False)
            self.lpips_raw_line.set_visible(False)
            self.ax2.set_visible(False)

        # Update best loss marker
        if self.best_l1 < float('inf'):
            self.best_l1_marker.set_data([self.best_l1_epoch], [self.best_l1])
            self.best_l1_annot.set_text(f" Best: {self.best_l1:.5f}")
            self.best_l1_annot.xy = (self.best_l1_epoch, self.best_l1)
            self.best_l1_annot.set_visible(x_min <= self.best_l1_epoch <= x_max)
        if self.has_lpips and self.best_lpips < float('inf'):
            self.best_lpips_marker.set_data([self.best_lpips_epoch], [self.best_lpips])

        # Set axis limits
        if self.zoom_mode != 'custom':
            self.ax.set_xlim(x_min - (x_max - x_min) * 0.01,
                             x_max + (x_max - x_min) * 0.01)

            if view_l1_smooth and not self.log_scale:
                y_vals = view_l1_smooth
                if self.show_raw:
                    y_vals = y_vals + view_l1_raw
                y_min, y_max = min(y_vals), max(y_vals)
                margin = (y_max - y_min) * 0.1 or 0.01
                self.ax.set_ylim(max(0, y_min - margin), y_max + margin)
            elif self.log_scale:
                self.ax.relim()
                self.ax.autoscale_view()

        self._update_legend()
        self.canvas.draw_idle()

    def _get_view_range(self, steps):
        if not steps:
            return 0, 1

        total_max = steps[-1]
        total_min = steps[0]

        if self.zoom_mode == 'all':
            return total_min, total_max
        elif self.zoom_mode == 'custom':
            xlim = self.ax.get_xlim()
            return xlim[0], xlim[1]

        # Last N epochs
        epoch_counts = {'last_1': 1, 'last_5': 5, 'last_10': 10, 'last_20': 20}
        n_epochs = epoch_counts.get(self.zoom_mode, 10)
        return max(total_min, total_max - n_epochs), total_max

    def update_stats(self):
        if not self.steps:
            return

        ep = self.steps[-1]
        l1 = self.l1_losses[-1]

        self.stat_labels['epoch'].configure(text=f"{ep:.2f}")
        self.stat_labels['l1_cur'].configure(text=f"{l1:.5f}")
        self.stat_labels['points'].configure(text=f"{len(self.steps):,}")

        if self.best_l1 < float('inf'):
            self.stat_labels['l1_best'].configure(text=f"{self.best_l1:.5f}")
            self.stat_labels['l1_best_ep'].configure(text=f"{self.best_l1_epoch:.1f}")

        if self.has_lpips and self.lpips_losses:
            self.stat_labels['lpips_cur'].configure(text=f"{self.lpips_losses[-1]:.5f}")
            if self.best_lpips < float('inf'):
                self.stat_labels['lpips_best'].configure(text=f"{self.best_lpips:.5f}")

        if self.time_per_step:
            avg_time = sum(self.time_per_step) / len(self.time_per_step)
            self.stat_labels['rate'].configure(text=f"{avg_time:.3f}s")

    def update_text_display(self):
        if not hasattr(self, 'text_display'):
            return

        self.text_display.configure(state=tk.NORMAL)
        self.text_display.delete(1.0, tk.END)

        n = min(50, len(self.steps))
        self.text_display.insert(tk.END, f"Last {n} data points:\n\n")
        self.text_display.insert(tk.END, f"{'Epoch':<10} {'L1 Loss':<14} {'LPIPS':<14}\n")
        self.text_display.insert(tk.END, "-" * 40 + "\n")

        for i in range(-n, 0):
            epoch = self.steps[i]
            l1 = self.l1_losses[i]
            lpips = self.lpips_losses[i] if self.has_lpips and len(self.lpips_losses) > abs(i) else None
            line = f"{epoch:<10.2f} {l1:<14.6f}"
            if lpips is not None:
                line += f" {lpips:<14.6f}"
            self.text_display.insert(tk.END, line + "\n")

        self.text_display.configure(state=tk.DISABLED)

    def clear_data(self):
        self.steps.clear()
        self.l1_losses.clear()
        self.lpips_losses.clear()
        self.has_lpips = False
        self.last_position = 0
        self.best_l1 = float('inf')
        self.best_l1_epoch = 0
        self.best_lpips = float('inf')
        self.best_lpips_epoch = 0
        self.time_per_step.clear()

        if HAS_MATPLOTLIB:
            self.l1_line.set_data([], [])
            self.l1_raw_line.set_data([], [])
            self.lpips_line.set_data([], [])
            self.lpips_raw_line.set_data([], [])
            self.best_l1_marker.set_data([], [])
            self.best_lpips_marker.set_data([], [])
            self.best_l1_annot.set_visible(False)
            self.canvas.draw_idle()

        # Reset stat labels
        if hasattr(self, 'stat_labels'):
            for key in self.stat_labels:
                self.stat_labels[key].configure(text='--')
            self.stat_labels['points'].configure(text='0')

    def update_loop(self):
        if not self.is_monitoring:
            return
        self.read_log_file()
        self.root.after(self.update_interval, self.update_loop)


def main():
    parser = argparse.ArgumentParser(description='TuNet Training Monitor')
    parser.add_argument('--log_file', type=str, help='Path to training.log file')
    parser.add_argument('--output_dir', type=str, help='Path to output directory containing training.log')
    args = parser.parse_args()

    log_file = None
    if args.log_file:
        log_file = args.log_file
    elif args.output_dir:
        log_file = os.path.join(args.output_dir, 'training.log')

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

    app = TrainingMonitor(root, log_file)
    root.mainloop()


if __name__ == "__main__":
    main()
