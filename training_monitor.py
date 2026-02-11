"""
TuNet Training Monitor - Real-time loss graph visualization

Usage:
    python training_monitor.py --log_file path/to/training.log
    python training_monitor.py --output_dir path/to/output  (will find training.log there)

The monitor reads the training log file and displays a live-updating loss graph.
"""

import os
import re
import argparse
import tkinter as tk
from tkinter import ttk, filedialog
from collections import deque
import time

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


class TrainingMonitor:
    def __init__(self, root, log_file=None):
        self.root = root
        self.root.title("Training Monitor")
        self.root.geometry("800x600")
        self.root.configure(bg='#1e1e1e')

        # Data storage
        self.max_points = 10000  # Maximum points to keep in memory
        self.steps = deque(maxlen=self.max_points)
        self.l1_losses = deque(maxlen=self.max_points)
        self.lpips_losses = deque(maxlen=self.max_points)
        self.has_lpips = False

        # File monitoring
        self.log_file = tk.StringVar(value=log_file or "")
        self.last_position = 0
        self.last_modified = 0
        self.is_monitoring = False

        # UI settings
        self.smoothing = tk.DoubleVar(value=0.0)  # Exponential smoothing factor
        self.update_interval = 1000  # ms between updates

        self.create_widgets()

        # Start monitoring if log file provided
        if log_file and os.path.exists(log_file):
            self.start_monitoring()

    def create_widgets(self):
        # Top control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Log file selection
        ttk.Label(control_frame, text="Log File:").pack(side=tk.LEFT, padx=5)
        log_entry = ttk.Entry(control_frame, textvariable=self.log_file, width=50)
        log_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(control_frame, text="Browse", command=self.browse_log).pack(side=tk.LEFT, padx=5)

        self.monitor_btn = ttk.Button(control_frame, text="Start Monitoring", command=self.toggle_monitoring)
        self.monitor_btn.pack(side=tk.LEFT, padx=5)

        # Options frame
        options_frame = ttk.Frame(self.root)
        options_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(options_frame, text="Curve Smoothing:").pack(side=tk.LEFT, padx=5)
        smooth_scale = ttk.Scale(options_frame, from_=0.0, to=0.99, variable=self.smoothing,
                                  orient=tk.HORIZONTAL, length=150)
        smooth_scale.pack(side=tk.LEFT, padx=5)

        ttk.Button(options_frame, text="Clear Data", command=self.clear_data).pack(side=tk.LEFT, padx=20)

        # Status label
        self.status_var = tk.StringVar(value="Not monitoring")
        status_label = ttk.Label(options_frame, textvariable=self.status_var)
        status_label.pack(side=tk.RIGHT, padx=10)

        # Graph frame
        if HAS_MATPLOTLIB:
            self.create_matplotlib_graph()
        else:
            self.create_fallback_display()

    def create_matplotlib_graph(self):
        """Create matplotlib figure for loss visualization."""
        # Create figure with dark theme
        self.fig = Figure(figsize=(10, 6), dpi=100, facecolor='#1e1e1e')
        self.ax = self.fig.add_subplot(111)

        # Style the axes
        self.ax.set_facecolor('#2d2d2d')
        self.ax.set_xlabel('Epoch', color='white', fontsize=12)
        self.ax.set_ylabel('Loss', color='white', fontsize=12)
        self.ax.set_title('TuNet Training Loss', color='white', fontsize=14, fontweight='bold')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('#555555')
        self.ax.spines['top'].set_color('#555555')
        self.ax.spines['left'].set_color('#555555')
        self.ax.spines['right'].set_color('#555555')
        self.ax.grid(True, color='#404040', linestyle='-', linewidth=0.5, alpha=0.7)

        # Create initial empty lines
        self.l1_line, = self.ax.plot([], [], 'r-', linewidth=1.5, label='L1 Loss', alpha=0.9)
        self.lpips_line, = self.ax.plot([], [], 'b-', linewidth=1.5, label='LPIPS Loss', alpha=0.9)

        self.ax.legend(loc='upper right', facecolor='#2d2d2d', edgecolor='#555555',
                       labelcolor='white', fontsize=10)

        # Embed in tkinter
        graph_frame = ttk.Frame(self.root)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig.tight_layout()

    def create_fallback_display(self):
        """Create text-based display if matplotlib not available."""
        fallback_frame = ttk.Frame(self.root)
        fallback_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(fallback_frame, text="matplotlib not installed - showing text output",
                  font=('Arial', 12)).pack(pady=10)

        self.text_display = tk.Text(fallback_frame, bg='#1e1e1e', fg='white',
                                     font=('Consolas', 10), state=tk.DISABLED)
        self.text_display.pack(fill=tk.BOTH, expand=True)

    def browse_log(self):
        """Browse for log file."""
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
        """Toggle monitoring on/off."""
        if self.is_monitoring:
            self.stop_monitoring()
        else:
            self.start_monitoring()

    def start_monitoring(self):
        """Start monitoring the log file."""
        log_path = self.log_file.get()
        if not log_path or not os.path.exists(log_path):
            self.status_var.set("Log file not found")
            return

        self.is_monitoring = True
        self.monitor_btn.configure(text="Stop Monitoring")
        self.status_var.set("Monitoring...")

        # Read existing content first
        self.read_log_file(full_read=True)

        # Start update loop
        self.update_loop()

    def stop_monitoring(self):
        """Stop monitoring the log file."""
        self.is_monitoring = False
        self.monitor_btn.configure(text="Start Monitoring")
        self.status_var.set("Stopped")

    def read_log_file(self, full_read=False):
        """Read new content from log file."""
        log_path = self.log_file.get()
        if not log_path or not os.path.exists(log_path):
            return

        try:
            # Check if file has been modified
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
            self.status_var.set(f"Error: {str(e)[:30]}")

    def parse_log_content(self, content):
        """Parse log content and extract loss values."""
        # Pattern to match training log lines
        # Example: Epoch[1] Step[100] (100/1000), L1:0.1234(Avg:0.1500), LPIPS:0.0500(Avg:0.0600)
        # Also handles: L1:0.1234(Avg:0.1500) without LPIPS

        pattern = r'Epoch\[(\d+)\]\s*Step\[(\d+)\].*?L1:([\d.]+)'
        lpips_pattern = r'LPIPS:([\d.]+)'

        lines = content.split('\n')
        new_data = False

        for line in lines:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                l1_loss = float(match.group(3))

                # Check for LPIPS
                lpips_match = re.search(lpips_pattern, line)
                lpips_loss = float(lpips_match.group(1)) if lpips_match else None

                # Use epoch as x-axis (with fractional parts based on step within epoch)
                # Try to extract iterations_per_epoch from the log
                iter_match = re.search(r'\((\d+)/(\d+)\)', line)
                if iter_match:
                    step_in_epoch = int(iter_match.group(1))
                    total_steps = int(iter_match.group(2))
                    x_value = epoch - 1 + (step_in_epoch / total_steps)
                else:
                    x_value = epoch

                self.steps.append(x_value)
                self.l1_losses.append(l1_loss)

                if lpips_loss is not None:
                    self.lpips_losses.append(lpips_loss)
                    self.has_lpips = True
                elif self.has_lpips:
                    # Keep arrays same length
                    self.lpips_losses.append(self.lpips_losses[-1] if self.lpips_losses else 0)

                new_data = True

        if new_data:
            self.update_graph()

    def apply_smoothing(self, values):
        """Apply exponential moving average smoothing."""
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

    def update_graph(self):
        """Update the graph with current data."""
        if not HAS_MATPLOTLIB:
            self.update_text_display()
            return

        if not self.steps:
            return

        steps = list(self.steps)
        l1_smooth = self.apply_smoothing(self.l1_losses)

        self.l1_line.set_data(steps, l1_smooth)

        if self.has_lpips and self.lpips_losses:
            lpips_smooth = self.apply_smoothing(self.lpips_losses)
            self.lpips_line.set_data(steps, lpips_smooth)
            self.lpips_line.set_visible(True)
        else:
            self.lpips_line.set_visible(False)

        # Adjust axes
        self.ax.relim()
        self.ax.autoscale_view()

        # Set reasonable y-axis limits
        if l1_smooth:
            max_loss = max(l1_smooth)
            min_loss = min(l1_smooth)
            margin = (max_loss - min_loss) * 0.1 or 0.1
            self.ax.set_ylim(max(0, min_loss - margin), max_loss + margin)

        # Update legend based on what's visible
        if self.has_lpips:
            self.ax.legend(loc='upper right', facecolor='#2d2d2d', edgecolor='#555555',
                          labelcolor='white', fontsize=10)

        self.canvas.draw_idle()

        # Update status
        if self.steps:
            self.status_var.set(f"Epoch: {self.steps[-1]:.2f} | L1: {self.l1_losses[-1]:.4f}" +
                               (f" | LPIPS: {self.lpips_losses[-1]:.4f}" if self.has_lpips and self.lpips_losses else "") +
                               f" | Points: {len(self.steps)}")

    def update_text_display(self):
        """Update text display for fallback mode."""
        if not hasattr(self, 'text_display'):
            return

        self.text_display.configure(state=tk.NORMAL)
        self.text_display.delete(1.0, tk.END)

        # Show last 50 data points
        n = min(50, len(self.steps))
        self.text_display.insert(tk.END, f"Last {n} data points:\n\n")
        self.text_display.insert(tk.END, f"{'Epoch':<10} {'L1 Loss':<12} {'LPIPS':<12}\n")
        self.text_display.insert(tk.END, "-" * 35 + "\n")

        for i in range(-n, 0):
            epoch = self.steps[i]
            l1 = self.l1_losses[i]
            lpips = self.lpips_losses[i] if self.has_lpips and len(self.lpips_losses) > abs(i) else None

            line = f"{epoch:<10.2f} {l1:<12.6f}"
            if lpips is not None:
                line += f" {lpips:<12.6f}"
            self.text_display.insert(tk.END, line + "\n")

        self.text_display.configure(state=tk.DISABLED)

    def clear_data(self):
        """Clear all collected data."""
        self.steps.clear()
        self.l1_losses.clear()
        self.lpips_losses.clear()
        self.has_lpips = False
        self.last_position = 0

        if HAS_MATPLOTLIB:
            self.l1_line.set_data([], [])
            self.lpips_line.set_data([], [])
            self.canvas.draw_idle()

        self.status_var.set("Data cleared")

    def update_loop(self):
        """Main update loop."""
        if not self.is_monitoring:
            return

        self.read_log_file()
        self.root.after(self.update_interval, self.update_loop)


def main():
    parser = argparse.ArgumentParser(description='TuNet Training Monitor')
    parser.add_argument('--log_file', type=str, help='Path to training.log file')
    parser.add_argument('--output_dir', type=str, help='Path to output directory containing training.log')
    args = parser.parse_args()

    # Determine log file path
    log_file = None
    if args.log_file:
        log_file = args.log_file
    elif args.output_dir:
        log_file = os.path.join(args.output_dir, 'training.log')

    if not HAS_MATPLOTLIB:
        print("\n" + "="*60)
        print("WARNING: matplotlib is not installed!")
        print("For the best experience, install it with:")
        print("  pip install matplotlib")
        print("="*60 + "\n")

    root = tk.Tk()

    # Set dark theme for ttk
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('.', background='#1e1e1e', foreground='white')
    style.configure('TFrame', background='#1e1e1e')
    style.configure('TLabel', background='#1e1e1e', foreground='white')
    style.configure('TButton', background='#3d3d3d', foreground='white')
    style.configure('TEntry', fieldbackground='#3d3d3d', foreground='white')
    style.configure('TScale', background='#1e1e1e', troughcolor='#3d3d3d')

    app = TrainingMonitor(root, log_file)
    root.mainloop()


if __name__ == "__main__":
    main()
