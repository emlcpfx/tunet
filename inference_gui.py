import math
import os
import json
import queue
import shutil
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging

# Import inference functions
from inference import load_model_and_config, process_image, denormalize, NORM_MEAN, NORM_STD, load_image_any_format

import torch
import torchvision.transforms as T

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inference_gui_settings.json')


class InferenceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TuNet Inference")
        self.root.geometry("700x550")
        self.root.resizable(True, True)

        # Variables
        self.checkpoint_path = tk.StringVar()
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.stride = tk.IntVar(value=256)
        self.auto_stride = tk.BooleanVar(value=True)
        self.half_res = tk.BooleanVar(value=False)
        self.skip_existing = tk.BooleanVar(value=True)

        self.is_running = False
        self.stop_requested = False

        # Queue: list of dicts with 'input_dir', 'output_dir', 'status'
        self.queue = []

        # Cached model so we don't reload between queue items
        self._cached_model = None
        self._cached_checkpoint = None
        self._cached_resolution = None
        self._cached_loss_mode = None

        self.create_widgets()
        self.setup_logging()
        self.load_settings()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Checkpoint selection
        row = 0
        ttk.Label(main_frame, text="Model Checkpoint:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.checkpoint_path, width=50).grid(row=row, column=1, sticky=tk.EW, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_checkpoint).grid(row=row, column=2, padx=5, pady=5)

        # Input directory
        row += 1
        ttk.Label(main_frame, text="Input Directory:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_dir, width=50).grid(row=row, column=1, sticky=tk.EW, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_input).grid(row=row, column=2, padx=5, pady=5)

        # Output directory
        row += 1
        ttk.Label(main_frame, text="Output Directory:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50).grid(row=row, column=1, sticky=tk.EW, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(row=row, column=2, padx=5, pady=5)

        # Options frame
        row += 1
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="5")
        options_frame.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=10)

        ttk.Label(options_frame, text="Stride:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.stride_spinbox = ttk.Spinbox(options_frame, from_=64, to=1024, increment=64,
                     textvariable=self.stride, width=8)
        self.stride_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Checkbutton(options_frame, text="Auto", variable=self.auto_stride,
                        command=self._update_stride_state).grid(row=0, column=2, sticky=tk.W, padx=2)

        ttk.Checkbutton(options_frame, text="Half-res (~4x faster)", variable=self.half_res).grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5)

        ttk.Checkbutton(options_frame, text="Skip existing output files", variable=self.skip_existing).grid(
            row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        self._update_stride_state()

        # Queue section
        row += 1
        queue_frame = ttk.LabelFrame(main_frame, text="Queue", padding="5")
        queue_frame.grid(row=row, column=0, columnspan=3, sticky=tk.NSEW, pady=5)
        queue_frame.columnconfigure(0, weight=1)
        queue_frame.rowconfigure(0, weight=1)

        # Queue listbox with scrollbar
        queue_list_frame = ttk.Frame(queue_frame)
        queue_list_frame.grid(row=0, column=0, sticky=tk.NSEW)
        queue_list_frame.columnconfigure(0, weight=1)
        queue_list_frame.rowconfigure(0, weight=1)

        self.queue_listbox = tk.Listbox(queue_list_frame, height=5, selectmode=tk.EXTENDED,
                                         font=('Consolas', 9))
        self.queue_listbox.grid(row=0, column=0, sticky=tk.NSEW)
        queue_scroll = ttk.Scrollbar(queue_list_frame, orient=tk.VERTICAL, command=self.queue_listbox.yview)
        queue_scroll.grid(row=0, column=1, sticky=tk.NS)
        self.queue_listbox.configure(yscrollcommand=queue_scroll.set)

        # Queue buttons
        queue_btn_frame = ttk.Frame(queue_frame)
        queue_btn_frame.grid(row=0, column=1, sticky=tk.N, padx=(5, 0))

        ttk.Button(queue_btn_frame, text="Add", width=8, command=self.add_to_queue).pack(pady=2)
        ttk.Button(queue_btn_frame, text="Remove", width=8, command=self.remove_from_queue).pack(pady=2)
        ttk.Button(queue_btn_frame, text="Clear", width=8, command=self.clear_queue).pack(pady=2)

        # Run / Stop buttons
        row += 1
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=10)

        self.run_button = ttk.Button(btn_frame, text="Run Inference", command=self.run_inference)
        self.run_button.pack(side=tk.LEFT, padx=5)

        self.run_queue_button = ttk.Button(btn_frame, text="Run Queue", command=self.run_queue)
        self.run_queue_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(btn_frame, text="Stop", command=self.request_stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Progress bar
        row += 1
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=5)

        self.progress_label = ttk.Label(main_frame, text="")
        self.progress_label.grid(row=row, column=0, columnspan=3, sticky=tk.E, pady=5)

        # Log output
        row += 1
        ttk.Label(main_frame, text="Log:").grid(row=row, column=0, sticky=tk.W)
        row += 1
        self.log_text = tk.Text(main_frame, height=6, width=70, state=tk.DISABLED)
        self.log_text.grid(row=row, column=0, columnspan=3, sticky=tk.NSEW, pady=5)

        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=row, column=3, sticky=tk.NS)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        # Grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(row, weight=1)

    def setup_logging(self):
        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget

            def emit(self, record):
                msg = self.format(record) + '\n'
                self.text_widget.after(0, self.append_text, msg)

            def append_text(self, msg):
                self.text_widget.configure(state=tk.NORMAL)
                self.text_widget.insert(tk.END, msg)
                self.text_widget.see(tk.END)
                self.text_widget.configure(state=tk.DISABLED)

        handler = TextHandler(self.log_text)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    # ─── Settings ───

    def load_settings(self):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                s = json.load(f)
            self.checkpoint_path.set(s.get('checkpoint_path', ''))
            self.input_dir.set(s.get('input_dir', ''))
            self.output_dir.set(s.get('output_dir', ''))
            self.stride.set(s.get('stride', 256))
            self.auto_stride.set(s.get('auto_stride', True))
            self._update_stride_state()
            self.half_res.set(s.get('half_res', False))
            self.skip_existing.set(s.get('skip_existing', True))
            self.device.set(s.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
            # Restore queue
            for item in s.get('queue', []):
                input_d = os.path.normpath(item.get('input_dir', ''))
                output_d = os.path.normpath(item.get('output_dir', ''))
                if os.path.isdir(input_d):
                    self.queue.append({'input_dir': input_d,
                                       'output_dir': output_d,
                                       'status': 'pending'})
            self.refresh_queue_display()
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def save_settings(self):
        s = {
            'checkpoint_path': self.checkpoint_path.get(),
            'input_dir': self.input_dir.get(),
            'output_dir': self.output_dir.get(),
            'stride': self.stride.get(),
            'auto_stride': self.auto_stride.get(),
            'half_res': self.half_res.get(),
            'skip_existing': self.skip_existing.get(),
            'device': self.device.get(),
            'queue': [{'input_dir': q['input_dir'], 'output_dir': q['output_dir']}
                      for q in self.queue if q['status'] == 'pending'],
        }
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(s, f, indent=2)
        except OSError:
            pass

    def on_close(self):
        self.save_settings()
        self.root.destroy()

    # ─── Browse ───

    def browse_checkpoint(self):
        path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if path:
            self.checkpoint_path.set(os.path.normpath(path))

    def browse_input(self):
        path = filedialog.askdirectory(title="Select Input Directory")
        if path:
            self.input_dir.set(os.path.normpath(path))

    def browse_output(self):
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_dir.set(os.path.normpath(path))

    # ─── Queue management ───

    def add_to_queue(self):
        input_d = self.input_dir.get()
        output_d = self.output_dir.get()
        if not input_d or not os.path.isdir(input_d):
            messagebox.showerror("Error", "Input directory is invalid.")
            return
        if not output_d:
            messagebox.showerror("Error", "Output directory is empty.")
            return
        self.queue.append({'input_dir': input_d, 'output_dir': output_d, 'status': 'pending'})
        self.refresh_queue_display()
        logging.info(f"Added to queue: {os.path.basename(input_d)} -> {os.path.basename(output_d)}")

    def remove_from_queue(self):
        selected = list(self.queue_listbox.curselection())
        if not selected:
            return
        # Remove in reverse order so indices stay valid
        for idx in reversed(selected):
            if idx < len(self.queue) and self.queue[idx]['status'] != 'processing':
                self.queue.pop(idx)
        self.refresh_queue_display()

    def clear_queue(self):
        self.queue = [q for q in self.queue if q['status'] == 'processing']
        self.refresh_queue_display()

    def refresh_queue_display(self):
        self.queue_listbox.delete(0, tk.END)
        for i, item in enumerate(self.queue):
            status = item['status']
            prefix = {'pending': '   ', 'processing': '>> ', 'done': 'OK ', 'error': '!! '}
            display = f"{prefix.get(status, '   ')}{os.path.basename(item['input_dir'])} -> {os.path.basename(item['output_dir'])}"
            self.queue_listbox.insert(tk.END, display)
            if status == 'processing':
                self.queue_listbox.itemconfigure(i, fg='#4488ff')
            elif status == 'done':
                self.queue_listbox.itemconfigure(i, fg='#44bb44')
            elif status == 'error':
                self.queue_listbox.itemconfigure(i, fg='#ff4444')

    # ─── Validation ───

    def validate_checkpoint(self):
        if not self.checkpoint_path.get():
            messagebox.showerror("Error", "Please select a model checkpoint.")
            return False
        if not os.path.exists(self.checkpoint_path.get()):
            messagebox.showerror("Error", "Checkpoint file not found.")
            return False
        return True

    def validate_inputs(self):
        if not self.validate_checkpoint():
            return False
        if not self.input_dir.get():
            messagebox.showerror("Error", "Please select an input directory.")
            return False
        if not os.path.isdir(self.input_dir.get()):
            messagebox.showerror("Error", "Input directory not found.")
            return False
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please select an output directory.")
            return False
        return True

    # ─── Stop ───

    def request_stop(self):
        self.stop_requested = True
        logging.info("Stop requested, finishing current image...")

    # ─── Run single ───

    def run_inference(self):
        if self.is_running:
            return
        if not self.validate_inputs():
            return
        self._start_processing()
        thread = threading.Thread(target=self._run_single_thread, daemon=True)
        thread.start()

    def _run_single_thread(self):
        self.save_settings()
        try:
            model, resolution, loss_mode, device, use_amp, transform, stride, tile_batch = self._load_model()
            self._process_folder(model, self.input_dir.get(), self.output_dir.get(),
                                 resolution, stride, device, use_amp, transform, loss_mode, tile_batch)
            if not self.stop_requested:
                logging.info("Inference complete!")
                self.root.after(0, lambda: messagebox.showinfo("Done", "Inference completed successfully!"))
        except Exception as e:
            logging.error(f"Error: {e}")
            err_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Error", err_msg))
        finally:
            self.root.after(0, self._finish_processing)

    # ─── Run queue ───

    def run_queue(self):
        if self.is_running:
            return
        pending = [q for q in self.queue if q['status'] == 'pending']
        if not pending:
            messagebox.showinfo("Queue Empty", "No pending items in the queue. Add folders first.")
            return
        if not self.validate_checkpoint():
            return
        self._start_processing()
        thread = threading.Thread(target=self._run_queue_thread, daemon=True)
        thread.start()

    def _run_queue_thread(self):
        self.save_settings()
        try:
            model, resolution, loss_mode, device, use_amp, transform, stride, tile_batch = self._load_model()

            pending = [q for q in self.queue if q['status'] == 'pending']
            total_items = len(pending)

            for qi, item in enumerate(pending):
                if self.stop_requested:
                    logging.info("Queue stopped by user.")
                    break

                logging.info(f"--- Queue item {qi+1}/{total_items}: {os.path.basename(item['input_dir'])} ---")
                item['status'] = 'processing'
                self.root.after(0, self.refresh_queue_display)

                try:
                    self._process_folder(model, item['input_dir'], item['output_dir'],
                                         resolution, stride, device, use_amp, transform, loss_mode, tile_batch)
                    if not self.stop_requested:
                        item['status'] = 'done'
                    else:
                        item['status'] = 'pending'
                except Exception as e:
                    logging.error(f"Error processing {item['input_dir']}: {e}")
                    item['status'] = 'error'

                self.root.after(0, self.refresh_queue_display)

            if not self.stop_requested:
                done_count = sum(1 for q in self.queue if q['status'] == 'done')
                logging.info(f"Queue complete! {done_count}/{total_items} items processed.")
                self.root.after(0, lambda: messagebox.showinfo("Done", f"Queue complete! {done_count}/{total_items} items processed."))

        except Exception as e:
            logging.error(f"Queue error: {e}")
            err_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Error", err_msg))
        finally:
            self.root.after(0, self._finish_processing)

    # ─── Shared processing logic ───

    def _start_processing(self):
        self.is_running = True
        self.stop_requested = False
        self.run_button.configure(state=tk.DISABLED)
        self.run_queue_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.progress.configure(value=0)

    def _finish_processing(self):
        self.is_running = False
        self.stop_requested = False
        self.run_button.configure(state=tk.NORMAL)
        self.run_queue_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.progress.configure(value=0)
        self.progress_label.configure(text="")
        self._cached_model = None

    def _load_model(self):
        from glob import glob
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_str)
        use_amp = device_str == 'cuda'

        checkpoint = self.checkpoint_path.get()

        # Reuse cached model if same checkpoint
        if self._cached_model is not None and self._cached_checkpoint == checkpoint:
            logging.info("Reusing loaded model.")
            model = self._cached_model
            resolution = self._cached_resolution
            loss_mode = self._cached_loss_mode
        else:
            logging.info(f"Using device: {device}")
            logging.info("Loading model...")
            model, resolution, use_mask_input, loss_mode = load_model_and_config(checkpoint, device)
            # Compile the model for faster repeated inference (PyTorch 2.0+)
            # torch.compile requires Triton which is not available on Windows
            if os.name != 'nt':
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    logging.info("torch.compile enabled (reduce-overhead) — first frame will warm up.")
                except Exception as e:
                    logging.warning(f"torch.compile unavailable, using eager mode: {e}")
            else:
                logging.info("torch.compile skipped (Triton not supported on Windows).")
            self._cached_model = model
            self._cached_checkpoint = checkpoint
            self._cached_resolution = resolution
            self._cached_loss_mode = loss_mode

        transform = T.Compose([T.ToTensor(), T.Normalize(mean=NORM_MEAN, std=NORM_STD)])
        stride = max(1, self.stride.get())
        logging.info(f"Resolution: {resolution}, Stride: {stride} ({'auto' if self.auto_stride.get() else 'manual'})")

        if device.type == 'cuda':
            props = torch.cuda.get_device_properties(device)
            vram_gb = props.total_memory / (1024 ** 3)
            if vram_gb >= 20:
                tile_batch = 16
            elif vram_gb >= 10:
                tile_batch = 8
            elif vram_gb >= 6:
                tile_batch = 4
            else:
                tile_batch = 2
            logging.info(f"GPU: {props.name} ({vram_gb:.0f}GB) → tile batch = {tile_batch}")
        else:
            tile_batch = 1

        return model, resolution, loss_mode, device, use_amp, transform, stride, tile_batch

    def _update_stride_state(self):
        state = 'disabled' if self.auto_stride.get() else 'normal'
        self.stride_spinbox.configure(state=state)

    @staticmethod
    def _compute_optimal_stride(img_w, img_h, resolution):
        """Minimum stride that gives full pixel coverage with no gaps."""
        if img_w <= resolution and img_h <= resolution:
            return resolution
        def min_stride_for_dim(dim):
            excess = dim - resolution
            k = math.ceil(excess / resolution)
            return excess / k
        strides = []
        if img_w > resolution:
            strides.append(min_stride_for_dim(img_w))
        if img_h > resolution:
            strides.append(min_stride_for_dim(img_h))
        return max(1, int(math.floor(min(strides))))

    @staticmethod
    def _is_network_path(path):
        """Return True if path is a UNC share or mapped network drive."""
        norm = os.path.normpath(path)
        # UNC path: starts with \\
        if norm.startswith('\\\\'):
            return True
        # On Windows, check if drive letter is a network drive
        if os.name == 'nt' and len(norm) >= 2 and norm[1] == ':':
            import ctypes
            drive = norm[:3]  # e.g. "Z:\"
            dtype = ctypes.windll.kernel32.GetDriveTypeW(drive)
            DRIVE_REMOTE = 4
            return dtype == DRIVE_REMOTE
        return False

    def _copy_worker(self, copy_queue, tmp_dir):
        """Background thread: move files from tmp_dir to their final network destinations."""
        while True:
            item = copy_queue.get()
            if item is None:  # sentinel
                break
            tmp_path, final_path = item
            try:
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                shutil.move(tmp_path, final_path)
            except Exception as e:
                logging.error(f"Async copy failed {os.path.basename(final_path)}: {e}")
            finally:
                copy_queue.task_done()

    def _process_folder(self, model, input_dir, output_dir, resolution, stride,
                        device, use_amp, transform, loss_mode, tile_batch=4):
        from glob import glob
        import re
        import time

        img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.webp', '*.exr']
        input_files = sorted([f for ext in img_extensions
                              for f in glob(os.path.join(input_dir, ext))])

        if not input_files:
            logging.error(f"No images found in {input_dir}")
            return

        os.makedirs(output_dir, exist_ok=True)
        skip_existing = self.skip_existing.get()

        # Build output path list and filter
        work_items = []
        for img_path in input_files:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            match = re.match(r'^(.+?)_?(\d+)$', basename)
            if match:
                output_filename = f"{match.group(1)}_tunet_{match.group(2)}.png"
            else:
                output_filename = f"{basename}_tunet.png"
            output_path = os.path.join(output_dir, output_filename)
            work_items.append((img_path, output_path))

        skipped = 0
        if skip_existing:
            filtered = []
            for img_path, output_path in work_items:
                if os.path.exists(output_path):
                    skipped += 1
                else:
                    filtered.append((img_path, output_path))
            work_items = filtered

        logging.info(f"Found {len(input_files)} images in {os.path.basename(input_dir)}"
                     + (f" ({skipped} already done, {len(work_items)} remaining)" if skipped else ""))

        if not work_items:
            logging.info("All files already processed, nothing to do.")
            return

        # Auto-compute stride from first image dimensions
        if self.auto_stride.get():
            try:
                from PIL import Image as _PIL
                with _PIL.open(work_items[0][0]) as _img:
                    first_w, first_h = _img.size
                stride = self._compute_optimal_stride(first_w, first_h, resolution)
                # Count tiles for the log
                def _tile_pos(dim):
                    return len(set(list(range(0, dim - resolution, stride)) +
                                   ([dim - resolution] if dim > resolution else [0])))
                tiles = _tile_pos(first_w) * _tile_pos(first_h)
                logging.info(f"Auto-stride: {first_w}×{first_h} → stride={stride}, {tiles} tile(s)/frame")
                self.root.after(0, lambda s=stride: self.stride.set(s))
            except Exception as e:
                logging.warning(f"Auto-stride failed ({e}), using manual stride={stride}")

        # Prefetch images from disk (hides network read latency behind GPU compute)
        prefetch_q = queue.Queue(maxsize=2)

        def prefetch_worker():
            for item_pair in work_items:
                if self.stop_requested:
                    prefetch_q.put(None)
                    return
                img_path, _ = item_pair
                try:
                    pil = load_image_any_format(img_path)
                except Exception as e:
                    logging.error(f"Prefetch failed for {os.path.basename(img_path)}: {e}")
                    pil = None
                prefetch_q.put((item_pair, pil))
            prefetch_q.put(None)  # sentinel

        prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        prefetch_thread.start()

        # Use async local-write + background copy for network output dirs
        use_async_copy = self._is_network_path(output_dir)
        tmp_dir = None
        copy_queue = None
        copy_thread = None

        if use_async_copy:
            tmp_dir = tempfile.mkdtemp(prefix='tunet_')
            copy_queue = queue.Queue()
            copy_thread = threading.Thread(target=self._copy_worker,
                                           args=(copy_queue, tmp_dir), daemon=True)
            copy_thread.start()
            logging.info(f"Network output detected — writing to local temp, copying async. ({tmp_dir})")

        total = len(work_items)
        frame_times = []
        processed = 0

        try:
            stop_early = False
            while True:
                if self.stop_requested:
                    stop_early = True
                    break
                prefetched = prefetch_q.get()
                if prefetched is None:
                    break
                (img_path, output_path), pil_img = prefetched
                if pil_img is None:
                    logging.error(f"Skipping {os.path.basename(img_path)} (load failed during prefetch)")
                    continue
                write_path = (os.path.join(tmp_dir, os.path.basename(output_path))
                              if use_async_copy else output_path)

                t0 = time.perf_counter()
                process_image(model, img_path, write_path, resolution, stride, device,
                              tile_batch, transform, denormalize, use_amp, self.half_res.get(),
                              loss_mode=loss_mode, src_image=pil_img)
                elapsed = time.perf_counter() - t0

                processed += 1
                frame_times.append(elapsed)
                avg = sum(frame_times) / len(frame_times)
                eta_str = self._format_eta(int(avg * (total - processed)))

                if use_async_copy:
                    copy_queue.put((write_path, output_path))
                    qd = copy_queue.qsize()
                    qi = f"  [copy queue: {qd}]" if qd > 1 else ""
                else:
                    qi = ""

                logging.info(f"[{processed}/{total}] {os.path.basename(img_path)}  "
                             f"{elapsed:.2f}s  avg {avg:.2f}s  ETA {eta_str}{qi}")
                pct = int(processed / total * 100)
                self.root.after(0, lambda p=pct, c=processed, t=total, e=eta_str, fr=elapsed, a=avg:
                                self._update_progress(p, c, t, e, fr, a))

            if stop_early:
                logging.info(f"Stopped. {processed}/{total} processed this run, {total - processed} remaining.")

        finally:
            # Drain prefetch queue so the prefetch thread can exit (it may be blocked on put())
            try:
                while True:
                    prefetch_q.get_nowait()
            except queue.Empty:
                pass
            prefetch_thread.join(timeout=5)

            if use_async_copy and copy_queue is not None:
                pending = copy_queue.qsize()
                if pending > 0:
                    logging.info(f"Waiting for {pending} files to finish copying to network...")
                copy_queue.put(None)  # sentinel
                copy_thread.join()
                # Clean up temp dir (should be empty now)
                try:
                    os.rmdir(tmp_dir)
                except OSError:
                    pass

    def _format_eta(self, secs):
        if secs < 60:
            return f"{secs}s"
        elif secs < 3600:
            return f"{secs // 60}m {secs % 60:02d}s"
        else:
            h = secs // 3600
            m = (secs % 3600) // 60
            return f"{h}h {m:02d}m"

    def _update_progress(self, pct, current, total, eta="", frame_time=None, avg_time=None):
        self.progress.configure(value=pct)
        parts = [f"{current}/{total}"]
        if frame_time is not None:
            parts.append(f"{frame_time:.2f}s/frame")
        if avg_time is not None:
            parts.append(f"avg {avg_time:.2f}s")
        if eta:
            parts.append(f"ETA {eta}")
        self.progress_label.configure(text="  ".join(parts))


def main():
    root = tk.Tk()
    app = InferenceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
