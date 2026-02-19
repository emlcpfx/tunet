import os
import multiprocessing as mp
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
import queue
import time

import torch
import torchvision.transforms as T

# Import inference functions
from inference import load_model_and_config, process_image, denormalize, NORM_MEAN, NORM_STD


def gpu_worker(gpu_id, checkpoint_path, image_paths, output_dir, resolution, stride,
               batch_size, use_amp, half_res, overlap_factor, log_queue, progress_queue):
    """Worker function that runs on a specific GPU."""
    try:
        device = torch.device(f'cuda:{gpu_id}')
        log_queue.put((gpu_id, 'info', f'GPU {gpu_id}: Starting worker'))

        # Load model on this GPU
        model, res, _ = load_model_and_config(checkpoint_path, device)
        log_queue.put((gpu_id, 'info', f'GPU {gpu_id}: Model loaded'))

        transform = T.Compose([T.ToTensor(), T.Normalize(mean=NORM_MEAN, std=NORM_STD)])

        import re
        for i, img_path in enumerate(image_paths):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            match = re.match(r'^(.+?)_?(\d+)$', basename)
            if match:
                name_part = match.group(1)
                frame_num = match.group(2)
                output_filename = f"{name_part}_tunet_{frame_num}.png"
            else:
                output_filename = f"{basename}_tunet.png"

            output_path = os.path.join(output_dir, output_filename)
            log_queue.put((gpu_id, 'info', f'GPU {gpu_id}: Processing {os.path.basename(img_path)} ({i+1}/{len(image_paths)})'))

            process_image(model, img_path, output_path, resolution, stride, device,
                          batch_size, transform, denormalize, use_amp, half_res)

            progress_queue.put((gpu_id, i + 1, len(image_paths)))

        log_queue.put((gpu_id, 'info', f'GPU {gpu_id}: Finished all {len(image_paths)} images'))
        progress_queue.put((gpu_id, 'done', len(image_paths)))

    except Exception as e:
        log_queue.put((gpu_id, 'error', f'GPU {gpu_id}: Error - {str(e)}'))
        progress_queue.put((gpu_id, 'error', str(e)))


class MultiGPUInferenceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TuNet Multi-GPU Inference")
        self.root.geometry("700x550")
        self.root.resizable(True, True)

        # Variables
        self.checkpoint_path = tk.StringVar()
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.overlap_factor = tk.DoubleVar(value=0.5)
        self.batch_size = tk.IntVar(value=1)
        self.use_amp = tk.BooleanVar(value=True)
        self.half_res = tk.BooleanVar(value=False)

        # Detect GPUs
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.gpu_vars = []  # Checkboxes for each GPU

        self.is_running = False
        self.processes = []
        self.log_queue = None
        self.progress_queue = None

        self.create_widgets()
        self.setup_logging()

    def create_widgets(self):
        # Main frame with padding
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

        # GPU Selection frame
        row += 1
        gpu_frame = ttk.LabelFrame(main_frame, text=f"GPU Selection ({self.num_gpus} detected)", padding="5")
        gpu_frame.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=10)

        if self.num_gpus == 0:
            ttk.Label(gpu_frame, text="No CUDA GPUs detected. Multi-GPU inference requires CUDA.").pack()
        else:
            gpu_inner_frame = ttk.Frame(gpu_frame)
            gpu_inner_frame.pack(fill=tk.X)

            for i in range(self.num_gpus):
                var = tk.BooleanVar(value=True)  # All GPUs selected by default
                self.gpu_vars.append(var)

                gpu_name = torch.cuda.get_device_name(i)
                cb = ttk.Checkbutton(gpu_inner_frame, text=f"GPU {i}: {gpu_name}", variable=var)
                cb.grid(row=i // 2, column=i % 2, sticky=tk.W, padx=10, pady=2)

        # Options frame
        row += 1
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="5")
        options_frame.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=10)

        # Overlap factor
        ttk.Label(options_frame, text="Overlap Factor:").grid(row=0, column=0, sticky=tk.W, padx=5)
        overlap_spin = ttk.Spinbox(options_frame, from_=0.0, to=0.9, increment=0.1,
                                   textvariable=self.overlap_factor, width=10)
        overlap_spin.grid(row=0, column=1, sticky=tk.W, padx=5)

        # Batch size
        ttk.Label(options_frame, text="Batch Size:").grid(row=0, column=2, sticky=tk.W, padx=5)
        batch_spin = ttk.Spinbox(options_frame, from_=1, to=32, increment=1,
                                 textvariable=self.batch_size, width=10)
        batch_spin.grid(row=0, column=3, sticky=tk.W, padx=5)

        # AMP checkbox
        ttk.Checkbutton(options_frame, text="Use AMP (faster)", variable=self.use_amp).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        # Half-res checkbox
        ttk.Checkbutton(options_frame, text="Half-res (~4x faster)", variable=self.half_res).grid(
            row=1, column=2, columnspan=2, sticky=tk.W, padx=5, pady=5)

        # Run button
        row += 1
        self.run_button = ttk.Button(main_frame, text="Run Multi-GPU Inference", command=self.run_inference)
        self.run_button.grid(row=row, column=0, columnspan=3, pady=15)

        # Progress section
        row += 1
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5")
        progress_frame.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=5)

        self.progress_labels = {}
        self.progress_bars = {}

        if self.num_gpus > 0:
            for i in range(self.num_gpus):
                lbl = ttk.Label(progress_frame, text=f"GPU {i}: Idle")
                lbl.grid(row=i, column=0, sticky=tk.W, padx=5)
                self.progress_labels[i] = lbl

                pb = ttk.Progressbar(progress_frame, mode='determinate', length=200)
                pb.grid(row=i, column=1, sticky=tk.EW, padx=5, pady=2)
                self.progress_bars[i] = pb

            progress_frame.columnconfigure(1, weight=1)

        # Log output
        row += 1
        ttk.Label(main_frame, text="Log:").grid(row=row, column=0, sticky=tk.W)
        row += 1
        self.log_text = tk.Text(main_frame, height=10, width=80, state=tk.DISABLED)
        self.log_text.grid(row=row, column=0, columnspan=3, sticky=tk.NSEW, pady=5)

        # Scrollbar for log
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=row, column=3, sticky=tk.NS)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        # Configure grid weights
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
        # Remove existing handlers to avoid duplicates
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        logger.addHandler(handler)

    def browse_checkpoint(self):
        path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if path:
            self.checkpoint_path.set(path)

    def browse_input(self):
        path = filedialog.askdirectory(title="Select Input Directory")
        if path:
            self.input_dir.set(path)

    def browse_output(self):
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_dir.set(path)

    def validate_inputs(self):
        if not self.checkpoint_path.get():
            messagebox.showerror("Error", "Please select a model checkpoint.")
            return False
        if not os.path.exists(self.checkpoint_path.get()):
            messagebox.showerror("Error", "Checkpoint file not found.")
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

        # Check at least one GPU is selected
        selected_gpus = [i for i, var in enumerate(self.gpu_vars) if var.get()]
        if not selected_gpus:
            messagebox.showerror("Error", "Please select at least one GPU.")
            return False

        return True

    def run_inference(self):
        if self.is_running:
            return

        if not self.validate_inputs():
            return

        self.is_running = True
        self.run_button.configure(state=tk.DISABLED)

        # Reset progress bars
        for i in self.progress_bars:
            self.progress_bars[i]['value'] = 0
            self.progress_labels[i]['text'] = f"GPU {i}: Waiting..."

        # Start inference in separate thread (to manage processes)
        import threading
        thread = threading.Thread(target=self.inference_coordinator, daemon=True)
        thread.start()

    def inference_coordinator(self):
        """Coordinates multi-GPU inference using multiprocessing."""
        try:
            from glob import glob

            # Find images
            img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.webp', '*.exr']
            input_files = sorted([f for ext in img_extensions
                                  for f in glob(os.path.join(self.input_dir.get(), ext))])

            if not input_files:
                logging.error("No images found in input directory")
                self.root.after(0, self.inference_complete)
                return

            logging.info(f"Found {len(input_files)} images")
            os.makedirs(self.output_dir.get(), exist_ok=True)

            # Get selected GPUs
            selected_gpus = [i for i, var in enumerate(self.gpu_vars) if var.get()]
            num_selected = len(selected_gpus)

            logging.info(f"Using {num_selected} GPUs: {selected_gpus}")

            # Load model once to get resolution
            device = torch.device(f'cuda:{selected_gpus[0]}')
            _, resolution, _ = load_model_and_config(self.checkpoint_path.get(), device)

            overlap_pixels = int(resolution * self.overlap_factor.get())
            stride = max(1, resolution - overlap_pixels)

            logging.info(f"Resolution: {resolution}, Stride: {stride}")

            # Distribute images across GPUs
            images_per_gpu = [[] for _ in range(num_selected)]
            for i, img_path in enumerate(input_files):
                images_per_gpu[i % num_selected].append(img_path)

            # Create queues for communication
            self.log_queue = mp.Queue()
            self.progress_queue = mp.Queue()

            # Start worker processes
            self.processes = []
            start_time = time.time()

            for idx, gpu_id in enumerate(selected_gpus):
                if not images_per_gpu[idx]:
                    continue

                p = mp.Process(
                    target=gpu_worker,
                    args=(
                        gpu_id,
                        self.checkpoint_path.get(),
                        images_per_gpu[idx],
                        self.output_dir.get(),
                        resolution,
                        stride,
                        self.batch_size.get(),
                        self.use_amp.get(),
                        self.half_res.get(),
                        self.overlap_factor.get(),
                        self.log_queue,
                        self.progress_queue
                    )
                )
                p.start()
                self.processes.append(p)
                logging.info(f"Started worker on GPU {gpu_id} with {len(images_per_gpu[idx])} images")

            # Monitor progress
            completed_gpus = set()
            while len(completed_gpus) < len(self.processes):
                # Process log messages
                try:
                    while True:
                        gpu_id, level, msg = self.log_queue.get_nowait()
                        if level == 'info':
                            logging.info(msg)
                        elif level == 'error':
                            logging.error(msg)
                except:
                    pass

                # Process progress updates
                try:
                    while True:
                        gpu_id, current, total = self.progress_queue.get_nowait()
                        if current == 'done':
                            completed_gpus.add(gpu_id)
                            self.root.after(0, lambda g=gpu_id, t=total: self.update_progress(g, t, t, done=True))
                        elif current == 'error':
                            completed_gpus.add(gpu_id)
                            self.root.after(0, lambda g=gpu_id: self.update_progress(g, 0, 0, error=True))
                        else:
                            self.root.after(0, lambda g=gpu_id, c=current, t=total: self.update_progress(g, c, t))
                except:
                    pass

                time.sleep(0.1)

            # Wait for all processes to finish
            for p in self.processes:
                p.join()

            end_time = time.time()
            total_time = end_time - start_time
            logging.info(f"All GPUs finished! Total time: {total_time:.2f} seconds")
            logging.info(f"Processed {len(input_files)} images across {num_selected} GPUs")
            logging.info(f"Average: {total_time / len(input_files):.2f} sec/image")

            self.root.after(0, lambda: messagebox.showinfo("Done",
                f"Inference completed!\n\n"
                f"Processed {len(input_files)} images\n"
                f"Total time: {total_time:.2f} seconds\n"
                f"Average: {total_time / len(input_files):.2f} sec/image"))

        except Exception as e:
            logging.error(f"Error: {e}")
            import traceback
            logging.error(traceback.format_exc())
            err_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Error", err_msg))
        finally:
            self.root.after(0, self.inference_complete)

    def update_progress(self, gpu_id, current, total, done=False, error=False):
        """Update progress bar and label for a specific GPU."""
        if gpu_id in self.progress_bars:
            if error:
                self.progress_labels[gpu_id]['text'] = f"GPU {gpu_id}: Error"
                self.progress_bars[gpu_id]['value'] = 0
            elif done:
                self.progress_labels[gpu_id]['text'] = f"GPU {gpu_id}: Done ({total} images)"
                self.progress_bars[gpu_id]['value'] = 100
            else:
                pct = (current / total * 100) if total > 0 else 0
                self.progress_labels[gpu_id]['text'] = f"GPU {gpu_id}: {current}/{total}"
                self.progress_bars[gpu_id]['value'] = pct

    def inference_complete(self):
        self.is_running = False
        self.run_button.configure(state=tk.NORMAL)
        self.processes = []


def main():
    # Required for Windows multiprocessing
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)

    root = tk.Tk()
    app = MultiGPUInferenceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
