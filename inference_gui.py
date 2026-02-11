import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging

# Import inference functions
from inference import load_model_and_config, process_image, denormalize, NORM_MEAN, NORM_STD

import torch
import torchvision.transforms as T


class InferenceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TuNet Inference")
        self.root.geometry("600x400")
        self.root.resizable(True, True)

        # Variables
        self.checkpoint_path = tk.StringVar()
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.overlap_factor = tk.DoubleVar(value=0.5)
        self.batch_size = tk.IntVar(value=1)
        self.use_amp = tk.BooleanVar(value=True)
        self.half_res = tk.BooleanVar(value=False)
        self.device = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")

        self.is_running = False

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

        # Device selection
        ttk.Label(options_frame, text="Device:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        device_combo = ttk.Combobox(options_frame, textvariable=self.device,
                                    values=["cuda", "cpu"], width=8, state="readonly")
        device_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # AMP checkbox
        ttk.Checkbutton(options_frame, text="Use AMP (faster)", variable=self.use_amp).grid(
            row=1, column=2, sticky=tk.W, padx=5, pady=5)

        # Half-res checkbox
        ttk.Checkbutton(options_frame, text="Half-res (~4x faster)", variable=self.half_res).grid(
            row=1, column=3, sticky=tk.W, padx=5, pady=5)

        # Run button
        row += 1
        self.run_button = ttk.Button(main_frame, text="Run Inference", command=self.run_inference)
        self.run_button.grid(row=row, column=0, columnspan=3, pady=15)

        # Progress bar
        row += 1
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=5)

        # Log output
        row += 1
        ttk.Label(main_frame, text="Log:").grid(row=row, column=0, sticky=tk.W)
        row += 1
        self.log_text = tk.Text(main_frame, height=8, width=70, state=tk.DISABLED)
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
        return True

    def run_inference(self):
        if self.is_running:
            return

        if not self.validate_inputs():
            return

        self.is_running = True
        self.run_button.configure(state=tk.DISABLED)
        self.progress.start()

        # Run in separate thread
        thread = threading.Thread(target=self.inference_thread, daemon=True)
        thread.start()

    def inference_thread(self):
        try:
            from glob import glob
            import re

            device_str = self.device.get()
            if device_str == 'cuda' and not torch.cuda.is_available():
                logging.warning("CUDA not available, falling back to CPU")
                device_str = 'cpu'
            device = torch.device(device_str)

            use_amp = self.use_amp.get() and device_str == 'cuda'

            logging.info(f"Using device: {device}")
            logging.info("Loading model...")

            model, resolution = load_model_and_config(self.checkpoint_path.get(), device)

            transform = T.Compose([T.ToTensor(), T.Normalize(mean=NORM_MEAN, std=NORM_STD)])

            overlap_pixels = int(resolution * self.overlap_factor.get())
            stride = max(1, resolution - overlap_pixels)

            logging.info(f"Resolution: {resolution}, Stride: {stride}")

            # Find images
            img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.webp']
            input_files = sorted([f for ext in img_extensions
                                  for f in glob(os.path.join(self.input_dir.get(), ext))])

            if not input_files:
                logging.error("No images found in input directory")
                return

            logging.info(f"Found {len(input_files)} images")
            os.makedirs(self.output_dir.get(), exist_ok=True)

            for i, img_path in enumerate(input_files):
                logging.info(f"Processing {i+1}/{len(input_files)}: {os.path.basename(img_path)}")

                basename = os.path.splitext(os.path.basename(img_path))[0]
                match = re.match(r'^(.+?)_?(\d+)$', basename)
                if match:
                    name_part = match.group(1)
                    frame_num = match.group(2)
                    output_filename = f"{name_part}_tunet_{frame_num}.png"
                else:
                    output_filename = f"{basename}_tunet.png"

                output_path = os.path.join(self.output_dir.get(), output_filename)

                process_image(model, img_path, output_path, resolution, stride, device,
                              self.batch_size.get(), transform, denormalize, use_amp, self.half_res.get())

            logging.info("Inference complete!")
            self.root.after(0, lambda: messagebox.showinfo("Done", "Inference completed successfully!"))

        except Exception as e:
            logging.error(f"Error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, self.inference_complete)

    def inference_complete(self):
        self.is_running = False
        self.run_button.configure(state=tk.NORMAL)
        self.progress.stop()


def main():
    root = tk.Tk()
    app = InferenceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
