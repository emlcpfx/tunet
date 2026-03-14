import os
import logging

import torch


def cycle(iterable):
    """Infinitely cycle through an iterable, restarting when exhausted."""
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def collate_skip_none(batch):
    """Collate function that filters out None items from a batch."""
    original_size = len(batch)
    batch = [item for item in batch if item is not None]
    filtered_size = len(batch)
    if filtered_size == 0:
        return None
    if filtered_size < original_size:
        logging.debug(f"Collate skipped {original_size - filtered_size}/{original_size} items.")
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e:
        logging.error(f"Collate error after filtering: {e}")
        return None


def auto_detect_num_workers(resolution, current_os):
    """Auto-detect optimal num_workers based on GPU VRAM, resolution, and CPU cores."""
    cpu_count = os.cpu_count() or 4
    max_from_cpu = max(1, cpu_count - 2)

    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            gpu_name = props.name
        except Exception:
            vram_gb = 0
            gpu_name = "unknown"

        if vram_gb >= 20:
            gpu_workers = 8
        elif vram_gb >= 10:
            gpu_workers = 4
        elif vram_gb >= 6:
            gpu_workers = 2
        else:
            gpu_workers = 1
        reason = f"CUDA {gpu_name} ({vram_gb:.0f}GB VRAM)"
    elif current_os == 'Darwin' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        gpu_workers = 2
        reason = "MPS (Apple Silicon)"
    else:
        return 0, "CPU-only (no GPU acceleration)"

    res_factor = 1.5 if resolution >= 1024 else (1.0 if resolution >= 512 else 0.75)
    ideal = int(gpu_workers * res_factor)
    workers = max(1, min(ideal, max_from_cpu))

    if current_os == 'Windows':
        workers = min(workers, 4)

    reason += f" | CPUs={cpu_count}, res={resolution} -> {workers} workers"
    return workers, reason
