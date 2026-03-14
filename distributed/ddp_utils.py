import os
import logging
import platform

import torch
import torch.distributed as dist

CURRENT_OS = platform.system()


def setup_ddp():
    """Initialize the DDP process group if needed."""
    if not dist.is_available():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        print(f"INFO: PyTorch distributed not available. Running in single-process mode on {CURRENT_OS}.")
        return

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    if world_size == 1:
        logging.info("Running in single-process mode (World Size = 1).")
        if CURRENT_OS == 'Darwin' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            logging.debug("Single process mode: MPS available.")
        elif torch.cuda.is_available():
            try:
                torch.cuda.set_device(local_rank)
                logging.debug(f"Single process mode: CUDA device set to cuda:{local_rank}")
            except Exception as e:
                logging.error(f"Error setting CUDA device cuda:{local_rank}: {e}")
        else:
            logging.debug("Single process mode: No CUDA or MPS available.")
        return

    if not dist.is_initialized():
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        if CURRENT_OS == 'Darwin' and not torch.cuda.is_available():
            backend = 'gloo'
            logging.debug("Forcing Gloo backend for DDP on macOS (No CUDA detected).")

        try:
            init_method = f'tcp://{master_addr}:{master_port}'
            dist.init_process_group(backend=backend, init_method=init_method,
                                    world_size=world_size, rank=rank)

            if CURRENT_OS == 'Darwin' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device_info = "mps"
            elif torch.cuda.is_available():
                device_info = f"cuda:{local_rank}"
            else:
                device_info = "cpu"

            if rank == 0:
                logging.info(f"DDP Initialized: Rank {rank}/{world_size} using {device_info} (Backend: {backend})")

        except RuntimeError as e:
            if 'nccl' in str(e).lower() and backend == 'nccl':
                logging.warning(f"NCCL backend failed (OS: {CURRENT_OS}). Trying Gloo...")
                backend = 'gloo'
                try:
                    dist.init_process_group(backend=backend, init_method=init_method,
                                            world_size=world_size, rank=rank)
                    if rank == 0:
                        logging.info(f"DDP Re-Initialized using Gloo backend: Rank {rank}/{world_size}")
                except Exception as e_gloo:
                    logging.error(f"Gloo backend also failed on Rank {rank}: {e_gloo}", exc_info=True)
                    raise RuntimeError(f"DDP Initialization failed on Rank {rank} with both NCCL and Gloo.") from e_gloo
            else:
                logging.error(f"DDP initialization failed on Rank {rank} (Backend: {backend}): {e}", exc_info=True)
                raise RuntimeError(f"DDP Initialization failed on Rank {rank}.") from e
        except Exception as e:
            logging.error(f"Unexpected error during DDP initialization on Rank {rank}: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected DDP Initialization error on Rank {rank}.") from e


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0
