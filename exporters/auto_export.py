"""In-process model export for use during training.

Exports the current model to Flame/AE (ONNX) and/or Nuke (TorchScript)
without reloading from disk. Called at epoch boundaries by train.py.
"""

import os
import json
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from models.normalized import NormalizedUNet


def _timestamp_suffix():
    """Return a timestamp suffix like _031526_1430 (MMDDYY_HHMM)."""
    return datetime.now().strftime('_%m%d%y_%H%M')


def export_flame(model, config, output_dir, epoch, resolution, loss_mode='l1', ckpt_prefix='model'):
    """Export model to ONNX + JSON for Flame / After Effects.

    Args:
        model: The training model (raw or DDP-wrapped).
        config: Training config namespace.
        output_dir: Directory to write exports into.
        epoch: Current epoch number (for filename).
        resolution: Model training resolution.
        loss_mode: Loss type ('l1', 'l1+lpips', 'bce+dice', etc.).
        ckpt_prefix: Prefix for output filenames.
    """
    export_subdir = os.path.join(output_dir, 'exports', 'flame')
    os.makedirs(export_subdir, exist_ok=True)

    base_name = f'{ckpt_prefix}_epoch_{epoch:04d}{_timestamp_suffix()}'
    onnx_path = os.path.join(export_subdir, f'{base_name}.onnx')
    json_path = os.path.join(export_subdir, f'{base_name}.json')

    try:
        # Unwrap DDP if needed
        raw_model = model.module if isinstance(model, DDP) else model

        # Clone weights into a fresh model on CPU to avoid disturbing training
        import copy
        base_model_cpu = copy.deepcopy(raw_model).cpu()
        base_model_cpu.eval()

        use_sigmoid = (loss_mode == 'bce+dice')
        color_space = getattr(getattr(config, 'data', None), 'color_space', 'srgb') if config else 'srgb'
        wrapped = NormalizedUNet(base_model_cpu, use_sigmoid=use_sigmoid, color_space=color_space)
        wrapped.eval()

        dummy_input = torch.rand(1, 3, resolution, resolution)

        torch.onnx.export(
            wrapped, dummy_input, onnx_path,
            export_params=True, opset_version=18,
            do_constant_folding=True,
            input_names=['input_image'],
            output_names=['output_image'],
        )

        # Generate Flame JSON sidecar
        flame_data = {
            "ModelDescription": {
                "MinimumVersion": "2025.1",
                "Name": base_name,
                "Description": f"TuNet auto-export epoch {epoch} (Normalized: [0,1] input/output)",
                "SupportsSceneLinear": False,
                "KeepAspectRatio": False,
                "Padding": 1,
                "Inputs": [{
                    "Name": "input_image",
                    "Description": "Source Image ([0,1] Range)",
                    "Type": "Front", "Gain": 1.0, "Channels": "RGB"
                }],
                "Outputs": [{
                    "Name": "output_image",
                    "Description": "Processed Image ([0,1] Range)",
                    "Type": "Result", "InverseGain": 1.0,
                    "ScalingFactor": 1.0, "Channels": "RGB"
                }]
            }
        }
        with open(json_path, 'w') as f:
            json.dump(flame_data, f, indent=4)

        logging.info(f"Auto-export Flame: {onnx_path}")

    except Exception as e:
        logging.error(f"Auto-export Flame failed (epoch {epoch}): {e}", exc_info=False)
        # Clean up partial file
        if os.path.exists(onnx_path):
            try:
                os.remove(onnx_path)
            except OSError:
                pass


def export_nuke(model, config, output_dir, epoch, resolution, loss_mode='l1', ckpt_prefix='model'):
    """Export model to TorchScript (.pt) + Nuke helper script (.nk) for Nuke.

    Args:
        model: The training model (raw or DDP-wrapped).
        config: Training config namespace.
        output_dir: Directory to write exports into.
        epoch: Current epoch number (for filename).
        resolution: Model training resolution.
        loss_mode: Loss type.
        ckpt_prefix: Prefix for output filenames.
    """
    export_subdir = os.path.join(output_dir, 'exports', 'nuke')
    os.makedirs(export_subdir, exist_ok=True)

    base_name = f'{ckpt_prefix}_epoch_{epoch:04d}{_timestamp_suffix()}'
    pt_path = os.path.join(export_subdir, f'{base_name}.pt')
    nk_path = os.path.join(export_subdir, f'{base_name}.nk')
    cat_path = os.path.join(export_subdir, f'{base_name}.cat')

    try:
        # Unwrap DDP if needed
        raw_model = model.module if isinstance(model, DDP) else model

        import copy
        base_model_cpu = copy.deepcopy(raw_model).cpu()
        base_model_cpu.eval()

        use_sigmoid = (loss_mode == 'bce+dice')
        color_space = getattr(getattr(config, 'data', None), 'color_space', 'srgb') if config else 'srgb'
        wrapped = NormalizedUNet(base_model_cpu, use_sigmoid=use_sigmoid, color_space=color_space)
        wrapped.eval()

        scripted = torch.jit.script(wrapped)
        scripted.save(pt_path)

        # Generate .nk helper script
        from exporters.nuke_exporter import generate_nuke_script
        generate_nuke_script(pt_path, cat_path, nk_path)

        logging.info(f"Auto-export Nuke: {pt_path}")

    except Exception as e:
        logging.error(f"Auto-export Nuke failed (epoch {epoch}): {e}", exc_info=False)
        if os.path.exists(pt_path):
            try:
                os.remove(pt_path)
            except OSError:
                pass
