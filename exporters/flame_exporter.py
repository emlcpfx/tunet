# converter.py (Revised for Cross-Platform support - Minimal Change)

import os
import sys
import argparse
import json
import logging

# Force UTF-8 stdout/stderr on Windows so PyTorch's emoji prints don't crash
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import re
from datetime import datetime

import torch
import torch.onnx
from types import SimpleNamespace


def _clean_export_name(base_name):
    """Strip checkpoint suffixes like _tunet_latest or _tunet_epoch_000000034
    and append a _MMDDYY_HHMM timestamp."""
    cleaned = re.sub(r'_tunet_(latest|epoch_\d+)', '', base_name)
    timestamp = datetime.now().strftime('_%m%d%y_%H%M')
    return cleaned + timestamp

# Ensure project root is on sys.path so imports work when run as a standalone script
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models import create_model
from models.normalized import NormalizedUNet
from config import dict_to_namespace

# --- Modified load_model_for_export ---
def load_model_for_export(checkpoint_path, device):
    """Loads TuNet checkpoint, determines model size *directly from metadata*,
       instantiates the model, loads weights, and wraps it for normalization.
    """
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # --- Determine Config Source (New vs Old Format) ---
    config_source_obj = None; is_new_format = False; config_format_detected = "Unknown"
    if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
        config_format_detected = "New (dict under 'config' key)"
        logging.info(f"Detected format: {config_format_detected}")
        try: config_source_obj = dict_to_namespace(checkpoint['config']); is_new_format = True
        except Exception as e: raise ValueError(f"Config dict conversion failed: {e}") from e
    elif 'args' in checkpoint and isinstance(checkpoint['args'], argparse.Namespace):
        config_format_detected = "Old (argparse.Namespace under 'args' key)"
        logging.warning(f"Detected format: {config_format_detected}. Attempting compatibility.")
        config_source_obj = SimpleNamespace(**vars(checkpoint['args'])); is_new_format = False
    else: raise ValueError("Checkpoint missing required configuration metadata ('config' dict or 'args' namespace).")
    if config_source_obj is None: raise ValueError("Conf obj could not be loaded.")

    # --- Extract Parameters Safely ---
    logging.info("Extracting parameters from checkpoint metadata...")
    default_hidden_size = 64; default_resolution = 512; default_bilinear = True
    model_config = getattr(config_source_obj, 'model', SimpleNamespace())
    data_config = getattr(config_source_obj, 'data', SimpleNamespace())

    # Get model size from checkpoint metadata, preferring effective_model_size (accounts for training-time bumps)
    effective_size = checkpoint.get('effective_model_size', None)
    if effective_size is not None and effective_size > 0:
        model_size_saved = effective_size; size_source = "'effective_model_size' (top-level checkpoint key)"
    else:
        model_size_saved = getattr(model_config, 'model_size_dims', default_hidden_size); size_source = "'model.model_size_dims'"
        if model_size_saved == default_hidden_size:
            legacy_size = getattr(model_config, 'unet_hidden_size', default_hidden_size)
            if legacy_size != default_hidden_size: model_size_saved = legacy_size; size_source = "'model.unet_hidden_size' (legacy fallback)"
            else: size_source += " (or fallback, using default)"
    logging.info(f"  - Using Model Size from Checkpoint: {model_size_saved} (loaded via {size_source})")

    # Extract resolution (needed for dummy input)
    resolution = getattr(data_config, 'resolution', default_resolution); res_source = "'data.resolution'"
    if resolution == default_resolution:
        legacy_res = getattr(config_source_obj, 'resolution', default_resolution)
        if legacy_res != default_resolution: resolution = legacy_res; res_source = "'resolution' (legacy fallback)"
        else: res_source += " (or fallback, using default)"
    logging.info(f"  - Resolution: {resolution} (from {res_source})")

    # Extract bilinear mode
    bilinear_mode = getattr(model_config, 'bilinear', default_bilinear); bilinear_source = "'model.bilinear'"
    if bilinear_mode == default_bilinear:
         legacy_bilinear = getattr(config_source_obj, 'bilinear', default_bilinear)
         if legacy_bilinear != default_bilinear: bilinear_mode = legacy_bilinear; bilinear_source = "'bilinear' (legacy fallback)"
         else: bilinear_source += " (or fallback, using default)"
    logging.info(f"  - Bilinear Mode: {bilinear_mode} (from {bilinear_source})")

    # Extract loss mode for logging purposes (not used for size calculation here)
    training_config = getattr(config_source_obj, 'training', SimpleNamespace())
    loss_mode = getattr(training_config, 'loss', 'l1'); loss_source = "'training.loss'"
    if loss_mode == 'l1' and not is_new_format:
        quality_legacy = getattr(config_source_obj, 'quality', None)
        if quality_legacy == 'HQ': loss_mode = 'l1+lpips'; loss_source = "'quality'=='HQ' (legacy)"
        elif quality_legacy == 'LQ': loss_mode = 'l1'; loss_source = "'quality'=='LQ' (legacy)"
    logging.info(f"  - Loss Mode (for info): '{loss_mode}' (from {loss_source})")

    # --- Instantiate the Model ---
    model_type_saved = checkpoint.get('model_type', 'unet')
    if model_type_saved == 'unet' and is_new_format:
        model_type_saved = getattr(model_config, 'model_type', 'unet')
    recurrence_t = getattr(model_config, 'recurrence_steps', 2) if is_new_format else 2
    logging.info(f"Instantiating {model_type_saved.upper()} with hidden_size={model_size_saved} (from checkpoint metadata), bilinear={bilinear_mode}")
    try:
        base_model = create_model(model_type=model_type_saved, n_ch=3, n_cls=3, hidden_size=model_size_saved, bilinear=bilinear_mode, t=recurrence_t)
    except Exception as e:
        logging.error(f"Failed to instantiate {model_type_saved.upper()} model: {e}", exc_info=True)
        raise RuntimeError(f"Model instantiation failed with size {model_size_saved}") from e

    # --- Load State Dict ---
    if 'model_state_dict' not in checkpoint: raise KeyError("Checkpoint missing 'model_state_dict'.")
    state_dict = checkpoint['model_state_dict']
    if all(key.startswith('module.') for key in state_dict):
        logging.info("Removing 'module.' prefix from state_dict keys."); state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    elif any(key.startswith('module.') for key in state_dict): logging.warning("Mixed state_dict keys ('module.' prefix). Loading anyway...")
    logging.info("Loading state dict into UNet model (strict=True)...")
    try:
        base_model.load_state_dict(state_dict, strict=True)
        logging.info("State dict loaded successfully (strict=True).")
    except RuntimeError as e:
        logging.error(f"Failed state_dict load (strict=True): {e}")
        logging.info("Attempting load with strict=False...")
        try:
            incompatible_keys = base_model.load_state_dict(state_dict, strict=False)
            # <<< --- Rever correcao rapida --- >>>
            if incompatible_keys.missing_keys:
                logging.warning(f"Missing keys (strict=False): {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys:
                logging.warning(f"Unexpected keys (strict=False): {incompatible_keys.unexpected_keys}")
            logging.warning("State dict loaded (strict=False). Check warnings.")
            # <<< --- TO DO rever --- >>>
        except Exception as e_nonstrict:
            logging.error(f"Failed load (strict=False): {e_nonstrict}", exc_info=True)
            raise RuntimeError("Could not load model state_dict.") from e_nonstrict
    except Exception as e:
        logging.error(f"Unexpected state_dict loading error: {e}", exc_info=True)
        raise RuntimeError("Could not load model state_dict.") from e

    # --- Instantiate the WRAPPER ---
    use_sigmoid = (loss_mode == 'bce+dice')
    logging.info(f"Wrapping UNet model with Normalization layer (use_sigmoid={use_sigmoid})...")
    wrapped_model = NormalizedUNet(base_model, use_sigmoid=use_sigmoid)
    logging.info(f"Moving wrapped model to device: {device}")
    wrapped_model.to(device)
    wrapped_model.eval()
    logging.info("Model ready (loaded, wrapped, device, eval mode).")

    return wrapped_model, resolution # Return resolution needed for dummy input

# --- JSON Generation Function  ---
def generate_flame_json(onnx_base_name, resolution, output_json_path, model_name=None, model_desc=""):
    if model_name is None: model_name = onnx_base_name
    input_name = "input_image"; output_name = "output_image"
    flame_data = { "ModelDescription": { "MinimumVersion": "2025.1", "Name": model_name, "Description": model_desc + " (Normalized: [0,1] input/output)", "SupportsSceneLinear": False, "KeepAspectRatio": False, "Padding": 1, "Inputs": [{ "Name": input_name, "Description": "Source Image ([0,1] Range)", "Type": "Front", "Gain": 1.0, "Channels": "RGB" }], "Outputs": [{ "Name": output_name, "Description": "Processed Image ([0,1] Range)", "Type": "Result", "InverseGain": 1.0, "ScalingFactor": 1.0, "Channels": "RGB" }] } }
    try:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f: json.dump(flame_data, f, indent=4)
        logging.info(f"Generated Flame JSON configuration: {output_json_path}")
    except Exception as e: logging.error(f"Failed to write Flame JSON file '{output_json_path}': {e}", exc_info=True)


# --- Main Conversion Function  ---
def convert(args):
    # ... Setup, path determination, device selection ...
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if not os.path.isfile(args.checkpoint): logging.error(f"Checkpoint file not found: {args.checkpoint}"); return 1
    checkpoint_dir = os.path.dirname(args.checkpoint); checkpoint_basename = os.path.basename(args.checkpoint)
    raw_base = os.path.splitext(checkpoint_basename)[0]
    base_name = _clean_export_name(raw_base)
    output_dir = args.output_dir if args.output_dir else checkpoint_dir
    output_onnx_path = args.output_onnx if args.output_onnx else os.path.join(output_dir, f"{base_name}.onnx")
    output_json_path = args.output_json if args.output_json else os.path.join(output_dir, f"{base_name}.json")
    try: os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True); os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    except OSError as e: logging.error(f"Error creating output directories: {e}"); return 1
    logging.info(f"--- Conversion Settings ---"); logging.info(f"Input Checkpoint: {args.checkpoint}"); logging.info(f"Output ONNX Path: {output_onnx_path}"); logging.info(f"Output JSON Path: {output_json_path}"); logging.info(f"ONNX Opset: {args.opset}"); logging.info(f"Use GPU if available: {args.use_gpu}"); logging.info(f"Dynamic Batch Size: {args.dynamic_batch}"); logging.info(f"-------------------------")
    use_cuda = torch.cuda.is_available() and args.use_gpu; device = torch.device('cuda' if use_cuda else 'cpu')
    logging.info(f"Using device: {device}")

    # Load model using the revised function
    try:
        model, resolution = load_model_for_export(args.checkpoint, device)
        logging.info(f"Model loaded. Using resolution from checkpoint: {resolution}x{resolution}")
    except Exception as e: logging.error(f"Fatal error during model loading: {e}", exc_info=True); return 1

    # Create dummy input based on loaded resolution
    try: dummy_input = torch.rand(1, 3, resolution, resolution, device=device); logging.info(f"Created dummy input tensor: {list(dummy_input.shape)}")
    except Exception as e: logging.error(f"Failed to create dummy input (Res: {resolution}): {e}", exc_info=True); return 1

    # Define ONNX export parameters
    input_names = ["input_image"]; output_names = ["output_image"]
    dynamic_axes_config = None
    if args.dynamic_batch:
        dynamic_axes_config = { input_names[0]: {0: 'batch_size'}, output_names[0]: {0: 'batch_size'} }; logging.info("Dynamic batch size enabled for ONNX export.")

    # Export to ONNX
    logging.info(f"Starting ONNX export to: {output_onnx_path}")
    try:
        model.eval() # Ensure eval mode
        torch.onnx.export( model, dummy_input, output_onnx_path, export_params=True, opset_version=args.opset, do_constant_folding=True, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes_config )
        logging.info("ONNX export completed successfully.")

        # Generate JSON Sidecar
        json_base_name = os.path.splitext(os.path.basename(output_onnx_path))[0]
        generate_flame_json( onnx_base_name=json_base_name, resolution=resolution, output_json_path=output_json_path, model_name=args.model_name, model_desc=args.model_desc )
        logging.info("Conversion process finished successfully.")
        return 0
    except Exception as e:
        logging.error(f"ONNX export or JSON generation failed: {e}", exc_info=True)
        # --- Attempt to clean up potentially incomplete ONNX file ---
        if os.path.exists(output_onnx_path):
            try:
                os.remove(output_onnx_path)
                logging.info(f"Removed potentially incomplete ONNX file: {output_onnx_path}")
            except OSError as remove_e: # Catch specific OS errors during removal
                logging.warning(f"Could not remove incomplete ONNX file '{output_onnx_path}': {remove_e}")
        return 1

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PyTorch TuNet checkpoint to ONNX for Flame (with internal normalization). Handles new/old config formats.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the PyTorch checkpoint (.pth)')
    parser.add_argument('--output_onnx', type=str, default=None, help='Path to save the output ONNX model (.onnx). Defaults to same name/dir.')
    parser.add_argument('--output_json', type=str, default=None, help='Path to save the Flame JSON sidecar file (.json). Defaults to same name/dir.')
    parser.add_argument('--output_dir', type=str, default=None, help='Optional directory to save outputs if specific paths not given.')
    parser.add_argument('--opset', type=int, default=18, help='ONNX opset version. Default: 18')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for loading model if available.')
    parser.add_argument('--dynamic_batch', action='store_true', help='Enable dynamic batch size in the exported ONNX model.')
    parser.add_argument('--model_name', type=str, default=None, help='Name for the model in Flame UI (defaults to ONNX filename base).')
    parser.add_argument('--model_desc', type=str, default="TuNet by tpo, converted model.", help='Description for the model in Flame UI.')
    args = parser.parse_args()

    exit_code = convert(args)
    exit(exit_code)
