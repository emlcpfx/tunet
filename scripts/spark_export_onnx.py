"""Standalone ONNX export job for Spark Compute.

Rebuilds the model from a checkpoint's bundled metadata (no training config
needed on disk — train.py stamps `config`, `model_type`, `effective_model_size`,
`n_input_channels` into every .pth) and calls the in-tree exporters to write:

    <output_dir>/exports/flame/<prefix>_epoch_<NNNN>_<MMDDYY_HHMM>.onnx + .json
    <output_dir>/exports/nuke/<prefix>_epoch_<NNNN>_<MMDDYY_HHMM>.pt + .nk + .cat

Used by the "Export now" button in tunet-web — see
tunet-web/src/app/api/spark/jobs/[id]/export-onnx/route.ts.

Args:
    --checkpoint  Path to the .pth on the container (fetched by spark_export.sh)
    --output-dir  Where to drop the exports/ tree (Spark uploads /output/ for us)
    --epoch       Optional override; defaults to the value baked into the .pth
"""

import argparse
import logging
import os
import sys
from types import SimpleNamespace

import torch


def _dict_to_namespace(d):
    """Recursive dict -> SimpleNamespace; matches train.py's dict_to_sns."""
    if isinstance(d, dict):
        return SimpleNamespace(**{str(k).replace('-', '_'): _dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, (list, tuple)):
        return type(d)(_dict_to_namespace(x) for x in d)
    return d


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Path to .pth on disk')
    p.add_argument('--output-dir', required=True, help='Output dir for exports/ tree')
    p.add_argument('--epoch', type=int, default=None, help='Override epoch number used in filenames')
    p.add_argument('--ckpt-prefix', default=None, help='Override checkpoint prefix (filename stem)')
    args = p.parse_args()

    if not os.path.isfile(args.checkpoint):
        logging.error(f"Checkpoint not found: {args.checkpoint}")
        return 1

    logging.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    cfg_dict = ckpt.get('config') or {}
    config = _dict_to_namespace(cfg_dict) if cfg_dict else SimpleNamespace()
    config.data = getattr(config, 'data', SimpleNamespace())
    config.training = getattr(config, 'training', SimpleNamespace())
    config.model = getattr(config, 'model', SimpleNamespace())

    model_type = ckpt.get('model_type') or getattr(config.model, 'model_type', 'unet')
    n_input_ch = ckpt.get('n_input_channels') or 3
    eff_size = ckpt.get('effective_model_size') or getattr(config.model, 'model_size_dims', 64)
    recurrence_t = getattr(config.model, 'recurrence_steps', 2)
    loss_mode = getattr(config.training, 'loss', 'l1')
    color_space = getattr(config.data, 'color_space', 'srgb')
    resolution = getattr(config.data, 'resolution', 512)
    epoch = args.epoch if args.epoch is not None else int(ckpt.get('epoch') or 0)

    logging.info(
        f"Rebuilding {model_type.upper()} hidden={eff_size} n_ch={n_input_ch} "
        f"t={recurrence_t} loss={loss_mode} color={color_space} res={resolution} epoch={epoch}"
    )

    from models import create_model

    model = create_model(model_type=model_type, n_ch=n_input_ch, n_cls=3,
                         hidden_size=eff_size, t=recurrence_t)

    state = ckpt['model_state_dict']
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    # Use checkpoint stem as the prefix (e.g. "babies_resume_v7_instant_tunet_latest"
    # becomes "babies_resume_v7_instant"); strips the _tunet_latest / _tunet_epoch
    # suffix so the export filenames look sensible.
    ckpt_prefix = args.ckpt_prefix or _infer_prefix(args.checkpoint)
    logging.info(f"Using ckpt_prefix={ckpt_prefix}")

    from exporters.auto_export import export_flame, export_nuke

    export_flame(model, config, args.output_dir, epoch, resolution,
                 loss_mode=loss_mode, ckpt_prefix=ckpt_prefix)
    export_nuke(model, config, args.output_dir, epoch, resolution,
                loss_mode=loss_mode, ckpt_prefix=ckpt_prefix)

    logging.info("Export complete.")
    return 0


def _infer_prefix(checkpoint_path: str) -> str:
    """`<prefix>_tunet_latest.pth` / `<prefix>_tunet_epoch_NNNNNNNNN.pth` -> `<prefix>`."""
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    for marker in ('_tunet_latest', '_tunet_plateau', '_tunet_epoch_'):
        idx = stem.find(marker)
        if idx > 0:
            return stem[:idx]
    return stem


if __name__ == '__main__':
    sys.exit(main())
