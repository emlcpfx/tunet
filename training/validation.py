import logging

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from distributed import is_main_process
from .loss import dice_loss


def run_validation(model, val_dataloader, device, device_type, use_amp, use_lpips,
                   loss_fn_lpips, use_bce_dice, criterion_l1, current_ep_idx,
                   global_step, lambda_lpips=1.0, use_mask_input=False):
    """Run validation over entire val dataset, return avg losses."""
    if not is_main_process() or val_dataloader is None:
        return None, None
    model.eval()
    model_module = model.module if isinstance(model, DDP) else model
    total_l1, total_lp, count = 0.0, 0.0, 0
    use_amp_inf = use_amp and device_type == 'cuda'
    try:
        with torch.no_grad():
            for batch in val_dataloader:
                if batch is None:
                    continue
                src, dst = batch[0].to(device), batch[1].to(device)
                with autocast(device_type=device_type, enabled=use_amp_inf):
                    model_in = src
                    if use_mask_input:
                        dummy_mask = torch.zeros(src.shape[0], 1, src.shape[2], src.shape[3], device=device)
                        model_in = torch.cat([model_in, dummy_mask], dim=1)
                    out = model_module(model_in)
                    if use_bce_dice:
                        dst_01 = dst * 0.5 + 0.5
                        n_ch = out.shape[1]
                        bce_total = torch.tensor(0.0, device=device)
                        dc_total = torch.tensor(0.0, device=device)
                        criterion_bce = nn.BCEWithLogitsLoss()
                        for ch in range(n_ch):
                            out_ch = out[:, ch:ch + 1, :, :]
                            dst_ch = dst_01[:, ch:ch + 1, :, :]
                            bce_total = bce_total + criterion_bce(out_ch, dst_ch)
                            dc_total = dc_total + dice_loss(torch.sigmoid(out_ch), dst_ch)
                        l1 = (bce_total + dc_total) / n_ch
                        lp = torch.tensor(0.0, device=device)
                    else:
                        l1 = criterion_l1(out, dst)
                        lp = torch.tensor(0.0, device=device)
                        if use_lpips and loss_fn_lpips:
                            try:
                                lp = loss_fn_lpips(out, dst).mean()
                            except Exception:
                                lp = torch.tensor(0.0, device=device)
                total_l1 += l1.item()
                total_lp += lp.item()
                count += 1
    except Exception as e:
        logging.error(f"Validation error: {e}")
        model.train()
        return None, None
    model.train()
    if count == 0:
        return None, None
    avg_l1, avg_lp = total_l1 / count, total_lp / count
    loss_label = 'BCE+Dice' if use_bce_dice else 'L1'
    log_msg = f'Val Epoch[{current_ep_idx + 1}] Step[{global_step}], Val_{loss_label}:{avg_l1:.4f}'
    if use_lpips:
        log_msg += f', Val_LPIPS:{avg_lp:.4f}'
    logging.info(log_msg)
    return avg_l1, avg_lp
