import os
import re
import glob
import logging


def prune_checkpoints(output_dir, keep_last, ckpt_prefix=None):
    if keep_last < 0:
        return
    if keep_last == 0:
        logging.info("Pruning all epoch checkpoints (keep=0).")
    try:
        dir_prefix = ckpt_prefix or os.path.basename(os.path.normpath(output_dir))
        epoch_files = glob.glob(os.path.join(output_dir, f'{dir_prefix}_tunet_epoch_*.pth'))
        epoch_files += glob.glob(os.path.join(output_dir, 'tunet_epoch_*.pth'))
        epoch_files = list(set(epoch_files))
        ckpt_files_info = []
        for f_path in epoch_files:
            basename = os.path.basename(f_path)
            match = re.match(r"(?:.+_)?tunet_epoch_(\d+)\.pth", basename)
            if match:
                epoch_num = int(match.group(1))
                try:
                    mtime = os.path.getmtime(f_path)
                    ckpt_files_info.append({'path': f_path, 'epoch': epoch_num, 'mtime': mtime})
                except OSError as e:
                    logging.warning(f"Pruning: skip {basename}, cannot get mtime: {e}")
        ckpt_files_info.sort(key=lambda x: (x['epoch'], x['mtime']), reverse=True)
        if len(ckpt_files_info) <= keep_last:
            return
        files_to_remove = ckpt_files_info[keep_last:]
        logging.info(f"Pruning: Keeping last {keep_last} checkpoints. Removing {len(files_to_remove)}.")
        removed_count = 0
        for ckpt_info in files_to_remove:
            try:
                os.remove(ckpt_info['path'])
                logging.debug(f"  Removed: {os.path.basename(ckpt_info['path'])}")
                removed_count += 1
            except Exception as e:
                logging.warning(f"  Failed remove {os.path.basename(ckpt_info['path'])}: {e}")
        if removed_count > 0:
            logging.info(f"Pruning finished. Removed {removed_count} checkpoint(s).")
    except Exception as e:
        logging.error(f"Checkpoint pruning error in '{output_dir}': {e}", exc_info=True)
