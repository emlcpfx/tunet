"""
Merge two tunet .pth checkpoints by interpolating model weights.
Usage: python merge_models.py [--alpha 0.5]
  alpha=1.0 → 100% model A, alpha=0.0 → 100% model B
"""

import argparse
import torch
import os

MODEL_A = r"Q:\Work\Bouffant\models\PaintHands_finetune_Vanity\PaintHands_finetune_Vanity_tunet_latest.pth"
MODEL_B = r"D:\Work\Dropbox\Documents_DB\Work_DB\Bouffant_db\tunet_models\hands_finetune_trainingdata_0320\finetune_hands_eric\finetune_hands_eric_tunet_latest.pth"
OUTPUT = r"Q:\Work\Bouffant\models\PaintHands_finetune_Vanity\merged_model.pth"


def merge(alpha: float):
    print(f"Loading model A: {os.path.basename(MODEL_A)}")
    ckpt_a = torch.load(MODEL_A, map_location="cpu")
    print(f"Loading model B: {os.path.basename(MODEL_B)}")
    ckpt_b = torch.load(MODEL_B, map_location="cpu")

    state_a = ckpt_a["model_state_dict"]
    state_b = ckpt_b["model_state_dict"]

    print(f"Merging {len(state_a)} weight tensors (alpha={alpha:.2f} A + {1-alpha:.2f} B)...")
    merged_state = {}
    for key in state_a:
        merged_state[key] = alpha * state_a[key] + (1 - alpha) * state_b[key]

    # Use model A's checkpoint as the base, replace weights
    ckpt_a["model_state_dict"] = merged_state
    ckpt_a["epoch"] = 0
    ckpt_a["global_step"] = 0
    # Clear optimizer/scheduler state since it's no longer valid
    ckpt_a["optimizer_state_dict"] = None
    ckpt_a["scaler_state_dict"] = None

    print(f"Saving merged model to: {OUTPUT}")
    torch.save(ckpt_a, OUTPUT)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Blend ratio: 1.0=all A, 0.0=all B (default: 0.5)")
    args = parser.parse_args()
    merge(args.alpha)
