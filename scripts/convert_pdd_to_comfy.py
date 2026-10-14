"""Convert a pdd_lora/step-XXXXXX export into the Alibaba MiniMax-H3 PDD file layout,
loadable by ComfyUI-MiniMax-H3-PDD (H3 PDD Acceleration node).

Keys are renamed, PDD metadata is added, LoRA tensors are cast to --lora-dtype, heads stay fp32.

    python scripts/convert_pdd_to_comfy.py pdd_lora/step-002500 --ema --block-size 4 \
        --output ComfyUI/models/loras/MiniMax-H3-PDD-step2500-ema.safetensors
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HEAD_KEYS = {"proj_out.weight", "proj_out.bias", "audio_proj_out.weight", "audio_proj_out.bias"}
# exact string checked by the ComfyUI node
COMFY_TARGETS = "to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2,adaln_proj.linear"


def comfy_key(key: str) -> str:
    if key in HEAD_KEYS:
        return key
    for src, dst in ((".lora_A.", ".lora_down"), (".lora_B.", ".lora_up")):
        if src in key:
            module, _ = key.split(src)  # drops "<adapter>.weight"
            return module + dst
    raise ValueError(f"unexpected key {key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lora_dir", type=Path, help="pdd_lora/step-XXXXXX export dir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ema", action="store_true", help="convert pdd_ema.safetensors")
    parser.add_argument("--block-size", type=int, default=4, help="NFE = trajectory_steps / block_size")
    parser.add_argument("--lora-dtype", choices=["bfloat16", "float32"], default="bfloat16",
                        help="dtype of LoRA tensors; heads stay fp32")
    args = parser.parse_args()

    config = json.loads((args.lora_dir / "pdd_config.json").read_text())
    if sorted(COMFY_TARGETS.split(",")) != sorted(config["lora_targets"]):
        raise ValueError(f"LoRA targets {config['lora_targets']} differ from {COMFY_TARGETS}")
    if (config["video_shift"], config["audio_shift"]) != (12.0, 3.0):
        print(f"warning: shifts {config['video_shift']}/{config['audio_shift']}, node expects 12/3")
    if config["trajectory_steps"] % args.block_size:
        raise ValueError(f"block size {args.block_size} does not divide {config['trajectory_steps']}")

    src = args.lora_dir / ("pdd_ema.safetensors" if args.ema else "pdd.safetensors")
    state_dict = load_file(src)
    lora_dtype = getattr(torch, args.lora_dtype)
    renamed = {comfy_key(k): v if k in HEAD_KEYS else v.to(lora_dtype) for k, v in state_dict.items()}
    if len(renamed) != len(state_dict):
        raise ValueError("key collision after renaming")
    metadata = {
        "format": "pt",
        "pdd_num_steps": str(config["trajectory_steps"]),
        "pdd_block_size": str(args.block_size),
        "lora_rank": str(config["lora_rank"]),
        "lora_alpha": str(config["lora_alpha"]),
        "lora_targets": COMFY_TARGETS,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(renamed, args.output, metadata=metadata)

    num_lora = sum(k.endswith(".lora_down") for k in renamed)
    print(f"{args.output}: {num_lora} LoRA modules, {config['trajectory_steps'] // args.block_size} NFE")


if __name__ == "__main__":
    main()
