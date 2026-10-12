"""Cache MiniMax-H3 Qwen3-VL prompt conditioning for data-free PDD training (text-to-video+audio).

Keeps the ~62 GB text encoder out of the training run. Every prompt becomes a `<i:07d>.safetensors` entry holding
`prompt_embeds` (bf16, `(L, 5120)`, `hidden_states[50]` of the conditioner) and `text_token_tags` (int64, `(L,)`),
encoded as the diffusers `MiniMaxH3TextEncoderStep` does it (prompt tokenized verbatim, no chat template or special
tokens, through `get_qwen3vl_prompt_embeds`; every row tagged text). Entries are packed into
uncompressed zip shards `<output>/shard-<s:05d>.zip` of `--shard-size` consecutive prompts (bf16 activations do not
compress; ~10 KB per token, ~1 MB for a 100-token prompt). `<output>/index.json` lists `{"file_path", "member"}` per
prompt, in prompt order.

    torchrun --nproc_per_node 8 scripts/cache_h3_prompts.py \
        --model /path/to/MiniMax-H3 --prompts prompts.jsonl --output /data/h3_prompt_cache/train

Prompts: `.txt` (one per line), `.json` (list of strings or of objects), `.jsonl`, `.csv`, `.parquet`; objects and
tables are read from `--column` (default: the first of `text`, `prompt`, `caption`). Shards are written to `.tmp` and
renamed when complete, and existing shards are skipped, so a killed run resumes (losing at most the open shard per rank).
Re-run with the same `--shard-size`.
"""
import argparse
import json
import os
import zipfile
import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save
from tqdm import tqdm
from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MINIMAX_H3_TEXT_TAG
from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor


TEXT_COLUMNS = ("text", "prompt", "caption")
# MiniMax-H3 conditions on `hidden_states[50]` of Qwen3-VL (`MiniMaxH3ModularPipeline.text_encoder_layer`).
TEXT_ENCODER_LAYER = 50


def _from_record(record, column: str | None, where: str) -> str:
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        keys = [column] if column else [k for k in TEXT_COLUMNS if k in record]
        if keys and isinstance(record.get(keys[0]), str):
            return record[keys[0]]
    raise ValueError(f"{where}: cannot read a prompt from {record!r}")


def load_prompts(path: Path, column: str | None) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        prompts = path.read_text(encoding="utf-8").splitlines()
    elif suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        prompts = [_from_record(r, column, f"{path}[{i}]") for i, r in enumerate(records)]
    elif suffix == ".jsonl":
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        prompts = [_from_record(json.loads(line), column, f"{path}:{i + 1}") for i, line in enumerate(lines)]
    elif suffix in (".csv", ".parquet"):
        import pandas as pd

        frame = pd.read_csv(path) if suffix == ".csv" else pd.read_parquet(path)
        name = column or next((c for c in TEXT_COLUMNS if c in frame.columns), None)
        if name is None:
            raise ValueError(f"{path}: no prompt column among {TEXT_COLUMNS}, pass --column")
        prompts = frame[name].astype(str).tolist()
    else:
        raise ValueError(f"Unsupported prompt file {path}")
    prompts = [p.strip() for p in prompts]
    empty = [i for i, p in enumerate(prompts) if not p]
    if empty:
        raise ValueError(f"{path}: {len(empty)} empty prompts, e.g. at {empty[:5]}")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="MiniMax-H3 diffusers folder (tokenizer/, processor/, text_encoder/)")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--column", type=str, default=None)
    parser.add_argument("--shard-size", type=int, default=5000, help="prompts per zip shard")
    args = parser.parse_args()


    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=1), device_id=device)

    prompts = load_prompts(args.prompts, args.column)
    args.output.mkdir(parents=True, exist_ok=True)
    members = [f"{i:07d}.safetensors" for i in range(len(prompts))]
    num_shards = (len(prompts) + args.shard_size - 1) // args.shard_size
    shards = [args.output / f"shard-{s:05d}.zip" for s in range(num_shards)]
    todo = [s for s in range(rank, num_shards, world_size) if not shards[s].exists()]

    if todo:
        tokenizer = Qwen2TokenizerFast.from_pretrained(args.model / "tokenizer")
        processor = Qwen3VLProcessor.from_pretrained(args.model / "processor")
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model / "text_encoder", dtype=torch.bfloat16, device_map={"": device}
        ).eval()
        progress = tqdm(
            total=sum(min(args.shard_size, len(prompts) - s * args.shard_size) for s in todo),
            disable=rank != 0,
            desc="Caching prompts",
        )
        with torch.no_grad():
            for s in todo:
                tmp = shards[s].with_suffix(".zip.tmp")
                with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as archive:
                    for i in range(s * args.shard_size, min((s + 1) * args.shard_size, len(prompts))):
                        token_ids = tokenizer(prompts[i], add_special_tokens=False)["input_ids"]
                        prompt_embeds = get_qwen3vl_prompt_embeds(
                            text_encoder, processor, token_ids, {}, TEXT_ENCODER_LAYER, device=device, dtype=torch.bfloat16
                        )
                        text_token_tags = torch.full((len(token_ids),), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
                        data = save(
                            {
                                "prompt_embeds": prompt_embeds[0].cpu().contiguous(),
                                "text_token_tags": text_token_tags.cpu().contiguous(),
                            },
                            metadata={"prompt": prompts[i]},
                        )
                        archive.writestr(members[i], data)
                        progress.update()
                os.replace(tmp, shards[s])
        progress.close()

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        missing = [p.name for p in shards if not p.exists()]
        if missing:
            raise RuntimeError(f"{len(missing)} shards were not written, e.g. {missing[:3]}")
        index = [
            {"file_path": shards[i // args.shard_size].name, "member": members[i]} for i in range(len(prompts))
        ]
        (args.output / "index.json").write_text(json.dumps(index, indent=0))
        print(f"Cached {len(prompts)} prompts in {num_shards} shards to {args.output}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
