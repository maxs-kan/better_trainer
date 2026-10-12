import argparse
import html
import inspect
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import rootutils
import torch
from diffusers import ComponentsManager, MiniMaxH3ModularPipeline
from diffusers.modular_pipelines import InputParam, OutputParam, PipelineState
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3SetTimestepsStep,
)
from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopDenoiser
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
    MiniMaxH3Blocks,
)
from diffusers.utils.export_utils import encode_video
from safetensors import safe_open
from safetensors.torch import load_file

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.pdd_datamodule import get_data_enties
from src.models.net.minimax_h3 import HEAD_NAMES, time_grid


def build_html(output: Path, prompts: list[dict], runs: list["Run"]) -> None:
    def cell(name: str, index: int) -> str:
        if not (output / name / f"{index:03d}.mp4").exists():
            return '<td class="missing">missing</td>'
        return (f'<td data-col="{html.escape(name)}"><video src="{name}/{index:03d}.mp4" muted loop '
                f'playsinline preload="metadata"></video></td>')

    head = "".join(
        f'<th data-col="{html.escape(run.name)}">{html.escape(run.name)}<div class="sub">{run.nfe} NFE</div></th>'
        for run in runs
    )
    toggles = "".join(
        f'<label><input type="checkbox" checked data-toggle="{html.escape(run.name)}"> {html.escape(run.name)}</label>'
        for run in runs
    )
    rows = "".join(
        f'<tr><td class="prompt"><b>#{p["index"]}</b> <button class="row-play">▶ row</button>'
        f'<div>{html.escape(p["prompt"])}</div><div class="sub">seed {p["seed"]}</div></td>'
        + "".join(cell(run.name, p["index"]) for run in runs) + "</tr>"
        for p in prompts
    )
    page = PAGE.replace("{{HEAD}}", head).replace("{{ROWS}}", rows).replace("{{TOGGLES}}", toggles)
    tmp = output / f".index.html.{os.getpid()}"
    tmp.write_text(page)
    os.replace(tmp, output / "index.html")


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>H3 PDD Compare</title>
<style>
:root { --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --line:#e5e5ea; --panel:#f5f5f7; --accent:#0a66d8; }
@media (prefers-color-scheme: dark) { :root { --bg:#111214; --fg:#ececf1; --muted:#9a9aa3; --line:#2a2b30; --panel:#1b1c20; --accent:#5aa2ff; } }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; }
header { position:sticky; top:0; z-index:2; background:var(--panel); border-bottom:1px solid var(--line);
         padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; }
header label { white-space:nowrap; } button { cursor:pointer; }
.wrap { overflow-x:auto; padding:0 16px 40px; }
table { border-collapse:collapse; } th, td { border-bottom:1px solid var(--line); padding:8px; vertical-align:top; }
th { position:sticky; top:0; text-align:left; font-weight:600; }
td.prompt { width:260px; min-width:200px; font-size:13px; }
video { width:var(--w, 420px); display:block; border-radius:6px; background:#000; cursor:pointer; }
video.loud { outline:3px solid var(--accent); }
.sub, .cap { color:var(--muted); font-size:12px; font-weight:400; } .missing { color:var(--muted); }
.hidden { display:none; }
</style></head><body>
<header>
  <button id="play">▶ Play all</button><button id="pause">❚❚ Pause</button><button id="restart">⟲ Restart</button>
  <label>Width <input id="width" type="range" min="200" max="900" value="420"></label>
  <span class="sub">click a video = sound on for it only · ▶ row = restart that row in sync</span>
  <span>{{TOGGLES}}</span>
</header>
<div class="wrap"><table><thead><tr><th>prompt</th>{{HEAD}}</tr></thead><tbody>{{ROWS}}</tbody></table></div>
<script>
const videos = () => [...document.querySelectorAll('td:not(.hidden) video')];
const sync = (list) => { list.forEach(v => { v.currentTime = 0; }); list.forEach(v => v.play()); };
document.getElementById('play').onclick = () => videos().forEach(v => v.play());
document.getElementById('pause').onclick = () => document.querySelectorAll('video').forEach(v => v.pause());
document.getElementById('restart').onclick = () => sync(videos());
document.getElementById('width').oninput = (e) => document.body.style.setProperty('--w', e.target.value + 'px');
document.querySelectorAll('.row-play').forEach(b => b.onclick = () => {
  document.querySelectorAll('video').forEach(v => v.pause());
  sync([...b.closest('tr').querySelectorAll('td:not(.hidden) video')]);
});
document.querySelectorAll('video').forEach(v => v.onclick = () => {
  const on = v.muted;
  document.querySelectorAll('video').forEach(o => { o.muted = true; o.classList.remove('loud'); });
  if (on) { v.muted = false; v.classList.add('loud'); if (v.paused) v.play(); }
});
document.querySelectorAll('[data-toggle]').forEach(c => c.onchange = () => {
  document.querySelectorAll(`[data-col="${CSS.escape(c.dataset.toggle)}"]`).forEach(el => el.classList.toggle('hidden', !c.checked));
});
</script></body></html>
"""


def load_cached_prompts(index_file: Path, limit: int, seed: int) -> list[dict]:
    prompts = []
    archives = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, (path, member) in enumerate(get_data_enties(index_file)[:limit]):
            if path not in archives:
                archives[path] = zipfile.ZipFile(path)
            with safe_open(archives[path].extract(member, tmp), framework="pt") as entry:
                prompts.append(dict(
                    index=i,
                    prompt=(entry.metadata() or {}).get("prompt", member),
                    seed=seed + i,
                    prompt_embeds=entry.get_tensor("prompt_embeds"),
                    text_token_tags=entry.get_tensor("text_token_tags"),
                ))
    for archive in archives.values():
        archive.close()
    return prompts


def load_pdd_lora(lora_dir: Path, ema: bool, device: str) -> tuple[dict, dict[str, torch.Tensor]]:
    config = json.loads((lora_dir / "pdd_config.json").read_text())
    file = lora_dir / ("pdd_ema.safetensors" if ema else "pdd.safetensors")
    return config, load_file(file, device=device)


class MidpointSetTimestep(MiniMaxH3SetTimestepsStep):
    @property
    def inputs(self):
        return super().inputs + [
            InputParam(
                name="midpoint",
                type_hint=bool,
                default=False,
                description="Midpoint solver (2 NFE per step)."),
            InputParam(
                name="video_sigmas",
                type_hint=torch.Tensor,
                default=None,
                description="Video sheduler sigmas"),
            InputParam(
                name="audio_sigmas",
                type_hint=torch.Tensor,
                default=None,
                description="Audio sheduler sigmas")
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return super().intermediate_outputs + [
            OutputParam(
                "mid_timestep_plan",
                type_hint=list,
                description=(
                    "One `(timestep, timestep_indices)` pair per midpoint step"
                ),
            ),
            OutputParam("video_mid_time_steps", type_hint=torch.Tensor, description="Video half steps dt / 2."),
            OutputParam("audio_mid_time_steps", type_hint=torch.Tensor, description="Audio half steps dt / 2."),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        if block_state.video_sigmas is not None:
            components.scheduler.set_timesteps(sigmas=block_state.video_sigmas, device=device)
        else:
            components.scheduler.set_timesteps(block_state.num_inference_steps, device=device)
        if block_state.audio_sigmas is not None:
            components.audio_scheduler.set_timesteps(sigmas=block_state.audio_sigmas, device=device)
        else:
            components.audio_scheduler.set_timesteps(block_state.num_inference_steps, device=device)

        block_state.timesteps = components.scheduler.timesteps
        block_state.audio_timesteps = components.audio_scheduler.timesteps
        block_state.row_timestep_plan = self.timestep_plan(
            components, block_state, block_state.timesteps, block_state.audio_timesteps
        )

        block_state.mid_timestep_plan = None
        block_state.video_mid_time_steps = None
        block_state.audio_mid_time_steps = None
        if block_state.midpoint:
            # t = 1 - sigma, so the step dt = sigma_i - sigma_{i+1}.
            block_state.video_mid_time_steps = -components.scheduler.sigmas.diff() * 0.5
            block_state.audio_mid_time_steps = -components.audio_scheduler.sigmas.diff() * 0.5
            block_state.mid_timestep_plan = self.timestep_plan(
                components,
                block_state,
                block_state.timesteps + block_state.video_mid_time_steps,
                block_state.audio_timesteps + block_state.audio_mid_time_steps,
            )

        self.set_block_state(state, block_state)
        return components, state

    def timestep_plan(self, components, block_state, video_timesteps, audio_timesteps) -> list[tuple]:
        device = components._execution_device
        return [
            tuple(
                tensor.to(device)
                for tensor in self.build_row_timesteps(
                    block_state.video_indices,
                    block_state.audio_indices,
                    block_state.num_condition_video_rows,
                    block_state.num_condition_audio_rows,
                    block_state.text_indices.numel(),
                    float(timestep),
                    float(audio_timestep),
                    max(float(timestep), components.keyframe_noise_aug),
                    1.0,
                )
            )
            for timestep, audio_timestep in zip(video_timesteps, audio_timesteps)
        ]


class MidpointLoopDenoiser(MiniMaxH3LoopDenoiser):

    @property
    def inputs(self) -> list[InputParam]:
        return super().inputs + [
            InputParam(name="midpoint", type_hint=bool, default=False, description="Midpoint solver (2 NFE per step)."),
            InputParam(name="mid_timestep_plan", type_hint=list, description="Midpoint `(timestep, timestep_indices)`."),
            InputParam(name="video_mid_time_steps", type_hint=torch.Tensor, description="Video half steps dt / 2."),
            InputParam(name="audio_mid_time_steps", type_hint=torch.Tensor, description="Audio half steps dt / 2."),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state, i: int, t: torch.Tensor):
        transformer = getattr(components, self.transformer_name)
        unique_timesteps, timestep_indices = block_state.row_timestep_plan[i]
        # The layout tags its outputs `denoiser_input_fields`, and their names are the transformer's own parameter
        # names, so the rows of the packed sequence are described to it without this block enumerating them.
        layout_kwargs = {
            name: value
            for name, value in block_state.denoiser_input_fields.items()
            if name in inspect.signature(transformer.forward).parameters
        }
        noise_pred, audio_noise_pred = transformer(
            hidden_states=block_state.latents[None],
            audio_hidden_states=block_state.audio_latents[None],
            encoder_hidden_states=block_state.prompt_embeds,
            timestep=unique_timesteps,
            timestep_indices=timestep_indices,
            attention_kwargs=block_state.attention_kwargs,
            return_dict=False,
            **layout_kwargs,
        )
        if not block_state.midpoint:
            block_state.noise_pred, block_state.audio_noise_pred = noise_pred, audio_noise_pred
        else:
            video_time_step = block_state.video_mid_time_steps[i]
            audio_time_step = block_state.audio_mid_time_steps[i]
            mid_timesteps, mid_indices = block_state.mid_timestep_plan[i]

            video_mid = block_state.latents[None] + video_time_step * noise_pred.float()
            audio_mid = block_state.audio_latents[None] + audio_time_step * audio_noise_pred.float()

            block_state.noise_pred, block_state.audio_noise_pred = transformer(
                hidden_states=video_mid.to(block_state.latents.dtype),
                audio_hidden_states=audio_mid.to(block_state.audio_latents.dtype),
                encoder_hidden_states=block_state.prompt_embeds,
                timestep=mid_timesteps,
                timestep_indices=mid_indices,
                attention_kwargs=block_state.attention_kwargs,
                return_dict=False,
                **layout_kwargs,
            )
        return components, block_state


class PDDWrapper:

    def __init__(self, pipe, config: dict, state_dict: dict[str, torch.Tensor], block_size: int):
        self.pipe = pipe
        transformer = pipe.transformer
        heads = {k: v for k, v in state_dict.items() if k.split(".")[0] in HEAD_NAMES}
        lora = {k: v for k, v in state_dict.items() if k not in heads}
        pipe.load_lora_weights(lora, adapter_name="pdd")

        self.trajectory_steps = config["trajectory_steps"]
        if self.trajectory_steps % block_size:
            raise ValueError(f"block size {block_size} does not divide trajectory_steps {self.trajectory_steps}")
        self.block_size = block_size
        self.num_blocks = self.trajectory_steps // block_size
        self.video_times = time_grid(config["video_shift"], config["trajectory_steps"])
        self.audio_times = time_grid(config["audio_shift"], config["trajectory_steps"])
        self.video_sigmas = (1 - self.video_times[::block_size]).float()
        self.audio_sigmas = (1 - self.audio_times[::block_size]).float()

        self.block_heads = self.fuse_heads(heads)
        self.params = {
            f"{head}.{name}": getattr(getattr(transformer, head), name) for head in HEAD_NAMES for name in ("weight", "bias")
        }
        self.base_heads = {key: param.data for key, param in self.params.items()}

        self._call = 0
        self.handle = transformer.register_forward_pre_hook(self._swap_heads)

    def fuse_heads(self, heads: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        fused = [{} for _ in range(self.num_blocks)]
        for head, time in zip(HEAD_NAMES, (self.video_times, self.audio_times)):
            weight, bias = heads[f"{head}.weight"], heads[f"{head}.bias"]
            dt = time.diff().float().view(-1, self.block_size).to(weight.device)
            delta = dt / dt.sum(1, keepdim=True)
            fused_weight = torch.einsum("sl,sloi->soi", delta, weight.view(-1, self.block_size, *weight.shape[1:]))
            fused_bias = torch.einsum("sl,slo->so", delta, bias.view(-1, self.block_size, *bias.shape[1:]))
            for i, block_heads in enumerate(fused):
                block_heads[f"{head}.weight"], block_heads[f"{head}.bias"] = fused_weight[i], fused_bias[i]
        return fused

    def reset(self) -> None:
        self._call = 0

    def _set_heads(self, heads: dict[str, torch.Tensor]) -> None:
        for key, tensor in heads.items():
            self.params[key].data = tensor

    def _swap_heads(self, module, args) -> None:
        if self._call >= self.num_blocks:
            raise RuntimeError(f"transformer call {self._call} past the {self.num_blocks} PDD blocks")
        self._set_heads(self.block_heads[self._call])
        self._call = self._call + 1

    def remove(self) -> None:
        self.handle.remove()
        self._set_heads(self.base_heads)
        self.pipe.unload_lora_weights()
        self._call = 0


@dataclass
class Run:
    name: str
    nfe: int
    midpoint: bool = False
    lora_dir: Path | None = None
    ema: bool = False
    block_size: int | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="MiniMax-H3 diffusers folder")
    parser.add_argument("--prompt-cache", type=Path, required=True, help="index.json of scripts/cache_h3_prompts.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lora", type=Path, action="append", default=[], help="pdd_lora/step-XXXXXX export dir")
    parser.add_argument("--base-nfe", type=int, nargs="*", default=[32, 8], help="Steps")
    parser.add_argument("--base-midpoint-steps", type=int, nargs="*", default=[32], help="midpoint steps (2 NFE each)")
    parser.add_argument("--block-size", type=int, nargs="*", default=[4, 8], help="PDD block sizes (grid steps per NFE)")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=124)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    prompts = load_cached_prompts(args.prompt_cache, args.num_prompts, args.seed)
    runs = [Run(f"base_{n}nfe", n) for n in args.base_nfe]
    runs += [Run(f"base_{n}x2_midpoint_{2 * n}nfe", n, midpoint=True) for n in args.base_midpoint_steps]
    for lora_dir in args.lora:
        config = json.loads((lora_dir / "pdd_config.json").read_text())
        for block_size in args.block_size:
            nfe = config["trajectory_steps"] // block_size
            runs.append(Run(f"pdd_{lora_dir.name}_L{block_size}_{nfe}nfe", nfe, lora_dir=lora_dir, block_size=block_size))
            runs.append(Run(f"pdd_{lora_dir.name}_ema_L{block_size}_{nfe}nfe", nfe, lora_dir=lora_dir, ema=True, block_size=block_size))
    args.output.mkdir(parents=True, exist_ok=True)
    for run in runs:
        (args.output / run.name).mkdir(exist_ok=True)

    blocks = MiniMaxH3Blocks().get_workflow("t2va")
    blocks.sub_blocks.pop("text_encoder")
    blocks.sub_blocks["denoise.set_timesteps"] = MidpointSetTimestep()
    blocks.sub_blocks["denoise.denoise"].sub_blocks["denoiser"] = MidpointLoopDenoiser()
    manager = ComponentsManager()
    pipe = blocks.init_pipeline(args.model, components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)
    manager.enable_auto_cpu_offload(device=device, memory_reserve_margin="12GB")

    for run in runs:
        todo = [p for p in prompts[rank::world_size] if not (args.output / run.name / f"{p['index']:03d}.mp4").exists()]
        if not todo:
            continue
        pdd = None
        if run.lora_dir is not None:
            pdd = PDDWrapper(pipe, *load_pdd_lora(run.lora_dir, run.ema, str(device)), block_size=run.block_size)
        for p in todo:
            if pdd is not None:
                pdd.reset()
            out = pipe(
                prompt_embeds=p["prompt_embeds"][None].to(device),
                text_token_tags=p["text_token_tags"],
                num_frames=args.num_frames,
                num_inference_steps=run.nfe + 1,  # sigma grid points, terminal 0 included
                midpoint=run.midpoint,
                video_sigmas=pdd.video_sigmas if pdd is not None else None,
                audio_sigmas=pdd.audio_sigmas if pdd is not None else None,
                generator=torch.Generator().manual_seed(p["seed"]),
                output=["videos", "audio", "sampling_rate"],
            )
            path = args.output / run.name / f"{p['index']:03d}.mp4"
            encode_video(out["videos"][0], fps=24, output_path=str(path), audio=out["audio"][0],
                         audio_sample_rate=out["sampling_rate"])
            print(f"[{run.name}] {path}", flush=True)
        if pdd is not None:
            pdd.remove()

    build_html(args.output, prompts, runs)
    print(f"[rank {rank}] wrote {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
