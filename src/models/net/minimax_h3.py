import contextlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_VIDEO_TAG,
    align_num_frames,
    audio_latent_num_frames,
    video_latent_num_frames,
)
from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors import safe_open
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from ...schemas.base_module_schemas import _float_dtype

HEAD_NAMES = ("proj_out", "audio_proj_out")
_WRAPPER_SEGMENTS = ("_checkpoint_wrapped_module", "_orig_mod", "base_layer")


def build_meta_transformer(
    pretrained_model_name_or_path: str,
    subfolder: str = "transformer",
    dtype: str = 'bfloat16',
) -> MiniMaxH3Transformer3DModel:
    with torch.device("meta"):
        config = MiniMaxH3Transformer3DModel.load_config(pretrained_model_name_or_path, subfolder=subfolder)
        net = MiniMaxH3Transformer3DModel.from_config(config)
        dtype = _float_dtype('load model', dtype)
        keep_fp32 = MiniMaxH3Transformer3DModel._keep_in_fp32_modules
        for name, tensor in list(net.named_parameters()) + list(net.named_buffers()):
            tensor.data = tensor.data.to(torch.float32 if any(name.startswith(k) for k in keep_fp32) else dtype)
    return net


def shifted_sigma(shift: float, sigma: torch.Tensor) -> torch.Tensor:
    return shift * sigma / (1 + (shift - 1) * sigma)


def time_grid(shift: float, num_steps: int) -> torch.Tensor:
    sigma = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return 1.0 - shifted_sigma(shift, sigma)


def model_shapes(checkpoint_dir: Path, transformer_subfolder: str) -> dict:
    vae = json.loads((checkpoint_dir / "vae" / "config.json").read_text())
    transformer = json.loads((checkpoint_dir / transformer_subfolder / "config.json").read_text())
    clip_length = vae["clip_length"]
    return dict(
        spatial_downsample=math.prod(vae["spatial_downsample_factors"]),
        vae_chunk=(clip_length, math.ceil(clip_length / math.prod(vae["temporal_downsample_factors"]))),
        patch_size=tuple(transformer["patch_size"]),
        in_channels=transformer["in_channels"],
        audio_in_channels=transformer["audio_in_channels"],
    )


class T2VALayout:
    def __init__(
        self,
        num_frames: int,
        height: int,
        width: int,
        checkpoint_dir: str | Path,
        device: torch.device,
        transformer_subfolder: str = "transformer",
    ):
        checkpoint_dir = Path(checkpoint_dir)
        
        self.check_canvas(checkpoint_dir, num_frames, height, width, transformer_subfolder)
        shapes = model_shapes(checkpoint_dir, transformer_subfolder)
        self.patch_size = shapes["patch_size"]
        self.latent_shape = (
            video_latent_num_frames(num_frames, *shapes["vae_chunk"]),
            height // shapes["spatial_downsample"],
            width // shapes["spatial_downsample"],
            audio_latent_num_frames(num_frames),
        )
        
        num_latent_frames, latent_height, latent_width, num_audio_latents = self.latent_shape
        patch_t, patch_h, patch_w = self.patch_size
        video_rows = (num_latent_frames // patch_t) * (latent_height // patch_h) * (latent_width // patch_w)
        
        self.video_shape = (1, video_rows, shapes["in_channels"] * patch_t * patch_h * patch_w)
        self.audio_shape = (1, num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS, shapes["audio_in_channels"])
        self.device = device

        self._indices: dict[str, torch.Tensor] | None = None
        self._cpu_indices: tuple[torch.Tensor, torch.Tensor, int] | None = None

    @staticmethod
    def check_canvas(
        checkpoint_dir: str | Path, num_frames: int, height: int, width: int, transformer_subfolder: str = "transformer"
    ) -> None:
        shapes = model_shapes(Path(checkpoint_dir), transformer_subfolder)
        vae_chunk, patch_size = shapes["vae_chunk"], shapes["patch_size"]
        aligned = align_num_frames(num_frames, *vae_chunk)
        if aligned != num_frames:
            raise ValueError(f"num_frames must be {vae_chunk[0]}n+{vae_chunk[1]}, got {num_frames} (nearest {aligned})")
        
        multiple_h = shapes["spatial_downsample"] * patch_size[1]
        multiple_w = shapes["spatial_downsample"] * patch_size[2]
        if height % multiple_h or width % multiple_w:
            raise ValueError(f"height / width must be multiples of {multiple_h} / {multiple_w}, got {height}x{width}")

    def register(self, text_token_tags: torch.Tensor) -> None:
        """Pack a request's text rows (`(L,)` modality tags) in front of the canvas rows; replaces any previous request."""
        num_latent_frames, latent_height, latent_width, num_audio_latents = self.latent_shape
        (
            position_ids,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            _,
            _
            ) = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
                    text_token_tags=text_token_tags.cpu(),
                    num_latent_frames=num_latent_frames,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    num_audio_latents=num_audio_latents,
                    patch_size=self.patch_size,
                    audio_channels=MINIMAX_H3_AUDIO_CHANNELS,
                    audio_tag=MINIMAX_H3_AUDIO_TAG,
                    video_tag=MINIMAX_H3_VIDEO_TAG,
                )
        
        self._cpu_indices = (video_indices, audio_indices, text_indices.numel())
        self._indices = {
            "token_tags": token_tags.to(self.device),
            "position_ids": position_ids.to(self.device),
            "video_indices": video_indices.to(self.device),
            "audio_indices": audio_indices.to(self.device),
            "text_indices": text_indices.to(self.device),
        }

    def check_registered(self) -> None:
        if self._indices is None:
            raise RuntimeError("T2VALayout has no request: call register(text_token_tags) first")

    def forward_kwargs(self, video_time: float, audio_time: float) -> dict:
        self.check_registered()
        video_indices, audio_indices, num_text_tokens = self._cpu_indices
        timestep, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            video_indices=video_indices,
            audio_indices=audio_indices,
            num_condition_video_rows=0,
            num_condition_audio_rows=0,
            num_text_tokens=num_text_tokens,
            video_timestep=float(video_time),
            audio_timestep=float(audio_time),
            condition_video_timestep=1.0,
            condition_audio_timestep=1.0
        )
        return dict(
                timestep=timestep.to(self.device),
                timestep_indices=timestep_indices.to(self.device),
                return_dict=False,
                **self._indices,
            )


class MultiStepHead(nn.Module):
    def __init__(self, source: nn.Linear, num_steps: int):
        super().__init__()
        self.num_steps = num_steps
        self.in_features = source.in_features
        self.out_features = source.out_features
        weight = source.weight.detach()
        bias = None if source.bias is None else source.bias.detach()
        self.weight = nn.Parameter(weight[None].repeat(num_steps, 1, 1).clone())
        self.bias = None if bias is None else nn.Parameter(bias[None].repeat(num_steps, 1).clone())
        self.register_buffer("teacher_weight", weight.clone(), persistent=False)
        self.register_buffer("teacher_bias", None if bias is None else bias.clone(), persistent=False)
        self.teacher = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.teacher:
            return F.linear(hidden_states, self.teacher_weight, self.teacher_bias)
        bias = None if self.bias is None else self.bias.flatten()
        hidden_states = F.linear(hidden_states, self.weight.flatten(0, 1), bias)
        hidden_states = hidden_states.unflatten(-1, (self.num_steps, self.out_features))
        return hidden_states


def attach_multistep_heads(net: MiniMaxH3Transformer3DModel, num_steps: int) -> None:
    for name in HEAD_NAMES:
        head = MultiStepHead(getattr(net, name), num_steps)
        setattr(net, name, head)


@contextlib.contextmanager
def teacher_mode(net: nn.Module):
    heads = [m for m in net.modules() if isinstance(m, MultiStepHead)]
    adapters = [m for m in net.modules() if isinstance(m, BaseTunerLayer)]
    for m in heads:
        m.teacher = True
    for m in adapters:
        m._disable_adapters = True
    try:
        yield net
    finally:
        for m in heads:
            m.teacher = False
        for m in adapters:
            m._disable_adapters = False


@torch.no_grad()
def teacher_mean_velocity(
    forward_fn,
    video: torch.Tensor,
    audio: torch.Tensor,
    video_time: float,
    audio_time: float,
    video_time_step: float,
    audio_time_step: float,
    solver: str,
) -> tuple[torch.Tensor, torch.Tensor]:

    velocity_video, velocity_audio = forward_fn(video, audio, video_time, audio_time)
    if solver == "euler":
        return velocity_video.float(), velocity_audio.float()
    if solver != "midpoint":
        raise ValueError(f"Unknown solver {solver}")

    video_mid = video + video_time_step / 2 * velocity_video.float()
    audio_mid = audio + audio_time_step / 2 * velocity_audio.float()

    velocity_video, velocity_audio = forward_fn(
        video_mid,
        audio_mid,
        video_time + 0.5 * video_time_step,
        audio_time + 0.5 * audio_time_step,
    )
    return velocity_video.float(), velocity_audio.float()


def checkpoint_key(name: str) -> str:
    return ".".join(part for part in name.split(".") if part not in _WRAPPER_SEGMENTS)

def checkpoint_files(checkpoint_dir: Path) -> dict[str, Path]:
    index = checkpoint_dir / "diffusion_pytorch_model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"]
    return {key: checkpoint_dir / file for key, file in weight_map.items()}

def _local_view(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    if not isinstance(tensor, DTensor):
        return tensor.data, 0
    shape, offset = compute_local_shape_and_global_offset(tensor.shape, tensor.device_mesh, tensor.placements)
    if any(offset[1:]) or tuple(shape[1:]) != tuple(tensor.shape[1:]):
        raise RuntimeError(f"Only dim 0 sharding is supported")
    local = tensor.to_local()
    return local, offset[0]


@torch.no_grad()
def load_checkpoint_to_device(net: nn.Module, checkpoint_dir: str | Path) -> None:
    """Fill the materialized `net` from a diffusers checkpoint folder, reading only each rank's local rows.

    Every parameter and persistent buffer is matched by name (`checkpoint_key`). The multi-step heads are filled by
    repeating their source layer and also get it as the teacher buffers. LoRA parameters are skipped (see
    `init_lora_parameters`). Missing or unused checkpoint keys raise.
    """
    checkpoint_dir = Path(checkpoint_dir)
    files = checkpoint_files(checkpoint_dir)

    targets: dict[str, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    heads = {name: module for name, module in net.named_modules() if isinstance(module, MultiStepHead)}
    for name, param in net.named_parameters():
        if ".lora_" in name:
            continue
        block_name = name.rpartition(".")[0]
        kind = "head" if block_name in heads else "tensor"
        targets[checkpoint_key(name)].append((kind, param))
    for block_name, head in heads.items():
        prefix = checkpoint_key(block_name)
        targets[f"{prefix}.weight"].append(("buffer", head.teacher_weight))
        if head.teacher_bias is not None:
            targets[f"{prefix}.bias"].append(("buffer", head.teacher_bias))
    for module_name, module in net.named_modules():
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer_name in module._non_persistent_buffers_set:
                continue
            name = f"{module_name}.{buffer_name}" if module_name else buffer_name
            targets[checkpoint_key(name)].append(("buffer", buffer))

    missing = sorted(set(targets) - set(files))
    if missing:
        raise KeyError(f"{len(missing)} model tensors are not in {checkpoint_dir}, e.g. {missing[:5]}")
    unused = sorted(set(files) - set(targets))
    if unused:
        raise KeyError(f"{len(unused)} checkpoint tensors match no model tensor, e.g. {unused[:5]}")

    by_file: dict[Path, list[str]] = defaultdict(list)
    for key in targets:
        by_file[files[key]].append(key)

    for file, keys in by_file.items():
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in keys:
                source = handle.get_slice(key)
                source_shape = tuple(source.get_shape())
                for kind, tensor in targets[key]:
                    local, start = _local_view(tensor)
                    if kind == "head":
                        if tuple(tensor.shape[1:]) != source_shape:
                            raise ValueError(f"{key}: head shape {tensor.shape} vs checkpoint {source_shape}")
                        row = handle.get_tensor(key).to(local.device, local.dtype)
                        local.copy_(row.unsqueeze(0).expand_as(local))
                        continue
                    if tuple(tensor.shape) != source_shape:
                        raise ValueError(f"{key}: model shape {tensor.shape} vs checkpoint {source_shape}")
                    if local.numel() == 0:
                        continue
                    if isinstance(tensor, DTensor):
                        rows = source[start:start + local.shape[0]]
                    else:
                        rows = handle.get_tensor(key)
                    local.copy_(rows.to(local.device, local.dtype))


@torch.no_grad()
def init_lora_parameters(net: nn.Module, init_lora_weights: bool | str = True) -> None:
    for module in net.modules():
        if not isinstance(module, LoraLayer):
            continue
        for adapter in module.lora_A.keys():
            lora_a, _ = _local_view(module.lora_A[adapter].weight)
            lora_b, _ = _local_view(module.lora_B[adapter].weight)
            if init_lora_weights is True:
                # kaiming_uniform_(a=sqrt(5)) of the full matrix: fan_in is the (unsharded) input dim.
                bound = 1.0 / math.sqrt(lora_a.shape[1])
                lora_a.uniform_(-bound, bound)
            elif init_lora_weights == "gaussian":
                lora_a.normal_(0.0, 1.0 / module.r[adapter])
            else:
                raise ValueError(f"Unsupported init_lora_weights={init_lora_weights}")
            lora_b.zero_()
            if module.lora_B[adapter].bias is not None:
                lora_b_bias, _ = _local_view(module.lora_B[adapter].bias)
                lora_b_bias.zero_()


@torch.no_grad()
def init_non_persistent_buffers(net: MiniMaxH3Transformer3DModel) -> None:    
    for module in net.modules():
        if isinstance(module, MiniMaxH3RotaryPosEmbed):
            rope_theta = float(net.config.rope_theta)
            inv_freq = 1.0 / (
                rope_theta ** (torch.arange(0, 2 * module.rope_freq_dim, 2, dtype=torch.float32) / (2 * module.rope_freq_dim))
            )
            module.inv_freq.copy_(inv_freq)
