import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import MiniMaxH3Transformer3DModel
from lightning.pytorch.utilities import rank_zero_info
from safetensors.torch import save_file
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from ..callbacks.ema import EMA, EMA_KEY
from .module.fsdp_mixin import FSDPMixin
from .module.pdd_module import PDDModule, PDDTrajectory
from .net.minimax_h3 import (
    T2VALayout,
    attach_multistep_heads,
    init_lora_parameters,
    init_non_persistent_buffers,
    load_checkpoint_to_device,
    teacher_mean_velocity,
    teacher_mode,
    time_grid,
)

NET_NAME = "transformer"


class MiniMaxH3Trajectory:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        transformer_subfolder: str,
        trajectory_steps: int,
        num_frames: int,
        height: int,
        width: int,
        L_min: int,
        device: torch.device,
        dtype=torch.float32,
    ):

        p = Path(pretrained_model_name_or_path)
        self.layout = T2VALayout(num_frames, height, width, p, device, transformer_subfolder)
        
        video_shift = json.loads((p / "scheduler" / "scheduler_config.json").read_text())["shift"]
        audio_shift = json.loads((p / "audio_scheduler" / "scheduler_config.json").read_text())["shift"]
        self.shifts = (video_shift, audio_shift)
        video_time_grid = time_grid(video_shift, trajectory_steps)
        audio_time_grid = time_grid(audio_shift, trajectory_steps)

        self.video = PDDTrajectory(video_time_grid, L_min, self.layout.video_shape, device, dtype)
        self.audio = PDDTrajectory(audio_time_grid, L_min, self.layout.audio_shape, device, dtype)

        self.prompt_id: int | None = None
        self.prompt_embeds: torch.Tensor | None = None
        self._registered = False

    def init_trajectory(self, batch, generator):
        self.prompt_id = int(batch["index"])
        self.prompt_embeds = batch["prompt_embeds"]
        self.layout.register(batch["text_token_tags"])
        self._reset(generator)
        self._registered = True

    def _reset(self, generator):
        self.video.reset(generator)
        self.audio.reset(generator)
        self._registered = False

    def check_registered(self):
        if not self._registered:
            raise RuntimeError("Attempt to empty state")

    def update(self, video_pred: torch.Tensor, audio_pred: torch.Tensor):
        self.check_registered()
        self.video.update(video_pred)
        self.audio.update(audio_pred)

    @property
    def done(self) -> bool:
        return self.video.done

    @property
    def current_index_step(self) -> int:
        self.check_registered()
        return self.video.current_index_step

    def times(self, index: int) -> tuple[float, float, float, float]:
        self.check_registered()
        return (
            self.video.time_grid[index],
            self.audio.time_grid[index],
            self.video.time_diff[index],
            self.audio.time_diff[index],
        )

    def model_inputs(
            self,
            video: torch.Tensor,
            audio: torch.Tensor,
            video_time: float,
            audio_time: float
        ) -> dict:
        self.check_registered()
        return dict(
            hidden_states=video,
            audio_hidden_states=audio,
            encoder_hidden_states=self.prompt_embeds,
            **self.layout.forward_kwargs(video_time, audio_time),
        )

    def current_inputs(self) -> dict:
        self.check_registered()
        video_time, audio_time, _, _ = self.times(self.current_index_step)
        return self.model_inputs(self.video.state, self.audio.state, video_time, audio_time)


#TODO: add num_targets_step
#TODO: manage sync gpu-cpu
class MiniMaxH3PDDTrainer(FSDPMixin, PDDModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        transformer_subfolder: str = "transformer",
        trajectory_steps: int = 32,
        L_min: int = 4,
        L_max: int | None = None,
        solver: str = "midpoint",
        video_loss_weight: float = 0.5,
        audio_loss_weight: float = 0.5,
        num_frames: int = 124,
        height: int = 768,
        width: int = 1344,
        seed: int = 42,
        **kwargs,
    ) -> None:

        L_max = L_max or L_min
        super().__init__(trajectory_steps=trajectory_steps, L_min=L_min, L_max=L_max, **kwargs)

        if list(self.net_configs) != [NET_NAME]:
            raise ValueError(f"Expects exactly one net named '{NET_NAME}', got {list(self.net_configs)}")

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.transformer_subfolder = transformer_subfolder
        self.solver = solver
        self.video_loss_weight = video_loss_weight
        self.audio_loss_weight = audio_loss_weight
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.seed = seed

        T2VALayout.check_canvas(pretrained_model_name_or_path, num_frames, height, width, transformer_subfolder)

        self.trajectory: MiniMaxH3Trajectory | None = None
        self._generator: torch.Generator | None = None


    def setup(self, stage: str) -> None:
        super().setup(stage)
        precision = str(self.trainer.precision)
        if precision != "32-true":
            raise RuntimeError(f"Run without autocast use precision=32-true, got {precision}")

        datamodule = self.trainer.datamodule
        block_steps = datamodule.hparams.block_steps
        if block_steps != self.trajectory_steps // self.L_min:
            raise RuntimeError(
                f"data.block_steps={block_steps} must equal model trajectory_steps / L_min = "
                f"{self.trajectory_steps // self.L_min}"
            )

    def set_up_module(self, *args, **kwargs):
        net = super().set_up_module(*args, **kwargs)
        attach_multistep_heads(net, self.trajectory_steps)
        return net
    
    def _parallelize_net(self, net, net_info):
        fp32_policy = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32, cast_forward_inputs=False)
        reshard = net_info.fsdp.reshard_after_forward
        for name in MiniMaxH3Transformer3DModel._keep_in_fp32_modules:
            kwargs = {} if reshard is None else {"reshard_after_forward": reshard}
            fully_shard(
                net.get_submodule(name),
                mesh=self.device_mesh["data_parallel"],
                mp_policy=fp32_policy,
                **kwargs,
                )
        super()._parallelize_net(net, net_info)
    
    def _load_pretrained_weights(self) -> None:
        net = self.transformer
        load_checkpoint_to_device(net, Path(self.pretrained_model_name_or_path) / self.transformer_subfolder)
        init_lora_parameters(net, self.net_configs[NET_NAME].lora_config.init_lora_weights)
        init_non_persistent_buffers(net)
        rank_zero_info(f"Loaded {self.transformer_subfolder} onto {self.device} shards")

    def on_fit_start(self) -> None:
        self._load_pretrained_weights()
        self._generator = torch.Generator(self.device)
        self.trajectory = MiniMaxH3Trajectory(
            self.pretrained_model_name_or_path,
            self.transformer_subfolder,
            self.trajectory_steps,
            self.num_frames,
            self.height,
            self.width,
            self.L_min,
            self.device,
            dtype=torch.float32,
        )


    def student(self, inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        video_out, audio_out = self.transformer(**inputs)
        return video_out.transpose(1, 2).float(), audio_out.transpose(1, 2).float()

    def reset_trajectory(self, batch: dict) -> None:
        step = self.trainer.fit_loop.total_batch_idx
        seed = np.random.SeedSequence([self.seed, self.trainer.global_rank, step]).generate_state(1, np.uint64)[0]
        self._generator.manual_seed(int(seed))
        self.trajectory.init_trajectory(batch, self._generator)

    def mid_states(self, video_pred, audio_pred):
        index = self.trajectory.current_index_step
        k = self.sample_target(index, self._generator)
        video, audio = self.trajectory.video, self.trajectory.audio
        video_k, _ = self.pdd_mid_step(video.state, video_pred, video.time_diff, index, k=k)
        audio_k, _ = self.pdd_mid_step(audio.state, audio_pred, audio.time_diff, index, k=k)
        return video_k, audio_k, k

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
    
        if (self.trajectory._registered 
            and self.trajectory.prompt_id != int(batch['index']) 
            and not self.trajectory.done
            ):
            raise ValueError("Prompt changes, before trajectory is finised")  
        if not self.trajectory._registered or self.trajectory.done:
            self.reset_trajectory(batch)

        def teacher_forward(video, audio, video_time, audio_time):
            return self.transformer(**self.trajectory.model_inputs(video, audio, video_time, audio_time))
        
        index = self.trajectory.current_index_step
        video_pred, audio_pred = self.student(self.trajectory.current_inputs())
        video_k, audio_k, k = self.mid_states(video_pred, audio_pred)
        with teacher_mode(self.transformer):
            target_video, target_audio = teacher_mean_velocity(
                teacher_forward,
                video_k,
                audio_k,
                *self.trajectory.times(k),
                self.solver
            )
        video_loss = F.mse_loss(video_pred[:, k], target_video)
        audio_loss =  F.mse_loss(audio_pred[:, k], target_audio)
        loss = self.video_loss_weight * video_loss + self.audio_loss_weight * audio_loss

        self.trajectory.update(video_pred, audio_pred)

        self.log_dict(
            {
                "train/loss": loss.detach(),
                "train/video_loss": video_loss.detach(),
                "train/audio_loss": audio_loss.detach(),
                "train/grid_index": float(index),
            },
            sync_dist=True,
            batch_size=1,
            on_step=True,
            on_epoch=False,
        )
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        clip_val = self._grad_norm_val() if self.grad_clip_warmdown is not None else self.trainer.gradient_clip_val
        self._log_grad_norm(optimizer, clip_val)

    def _log_grad_norm(self, optimizer, clip_val: float | None) -> None:
        grads = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        if not grads:
            return
        total_norm = torch.nn.utils.get_total_norm(grads, norm_type=2.0)
        total_norm = total_norm.full_tensor()
        logs = {"train/grad_norm": total_norm}
        if clip_val:
            logs["train/grad_norm_limit"] = float(clip_val)
        self.log_dict(logs, on_step=True, on_epoch=False, batch_size=1)

    def on_save_checkpoint(self, checkpoint) -> None:
        self.export_pdd_weights(self.global_step)
        return super().on_save_checkpoint(checkpoint)
    
    def _gather_trainable(self, ema: bool) -> dict[str, torch.Tensor]:
        params = [(name, param) for name, param in self.transformer.named_parameters() if param.requires_grad]
        if ema:
            optimizer = self.trainer.optimizers[0]
            sources = [
                optimizer.state.get(param).get(EMA_KEY) for _, param in params
            ]
        else:
            sources = [param for _, param in params]
        tensors = {}
        for (name, _), source in zip(params, sources):
            full = source.full_tensor()
            if self.trainer.is_global_zero:
                tensors[self._clean_key(name)] = full.detach().to("cpu", torch.float32).contiguous()
        return tensors

    def export_pdd_weights(self, step: int) -> None:
        use_ema = any(isinstance(callback, EMA) for callback in self.trainer.callbacks)
        weights = self._gather_trainable(ema=False)
        ema = self._gather_trainable(ema=True) if use_ema else {}
        if not self.trainer.is_global_zero:
            return
        dir = Path(self.trainer.default_root_dir) / "pdd_lora" / f"step-{step:06d}"
        dir.mkdir(parents=True, exist_ok=True)
        save_file(weights, dir / "pdd.safetensors", metadata={"format": "pt"})
        if ema:
            save_file(ema, dir / "pdd_ema.safetensors", metadata={"format": "pt"})
        
        lora = self.net_configs[NET_NAME].lora_config
        targets = lora.target_modules
        config = dict(
            trajectory_steps=self.trajectory_steps,
            L_min=self.L_min,
            L_max=self.L_max,
            video_shift=self.trajectory.shifts[0],
            audio_shift=self.trajectory.shifts[1],
            lora_rank=lora.r,
            lora_alpha=lora.lora_alpha,
            lora_targets=targets if isinstance(targets, str) else sorted(targets),
            transformer_subfolder=self.transformer_subfolder,
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            step=step,
            key_format="peft (lora_A/lora_B.<adapter>) + MultiStepHead proj_out/audio_proj_out (N, out, in)",
        )
        (dir / "pdd_config.json").write_text(json.dumps(config, indent=1))
