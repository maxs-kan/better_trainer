from pathlib import Path
import os
import math

import torch
from einops import rearrange
from diffusers.utils.pil_utils import pt_to_pil
from torchvision.transforms.functional import to_tensor
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions, VAE_IMAGE_SIZE
from PIL import Image
from safetensors.torch import save_file
from peft import get_peft_model_state_dict
import torch.nn.functional as F
import torchvision
import pandas as pd

from ..utils.utils import make_grid
from .module.base_module import BaseModule, LoRAEma
from ..losses.wavlet import WaveletsLoss


def scale_for_inference(img):
    image_width, image_height = img.size
    vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
    img = img.resize((vae_width, vae_height), Image.LANCZOS)
    return img


def unpack_latents(latents, height, width):
    batch_size, num_patches, channels = latents.shape

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // 4, height, width)
    return latents


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents

# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class QwenImageEditModule(BaseModule):
    def __init__(
        self,
        pipe,
        style_backbone,
        style_emb,
        optimizer: torch.optim.Optimizer,
        criterion,
        lora,
        compile_config,
        lr_scheduler = None,
        logit_normal_mean = 0,
        logit_normal_std = 1,
        num_diffusin_val_steps = 25,
        seed = 42, 
        use_gradient_checkpointing = False,
        ema_decay: float=0.9999, 
        ema_start_step: int=500, 
        use_ema: bool=True,
        flip_timesteps_peak=False,
        shift_timesteps=False,
        region_txt_attn=False,

    ) -> None:
        super().__init__(optimizer, lora, compile_config, lr_scheduler)

        self.save_hyperparameters(logger=False, ignore=['pipe', 'criterion', 'style_backbone', 'style_emb'])

        self.add_net(pipe.transformer, 'flow_net', grad = not lora.lora, lora = lora.lora)
        # self.add_net(style_backbone, 'style_backbone', grad=False, lora=False, eval=True)
        # self.add_net(style_emb, 'style_emb', grad=True, lora=False)
        
        self.pipe = pipe
        self.criterion = criterion
        self.pipe.text_encoder = None
        self.ema = None


    def configure_model(self):
        if self.hparams.use_gradient_checkpointing:
            self.flow_net.enable_gradient_checkpointing()
        
        assert self.flow_net.config.zero_cond_t
        if isinstance(self.criterion, WaveletsLoss):
            self.criterion.to(dtype=torch.float32)
        
        self.pipe.to(self.device)
        self.pipe.vae.eval()
        self.pipe.vae.requires_grad_(False)
        
        if self.trainer.global_rank == 0:
            os.makedirs(Path(self.trainer.default_root_dir) / 'lora', exist_ok=True)
        

    def on_fit_start(self):
        super().on_fit_start()
        if torch.distributed.is_initialized():
            for name in self.nets_names:
                net = getattr(self, name)
                for p in net.parameters():
                    if p.requires_grad:
                        torch.distributed.broadcast(p.data, src=0)
        if self.hparams.use_ema:
            self.ema = LoRAEma(self.flow_net, decay=self.hparams.ema_decay, start_step=self.hparams.ema_start_step)
            if hasattr(self, '_ema_state'):
                self.ema.load_state_dict(self._ema_state, self.device)
                del self._ema_state

    
    def shift_timesteps(self, timesteps, image_seq_len):
        mu = calculate_shift(
            image_seq_len,
            self.pipe.scheduler.config.get("base_image_seq_len", 256),
            self.pipe.scheduler.config.get("max_image_seq_len", 4096),
            self.pipe.scheduler.config.get("base_shift", 0.5),
            self.pipe.scheduler.config.get("max_shift", 1.15),
        )
        if self.hparams.flip_timesteps_peak:
            mu *= -1
        timesteps = math.exp(mu) / (math.exp(mu) + (1 / timesteps - 1) )
        return timesteps
    
    def sample_timesteps(self, b: int, image_seq_len) -> torch.Tensor:
        u = torch.normal(
                         mean=self.hparams.logit_normal_mean,
                         std=self.hparams.logit_normal_std,
                         size=(b,), device=self.device)

        timesteps = torch.sigmoid(u)
        if self.hparams.shift_timesteps:
            timesteps = self.shift_timesteps(timesteps, image_seq_len)
        
        return timesteps


    @torch.no_grad()
    def vae_encode(
        self,
        target_image,
        control_images,
        b,
        n,
    ):

        control_images = rearrange(control_images, 'b n c h w -> (b n) c h w', n=n)
        input_images = torch.cat([target_image, control_images], dim=0).unsqueeze(2)

        image_latents = self.pipe._encode_vae_image(image=input_images, generator=None)
        batch_cated, num_channels_latents, _, image_latent_height, image_latent_width = image_latents.shape
        image_latents = pack_latents(
                    image_latents, batch_cated, num_channels_latents, image_latent_height, image_latent_width
                )
        target_image, control_images = image_latents[:b], image_latents[b:]
        control_images = rearrange(control_images, '(b n) l d -> b (n l) d', n=n)
        return target_image, control_images, image_latent_height, image_latent_width


    def get_style_emb(
        self,
        style_image
    ):
        with torch.no_grad():
            x = self.style_backbone(style_image).last_hidden_state
        emb = self.style_emb(x)
        return emb


    def insert_style_emb(
        self,
        style_emb,
        prompt_emb,
        prompt_emb_mask,
        b,
    ):
        style_emb = style_emb.unsqueeze(1)
        prompt_emb = torch.cat([style_emb, prompt_emb], dim=1)
        style_mask = torch.ones((b, 1), dtype=prompt_emb_mask.dtype, device=self.device)
        prompt_emb_mask = torch.cat([style_mask, prompt_emb_mask], dim=1)
        return prompt_emb, prompt_emb_mask
        
    
        
    def forward(self, batch: dict, keep_batch=False) -> torch.Tensor:
        #TODO: add uncoditional training, timestep density
        z_0 = batch['target_image']
        control_imgs = batch['condition_images']
        prompt_emb, prompt_emb_mask = batch['prompt_embeds'], batch['prompt_embeds_mask']
        # style_img = batch['style_image']
        mask = batch['mask']

        B, _, H, W = z_0.shape
        n_control = control_imgs.shape[1]

        z_0, control_imgs, h_latent, w_latent = self.vae_encode(z_0, control_imgs, B, n_control)

        t     = self.sample_timesteps(B, z_0.shape[1])
        noise = torch.randn_like(z_0)
 
        z_t  = t.view(B, 1, 1) * noise + (1.0 - t.view(B, 1, 1)) * z_0
        v_target = noise - z_0
 
        z_input = torch.cat([z_t, control_imgs], dim=1)
        img_shapes = [
            [
                (1, h_latent // 2, w_latent // 2),
                *[
                    (1, h_latent // 2, w_latent // 2)
                    for _ in range(n_control)
                ],
            ]
        ] * B

        # style_emb = self.get_style_emb(style_img)
        # prompt_emb, prompt_emb_mask = self.insect_style_emb(style_emb, prompt_emb, prompt_emb_mask, B)

        attention_kwargs = {}
        if self.hparams.region_txt_attn:
            prompt_emb_mask = None
            attention_kwargs["attention_mask"] = batch['attn_mask']
 
        v_pred = self.flow_net(
                        hidden_states=z_input,
                        timestep=t,
                        encoder_hidden_states_mask=prompt_emb_mask,
                        encoder_hidden_states=prompt_emb,
                        img_shapes=img_shapes,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
        v_pred = v_pred[:, :z_t.shape[1]]

        with torch.autocast(device_type='cuda', enabled=False):
            if isinstance(self.criterion, WaveletsLoss):    
                z_0 = z_0.float()
                noise = noise.float()

                # reverse the noise to get the model prediction of the pure latents
                z_pred = noise - v_pred.float()

                z_pred = unpack_latents(z_pred, h_latent, w_latent)
                z_0 = unpack_latents(z_0, h_latent, w_latent)

                mask = F.interpolate(mask.float(), size=(h_latent // 2, w_latent // 2), mode='nearest')
                loss = self.criterion(z_pred, z_0)
                
            else:
                v_pred = v_pred.float()
                v_target = v_target.float()

                v_pred = unpack_latents(v_pred, h_latent, w_latent)
                v_target = unpack_latents(v_target, h_latent, w_latent)
                mask = F.interpolate(mask.float(), size=(h_latent, w_latent), mode='nearest')

                loss = self.criterion(v_pred, v_target)
            
            if keep_batch:
                n = torch.sum(mask, (1, 2, 3))
                loss = torch.sum(loss * mask, (1, 2, 3)) / n
            else:
                n = torch.sum(mask) if torch.sum(mask) > 0 else 1
                loss = torch.sum(loss * mask) / n
            
        return loss
 

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=False, batch_size=batch['target_image'].shape[0], sync_dist=True)
        return loss


    def on_after_backward(self):
        total_norm = 0.0
        n = 0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item()
                n += 1
        total_norm = total_norm / n
        self.log('train/grad_norm', total_norm, on_step=True, on_epoch=False)


    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema is not None:
            self.ema.update(self.flow_net, self.global_step)
 

    @torch.inference_mode()
    def run_pipe(self, batch):
        b = batch['condition_images'].shape[0]
        result = []
        generator = torch.manual_seed(self.hparams.seed)
        for i in range(b):
            imgs = []
            target = batch['target_image'][i: i + 1]
            control_imgs = batch['condition_images'][i: i + 1]
            prompt_emb = batch['prompt_embeds'][i: i + 1]
            prompt_emb_mask = batch['prompt_embeds_mask'][i: i + 1]
            # style_img = batch['style_image'][i : i + 1]

            # style_emb = self.get_style_emb(style_img)
            # prompt_emb, prompt_emb_mask = self.insect_style_emb(style_emb, prompt_emb, prompt_emb_mask, b=1)

            target = pt_to_pil(target)[0]
            control_imgs = control_imgs.unbind(dim=1)
            control_imgs = [pt_to_pil(x)[0] for x in control_imgs]
            # style_img = pt_to_pil(style_img)[0]

            # target = scale_for_inference(target)
            # control_imgs = [scale_for_inference(img) for img in control_imgs]
            # style_img = scale_for_inference(style_img)

            imgs.append(target)
            # imgs.append(style_img)
            imgs.extend(control_imgs)

            attention_kwargs = {}
            if self.hparams.region_txt_attn:
                attention_kwargs["attention_mask"] = batch['attn_mask'][i: i + 1]

            inputs = {
                "image": control_imgs, 
                "prompt_embeds": prompt_emb,
                "prompt_embeds_mask": prompt_emb_mask,
                "generator": generator,
                "true_cfg_scale": 1.0,
                "num_inference_steps": self.hparams.num_diffusin_val_steps,
                "num_images_per_prompt": 1,
                "height": target.size[1],
                "width": target.size[0],
                "vae_area": target.size[1] * target.size[0],
                "attention_kwargs": attention_kwargs
            }
            
            output = self.pipe(**inputs)
            output_image = output.images[0]
            imgs.append(output_image)
            result.append(imgs)

        result = make_grid(result)
        return result


    def validation_step(self, batch, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        grid = self.run_pipe(batch)
        grid_tensor = to_tensor(grid).to(self.device)

        # Gather grids from all ranks to rank 0
        gathered = self.all_gather(grid_tensor)
        if self.trainer.global_rank == 0:
            combined = torchvision.utils.make_grid(gathered, nrow=gathered.shape[0])
            self.logger.experiment.add_image("val/samples", combined, self.global_step)

    
    def on_save_checkpoint(self, checkpoint):
        if self.trainer.global_rank == 0:
            lora_path = Path(self.trainer.default_root_dir) / "lora"
            lora_state = get_peft_model_state_dict(self.pipe.transformer)
            lora_state = {f"transformer.{k}": v for k, v in lora_state.items()}
            save_file(lora_state, lora_path / f"lora_step_{self.global_step}.safetensors")

            if self.ema is not None and self.ema._initialized:
                ema_state = self.ema.shadow
                ema_state = {f"transformer.{k.replace('default.', '').replace('_orig_mod.', '')}": v for k, v in ema_state.items()}
                save_file(ema_state, lora_path / f'lora_ema_step_{self.global_step}.safetensors')
        if self.ema is not None and self.ema._initialized:
            checkpoint['ema'] = self.ema.state_dict()


        # style_dict = self.style_emb.state_dict()
        # style_dict = {k.removeprefix("style_emb."): v for k, v in style_dict.items()}
        # style_path = Path(self.trainer.default_root_dir) / "style_emb" / f"style_step_{self.global_step}.safetensors"
        # save_file(style_dict, style_path)
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        if 'ema' in checkpoint:
            self._ema_state = checkpoint['ema']
        if hasattr(super(), 'on_load_checkpoint'):
            super().on_load_checkpoint(checkpoint)


    def on_test_epoch_start(self):
        self.test_loss = []
        self.test_id = []
    

    def test_step(self, batch, batch_idx):
        loss  = self(batch, keep_batch=True)

        self.test_loss.append(loss)
        self.test_id.append(batch['sample_ids'])
        
        return None

    def on_test_epoch_end(self):
        loss = torch.cat(self.test_loss)
        ids = torch.cat(self.test_id)

        all_loss = self.all_gather(loss)
        all_ids = self.all_gather(ids)

        if self.trainer.is_global_zero:
            df = pd.DataFrame({'idx': all_ids.flatten().cpu().numpy(), 'pred_group': all_loss.flatten().cpu().numpy()})
            df.to_csv(f'{self.trainer.default_root_dir}/loss.csv', index=False)

        self.test_loss.clear()
        self.test_id.clear()