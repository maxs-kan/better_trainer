# better_trainer

Built on top of [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) for training in complex setups such as distributed training (FSDP2)
and mixed precision of modular models such as latent diffusion.

It provides nice features to set up model behavior (frozen / active), add LoRA, compile the model, apply activation checkpointing and shard the model across devices through the Hydra config.

I use this framework to train [Parallel Decoding Distillation](https://arxiv.org/abs/2607.26004) of the [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) video model as a case study.

- [Features](#features)
- [Case study: PDD distillation of MiniMax-H3](#case-study-pdd-distillation-of-minimax-h3)


## Case study: PDD distillation of MiniMax-H3

[PDD](https://arxiv.org/abs/2607.26004) (Parallel Decoding Distillation) distills a flow-matching model into a student
that predicts the velocities of several steps in one forward pass. Here it is applied as a LoRA to
MiniMax-H3 (≈33B DiT, text-to-video+audio).

### Setup

- **PDD:** teacher trajectories = 32 steps, min_block_size = 4, max_block_size = 8, i.e. 4-8 NFE at inference. Midpoint solver for teacher target.
- **Student:** frozen bf16 backbone + LoRA + MultiStepHeads: the
  video and audio output projections are replaced by 32 linear layers, initialized from the
  original.
- **Shapes:** 124 frames × 768 × 1344, t2va layout.
- **Optimization:** AdamW, LoRA: lr 1e-4, heads: lr 1e-5, EMA. One trajectory per rank (on 4 GPUs), with `accumulate_grad_batches` = N / min_block_size.

```bash
# 1. cache prompt embeddings
torchrun --nproc_per_node 8 scripts/compute_text_emb_h3.py --model /models/MiniMax-H3 \
    --prompts train.json --output /data/h3_cache/train
# 2. train
DIFFUSERS_ATTN_BACKEND=_native_cudnn python src/train.py experiment=pdd_h3_t2va paths.h3_model_dir=/models/MiniMax-H3 \
    paths.prompt_cache_train=/data/h3_cache/train/index.json
```


### Data construction

PDD has a data-free algorithm, so only prompts are needed for training, not videos. The prompt dataset is constructed from [Vript](https://huggingface.co/datasets/Mutonix/Vript), [VidProM](https://huggingface.co/datasets/WenhaoWang/VidProM) and [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M), to mix human prompts used to generate videos with machine generated captions. The dataset is filtered mostly by prompt length (> 10 words) and corresponding video length (≈ 5 ± 2 s).


### Training optimization

- **Only the DiT is used.** No text encoder (cached embeddings) and no VAE (data-free training).
- **On-device sharded loading.** The model is built on `meta`, sharded with FSDP2, and each rank reads
  only its rows of the weights.
- **Mixed precision.** Frozen backbone in bf16; LoRA and heads in fp32 (master weights);
  bf16 all-gather, fp32 gradient reduction. Layers that base model keeps in fp32 get their own FSDP unit without parameter casting. The PDD trajectory state is kept in fp32.
- **Activation checkpointing** on every `MiniMaxH3TransformerBlock`.
- **One model for student and teacher**, switched by disabling LoRA and the MultiStepHeads.
- **Compiled blocks.** `MiniMaxH3TransformerBlock` and `MiniMaxH3TokenRefinerBlock` are compiled with full graph.
- **Efficient attention.** The diffusers implementation of MiniMax H3 is used, so set `DIFFUSERS_ATTN_BACKEND=_native_cudnn` on H100 GPUs.


### Results
The model was trained on 2500 unique trajectories with a total batch size = 4 in terms of trajectory: gradient accumulation step = number of PDD step trajectrory, with one trajectory per device. The main observations:

- **EMA** is crucial for this setup, most probably because of the batch size, which is particularly small here: the authors of the paper mention that a bigger batch size is one of the most important hyperparameters. As can be seen from the examples below, the non-EMA weights produce weird results, while the EMA weights are much cleaner, although they change the global view of the teacher shot more.


- **Domain adaptation.** The prompt pool contains human written prompts from the early era of image/video
  generation, with tags for style, position and so on, on which the base model works poorly in terms of prompt understanding (see the examples below).
  The distilled model handles such prompts noticeably better, which at first glance looks strange, since training
  contains no ground truth videos for them. I have some hypotesys for such effect: consider a concept K that is rare in the teacher training data: the
  base model knows it but generates it unreliably. If the prompts for distilation training lack K, the student is expected to
  forget it; On the other side, if (in limit) it is only concept K in prompts for student training, studen can become more reliable than the teacher. In
  PDD the teacher targets never samples of its own: it only gives the velocity at states of the student's trajectory, so its
  local knowledge is used while its own global errors may be avoided, and averaging across trajectory may cancel out its
  failures while the consistent structure of K adds up. The same applies to the prompt writing style S:
  the student learns no new world knowledge, it learns to express the teacher knowledge for style S, i.e. a form
  of domain adaptation. This suggests a practical use: fix rules for prompt construction and adapt the model to such
  standardized prompts without target videos.

## Features

### 
The main function of `src/models/module/base_module.BaseModule` is to set up network behavior through the config, i.e. frozen / active params, train / eval mode, param dtypes, optimizer configuration, and optionally torch.compile / activation checkpointing. This follows the paradigm that every nn.Module should be added explicitly from the config. This provides the following main features, which can easily be enabled from the config:
- **[torchtitan](https://github.com/pytorch/torchtitan)-style per-block wrapping.** Activation checkpointing → torch.compile →
  FSDP2 sharding are applied per block class (such as TransformerBlock). The order follows the torchtitan logic for efficient utilization.
- **Light checkpoints.** The module tracks active vs frozen state-dict keys. By default frozen weights are dropped from checkpoints (only LoRA, heads and optimizer state are saved).
- **Explicit dtypes for frozen / active parameters.** Although PyTorch Lightning has built-in mixed precision training, it may be useful to explicitly cast the dtype of frozen / active params through the config, to make sure that the optimizer state is in the expected dtype (mainly fp32).
- **Optimizer parameter groups.** Set up param groups with custom lr, weight_decay, etc., for example weight_decay=0 for biases in linear layers.
- **Gradient clip warm-down.** Decays the gradient clip norm linearly from max to base.
- **FSDPMixin.** Nets are built and set up on the `meta` device, then sharded with [FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) over the data-parallel mesh.
- **EMA callback.** EMA of the trainable parameters is stored in the optimizer state, so it is sharded and checkpointed with the
optimizer.

A net is declared in the model config, for example a LoRA on a frozen bf16 backbone, compiled, sharded and checkpointed per
block:

```yaml
net_configs:
  _target_: src.schemas.fsdp_schemas.FSDPNetConfigs
  transformer:
    init_fn: src.models.net.minimax_h3.build_meta_transformer
    eval_mode: false
    grad: false
    lora_config: {r: 64, lora_alpha: 64, target_modules: [to_q, to_k, to_v, to_out.0]}
    fsdp:
      shard_modules: [diffusers.models.transformers.transformer_minimax_h3.MiniMaxH3TransformerBlock]
      mp_policy: {param_dtype: bfloat16, reduce_dtype: float32}
    activation_checkpointing:
      ac_modules: [diffusers.models.transformers.transformer_minimax_h3.MiniMaxH3TransformerBlock]
    dtype_config: {cast_dtype: true, active_dtype: float32, frozen_dtype: null}
    compile_config:
      compile: true
      compile_modules: [diffusers.models.transformers.transformer_minimax_h3.MiniMaxH3TransformerBlock]
      fullgraph: true
      mode: default
      dynamic: null
```


## Quick start

```bash
conda env create -f environment.yaml
conda activate better_trainer
uv pip install -r requirements.txt
python src/train.py experiment=<experiment>
```
