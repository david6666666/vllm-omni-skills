---
name: optimize-diffusion-perf
description: Guide for achieving optimal inference performance with vLLM-Omni diffusion models. Covers all lossless and lossy optimization methods (parallelism, torch.compile, CPU offload, quantization, cache acceleration), per-model support tables, and ready-to-use recipes. Use when asked to speed up diffusion inference, reduce latency, lower VRAM usage, or tune a diffusion pipeline.
---

# vLLM-Omni Diffusion: Optimal Performance Guide

Use this guide when a user asks how to speed up diffusion inference, reduce latency, lower VRAM, or tune a diffusion pipeline in vLLM-Omni.

## Step 0: Understand the Baseline

Before optimizing, establish a baseline:

1. **Identify the model** and its pipeline class (check `model_index.json` → `_class_name`)
2. **Run a baseline** with `--enforce-eager` (disables torch.compile) and no parallelism
3. **Record**: total time, denoising it/s, VRAM usage, output quality

```bash
# Baseline example (text-to-image)
python examples/offline_inference/text_to_image/text_to_image.py \
  --model <MODEL> --enforce-eager --prompt "..." --output baseline.png

# Baseline example (text-to-video)
python examples/offline_inference/text_to_video/text_to_video.py \
  --model <MODEL> --enforce-eager --prompt "..." --output baseline.mp4
```

## Step 1: Apply Lossless Optimizations

These do **not** affect output quality. Apply in order of impact.

### 1.1 torch.compile (Regional Compilation)

**What**: Compiles repeated DiT transformer blocks via `torch.compile(dynamic=True)`. Fuses ops, reduces kernel launch overhead.

**How**: Enabled by **default**. Use `--enforce-eager` to disable.

**Speedup**: ~1.2–1.5× on denoising loop.

**Requirements**: Model transformer must define `_repeated_blocks` attribute. First request is slower (compilation warmup).

**Config**: `OmniDiffusionConfig.enforce_eager` (default `False` = compile enabled).

**Source**: `vllm_omni/diffusion/compile.py`, `vllm_omni/diffusion/worker/diffusion_model_runner.py`

### 1.2 Multi-GPU Parallelism

All configured via `DiffusionParallelConfig`. Check the support table below before enabling.

#### Sequence Parallelism (Ulysses-SP)

**What**: Splits sequence tokens across GPUs using all-to-all communication (DeepSpeed Ulysses).

**How**: `--ulysses-degree N` (offline) or `--usp N` (online serving)

**Speedup**: Near-linear scaling. Best for long-sequence models (video, high-res image).

```python
from vllm_omni.diffusion.data import DiffusionParallelConfig
parallel_config = DiffusionParallelConfig(ulysses_degree=2)
omni = Omni(model="...", parallel_config=parallel_config)
```

#### Ring Attention

**What**: Ring-based P2P communication for attention across GPUs.

**How**: `--ring-degree N` (offline) or `--ring N` (online serving)

**Note**: Can combine with Ulysses: `ulysses_degree × ring_degree = total SP GPUs`.

#### CFG Parallel

**What**: Runs positive/negative CFG branches on separate GPUs. Only rank 0 computes scheduler step.

**How**: `--cfg-parallel-size 2`

**Speedup**: ~2× on models using classifier-free guidance.

**Constraint**: Requires exactly 2 GPUs. Only for models that use CFG.

```bash
# 4-GPU: CFG parallel (2) × Ulysses (2)
python text_to_image.py --model Qwen/Qwen-Image \
  --cfg-parallel-size 2 --ulysses-degree 2
```

#### Tensor Parallelism (TP)

**What**: Shards DiT linear layers across GPUs using `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`.

**How**: `--tensor-parallel-size N`

**Note**: Only DiT blocks are sharded — text encoder is replicated on all ranks (extra VRAM per GPU). See [Issue #771](https://github.com/vllm-project/vllm-omni/issues/771).

#### VAE Patch Parallelism

**What**: Shards VAE decode spatially across ranks using tiling.

**How**: `--vae-patch-parallel-size N`

**Constraint**: Auto-enables `--vae-use-tiling`.

#### HSDP (Hybrid Sharded Data Parallel)

**What**: Shards model weights across GPUs using PyTorch FSDP2. Reduces per-GPU VRAM.

**How**: Via `DiffusionParallelConfig(use_hsdp=True)`. Requires multi-GPU.

#### Expert Parallel

**What**: Shards MoE experts across devices with all-to-all token routing.

**How**: `--enable-expert-parallel`

**Constraint**: Only for MoE models (e.g., HunyuanImage3.0).

### 1.3 CPU Offload

Two mutually exclusive strategies. Both single-GPU only.

#### Model-level (Sequential) Offload

**What**: Swaps DiT ↔ encoders on GPU. Only one group is on GPU at a time.

**How**: `--enable-cpu-offload` or `Omni(enable_cpu_offload=True)`

**Tradeoff**: Adds H2D transfer latency between encoder and denoising phases.

#### Layerwise (Blockwise) Offload

**What**: Keeps only 1 transformer block on GPU at a time. Async prefetch via separate CUDA stream.

**How**: `--enable-layerwise-offload` or `Omni(enable_layerwise_offload=True)`

**Best for**: Large video models (Wan A14B) where per-block compute >> H2D transfer → nearly zero-cost offload.

**Requirement**: Model DiT must define `_layerwise_offload_blocks_attr`.

**VRAM savings**: Dramatic (e.g., 40+ GB → ~11 GB for Wan A14B).

### 1.4 VAE Memory Optimizations

- `--vae-use-slicing`: Process VAE in slices (saves VRAM).
- `--vae-use-tiling`: Process VAE in tiles (saves VRAM, enables patch parallel).

Both are boolean flags. Use when OOM during VAE decode.

### 1.5 Quantization

#### FP8 (W8A8)

**What**: Online quantization of DiT linear layers to FP8.

**How**: `--quantization fp8`

**Requirements**: Ada/Hopper GPU (SM89+). Native hardware FP8.

**VRAM**: ~50% reduction on DiT weights. **Speedup**: 1.3–1.5×.

```bash
python text_to_image.py --model Qwen/Qwen-Image --quantization fp8
```

**Layer skipping**: `--ignored-layers 'add_kv_proj,to_add_out'` to exclude specific layers from quantization.

#### GGUF (Pre-quantized)

**What**: Loads pre-quantized GGUF weights for transformer.

**How**: `--quantization gguf --gguf-model <path-or-hf-id>`

**Source**: `docs/user_guide/diffusion/quantization/gguf.md`

## Step 2: Apply Lossy Optimizations (Optional)

These trade quality for speed. Always compare output quality against baseline.

### 2.1 TeaCache

**What**: Caches transformer computations when consecutive timesteps are similar. Skips redundant forward passes.

**Speedup**: 1.5–2.0× depending on `rel_l1_thresh`.

**How**:
```python
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2},
)
```

**CLI**: `--cache-backend tea_cache`

**Online**: `vllm serve <MODEL> --omni --cache-backend tea_cache --cache-config '{"rel_l1_thresh": 0.2}'`

**Quality tuning**:
- `0.1–0.2`: minimal quality loss (~1.5× speedup)
- `0.4`: slight quality loss (~1.8× speedup)
- `0.6–0.8`: noticeable quality loss (~2.0–2.25× speedup)

**Supported models**: Qwen-Image family, BAGEL. See `docs/user_guide/diffusion/teacache.md`.

### 2.2 Cache-DiT (DBCache + TaylorSeer + SCM)

**What**: Hybrid caching with three sub-methods:
- **DBCache**: Caches intermediate block outputs when residuals are small
- **TaylorSeer**: Taylor expansion to forecast future hidden states
- **SCM**: Step Computation Masking — selectively skip entire denoising steps

**Speedup**: 1.5–2.5× depending on configuration.

**How**:
```python
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",
    cache_config={
        # DBCache
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
        "residual_diff_threshold": 0.24,
        "max_continuous_cached_steps": 3,
        # TaylorSeer (optional)
        "enable_taylorseer": False,
        "taylorseer_order": 1,
        # SCM (optional)
        "scm_steps_mask_policy": None,  # "slow"/"medium"/"fast"/"ultra"
        "scm_steps_policy": "dynamic",
    },
)
```

**CLI**: `--cache-backend cache_dit`

**Excluded models**: `NextStep11Pipeline`, `StableDiffusionPipeline` (see `_NO_CACHE_ACCELERATION` in `registry.py`).

**Source**: `docs/user_guide/diffusion/cache_dit_acceleration.md`

### 2.3 Fewer Inference Steps

Reducing `--num-inference-steps` gives linear speedup but affects quality. Typical ranges:
- Image models: 20–50 steps
- Video models: 20–40 steps
- Distilled models: 4–8 steps

## Model Support Tables

### ImageGen Parallelism

| Model | Ulysses-SP | Ring-SP | CFG | TP | VAE-Patch | EP | HSDP |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen-Image | ✅ | ✅ | ✅ | ✅ | ✅ | N/A | ❌ |
| Z-Image | ✅ | ✅ | ❌ | ✅(TP=2) | ✅ | N/A | ❌ |
| FLUX.1-dev | ❌ | ❌ | ✅ | ✅ | ❌ | N/A | ✅ |
| FLUX.2-dev | ❌ | ❌ | ❌ | ✅ | ❌ | N/A | ✅ |
| FLUX.2-klein | ✅ | ✅ | ❌ | ✅ | ❌ | N/A | ✅ |
| SD 3.5 | ❌ | ❌ | ❌ | ✅ | ✅ | N/A | ❌ |
| HunyuanImage3.0 | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
| LongCat-Image | ✅ | ✅ | ❌ | ✅ | ❌ | N/A | ❌ |

### VideoGen Parallelism

| Model | Ulysses-SP | Ring | TP | HSDP | VAE-Patch |
|-------|:---:|:---:|:---:|:---:|:---:|
| Wan2.2 T2V | ✅ | ✅ | ✅ | ✅ | ✅ |
| LTX-2 | ✅ | ✅ | ✅ | ❌ | ❌ |

### CPU Offload Support

| Model | Model-level | Layerwise |
|-------|:-----------:|:---------:|
| Wan2.2 T2V/I2V | ✅ | ✅ |
| Qwen-Image | ✅ | ✅ |
| Others | ✅ (expected) | Needs `_layerwise_offload_blocks_attr` |

### Cache Acceleration

**TeaCache**: Qwen-Image, Qwen-Image-Edit, Qwen-Image-Edit-2509, Qwen-Image-Layered, BAGEL

**Cache-DiT**: All DiT-based models except `NextStep11Pipeline`, `StableDiffusionPipeline`

## Quick Recipes

### Recipe A: Maximum speed, single GPU, lossless (Image model)

```bash
python text_to_image.py \
  --model Qwen/Qwen-Image \
  --prompt "..." \
  --quantization fp8
# torch.compile is on by default
```

### Recipe B: Maximum speed, multi-GPU, lossless (Image model, 4 GPUs)

```bash
python text_to_image.py \
  --model Qwen/Qwen-Image \
  --prompt "..." \
  --cfg-parallel-size 2 --ulysses-degree 2 \
  --quantization fp8
```

### Recipe C: Low VRAM, single GPU (Video model)

```bash
python text_to_video.py \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "..." \
  --enable-layerwise-offload \
  --vae-use-slicing --vae-use-tiling
```

### Recipe D: Maximum speed, multi-GPU, lossless (Video model, 8 GPUs)

```bash
python text_to_video.py \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "..." \
  --ulysses-degree 4 --ring-degree 2 \
  --vae-patch-parallel-size 8 \
  --quantization fp8
```

### Recipe E: Lossy speedup with cache acceleration (Image model)

```bash
python text_to_image.py \
  --model Qwen/Qwen-Image \
  --prompt "..." \
  --cache-backend cache_dit
```

### Recipe F: LTX-2 video, 2-GPU SP, lossless

```bash
python text_to_video.py \
  --model Lightricks/LTX-2 \
  --prompt "..." \
  --ulysses-degree 2 \
  --height 768 --width 1280 --num-frames 97
```

## Decision Flowchart

```
Is output quality paramount?
├── YES → Use only Step 1 (lossless)
│   ├── Single GPU? → torch.compile (default) + FP8 quantization
│   ├── Multi-GPU? → Add SP/TP/CFG parallel (check support table)
│   └── OOM? → Enable CPU offload or VAE slicing/tiling
└── NO → Also apply Step 2 (lossy)
    ├── TeaCache supported? → Use tea_cache with rel_l1_thresh=0.2
    └── DiT model? → Use cache_dit with defaults
```

## Tips

- **Always benchmark with torch.compile enabled** (remove `--enforce-eager`). First request is slower but subsequent ones are faster.
- **CFG parallel + Ulysses is usually better** than pure Ulysses at the same GPU count for CFG models.
- **Layerwise offload is nearly free for video models** where per-block compute dwarfs H2D transfer time.
- **Combine lossless + lossy**: e.g., torch.compile + FP8 + TeaCache for maximum throughput.
- **Check `_NO_CACHE_ACCELERATION`** in `registry.py` before enabling cache backends — UNet-based and some specialized models don't support them.

## Key Source Files

| File | What |
|------|------|
| `vllm_omni/diffusion/data.py` | `OmniDiffusionConfig`, `DiffusionParallelConfig`, `DiffusionCacheConfig` |
| `vllm_omni/diffusion/compile.py` | Regional torch.compile logic |
| `vllm_omni/diffusion/registry.py` | `_NO_CACHE_ACCELERATION`, model registry |
| `vllm_omni/diffusion/distributed/cfg_parallel.py` | CFGParallelMixin |
| `vllm_omni/diffusion/cache/` | TeaCache and CacheDiT backends |
| `vllm_omni/diffusion/offloader/` | CPU offload backends |
| `docs/user_guide/diffusion/` | All user-facing docs |
