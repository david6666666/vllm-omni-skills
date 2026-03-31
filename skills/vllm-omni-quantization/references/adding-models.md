# Adding Quantization Support to a New Model

Use this file when extending `vllm-omni` quantization to a new model architecture.

The current implementation pattern is built around the unified quantization framework:

- `vllm_omni.quantization.build_quant_config()`
- `ComponentQuantizationConfig`
- model-specific wiring in pipelines and transformers

## Decide the Integration Type First

Classify the work before changing code:

| Case | Typical Path |
|------|--------------|
| New AR or generic method | upstream `vllm` first |
| New diffusion method on an existing DiT family | `vllm-omni` wrapper + model wiring |
| New pre-quantized omni checkpoint | unified config + model-config normalization + component scoping |

Rule: if kernels, generic loader semantics, or config classes are missing, upstream `vllm` owns that work. `vllm-omni` should not create a private quantization stack.

## Diffusion Model Checklist

### 1. Thread `quant_config` into the transformer

- Accept `quant_config: QuantizationConfig | None` in the transformer constructor
- Pass it into all vLLM linear layers such as `QKVParallelLinear`, `ColumnParallelLinear`, and `RowParallelLinear`
- Always pass stable `prefix` values so `ignored_layers` matching works

Common failure: only part of the architecture receives `quant_config`, so some layers quantize and others silently stay BF16.

### 2. Build quant config in the pipeline

Use:

```python
from vllm_omni.quantization import build_quant_config

quant_config = build_quant_config(od_config.quantization_config)
```

This replaces older diffusion-specific factories. Prefer the unified entrypoint even for diffusion models.

### 3. Handle non-transformer components correctly

For text encoders and VAEs loaded via `from_pretrained()`:

- use `apply_fp8_weight_storage()` when the quantization path relies on hook-based FP8 storage
- explicitly mark pre-loaded component weights as loaded in `load_weights()`

Without this, strict loading checks will report missing `vae.*` or `text_encoder.*` parameters.

### 4. Identify sensitive layers

Use `ignored_layers` when output quality degrades.

Common sensitive layers:

| Layer | Typical Risk |
|-------|--------------|
| `img_mlp` | denoising latents have shifting dynamic range |
| `feed_forward` | large FFN dynamic range in DiT blocks |
| `proj_out` | final projection amplifies small errors |
| `lm_head` | precision-critical token selection |
| `mlp.gate` | MoE routing is precision-critical |

### 5. Update docs and support tables

At minimum:

- method page under `docs/user_guide/diffusion/quantization/`
- supported-model tables
- usage examples

## Multi-stage Omni Checklist

For models such as Qwen3-Omni:

1. Normalize quantization config from the model config, especially for pre-quantized checkpoints
2. Keep quantization scoped to the intended component, such as thinker `language_model`
3. Leave unsupported components in BF16 unless explicitly validated
4. Verify that the user-facing API remains unified even if the runtime scope is partial

Common failure: the checkpoint is pre-quantized for one component, but the runtime path tries to quantize or validate unrelated components.

## Testing Checklist

- Run a smoke test with quantization enabled
- Compare against a BF16 baseline with fixed seed and identical prompt or request settings
- For diffusion models, use LPIPS or an equivalent quality gate
- Verify memory reduction and throughput, not just one of them
- Confirm unsupported models fail clearly
- Confirm `ignored_layers` or component prefixes behave as expected

## Practical Signals from Current Implementations

- `Qwen-Image`: FP8 and Int8 are the main diffusion reference implementations
- `Z-Image`: supports all-layer FP8 and now also Int8 and GGUF paths
- `Qwen3-Omni`: the documented quantized path is currently pre-quantized thinker checkpoints through the unified framework

Use the current implementation that is closest to your target architecture as the reference, not the oldest quantization code in the repo.
