# Quantization Compatibility by Modality

## Summary

Quantization in `vllm-omni` is split:

- AR and omni-model quantization follow upstream `vllm` plus the unified `build_quant_config()` entrypoint
- diffusion quantization exists for selected DiT models through `fp8`, `int8`, and `gguf`
- multi-stage omni quantization is currently centered on pre-quantized thinker checkpoints, not whole-pipeline quantization

## Compatibility Matrix

| Model Type | AR Weight Quantization | Diffusion Quantization | KV Cache FP8 | Notes |
|------------|------------------------|------------------------|--------------|-------|
| **Omni models** (Qwen-Omni, Qwen3-Omni) | Yes | N/A | Yes | Qwen3-Omni currently centers on pre-quantized thinker checkpoints such as ModelOpt FP8 |
| **AR-only text models** | Yes | N/A | Yes | Full upstream support |
| **Multi-stage AR+DiT** (Qwen-Image, BAGEL) | Partial | Qwen-Image FP8 and Int8; Bagel FP8 framework support | AR stage only | Quantization scope is model-specific; confirm the documented path before assuming all stages are covered |
| **DiT-only image models** (Z-Image, FLUX, FLUX.2-klein, SD3) | No | Z-Image FP8, Int8, GGUF; FLUX FP8; FLUX.2-klein GGUF; SD3 none | N/A | Check method-specific docs before assuming coverage |
| **Video models** (Wan2.2) | No | No | N/A | Use cache and parallelism instead |
| **TTS models** (Qwen3-TTS) | Yes | N/A | Yes | Quality may degrade for voice cloning |
| **Audio models** (Stable-Audio) | No | No | N/A | Diffusion architecture, no quantization path documented here |

## Omni Models (Qwen2.5-Omni, Qwen3-Omni)

The AR backbone (thinker/talker stages) can be quantized. This reduces memory for the language model portion:

```bash
vllm serve Qwen/Qwen2.5-Omni-7B-AWQ --omni --quantization awq
```

**Expected impact**:
- Text understanding/generation quality: minimal degradation (~95%+ of BF16)
- Audio output quality: slight degradation, especially for voice characteristics
- Multi-modal understanding: minimal degradation

For Qwen3-Omni specifically, the currently documented quantized checkpoint path is pre-quantized ModelOpt on the thinker `language_model`, not whole-pipeline online quantization.

## Image Generation Models (FLUX, SD3, Qwen-Image, BAGEL, Z-Image)

Diffusion quantization is now model-specific rather than globally unsupported.

- `Qwen-Image`, `Qwen-Image-2512`: diffusion `fp8` and `int8` supported; start by checking `ignored_layers` guidance
- `Tongyi-MAI/Z-Image-Turbo`: diffusion `fp8`, `int8`, and `gguf` supported
- `FLUX.1-dev`: diffusion `fp8` is documented in the quantization overview
- `FLUX.2-klein`: diffusion `gguf` supported through model-specific adapter logic
- `Bagel`: framework-level FP8 support is documented in the quantization overview; verify the exact stage path before assuming blanket support
- `SD3`, most video diffusion models: no documented diffusion quantization path

For hybrids with an AR stage, AR quantization can still save memory even if the DiT stage is not quantized:

```bash
vllm serve Qwen/Qwen-Image-AWQ --omni --quantization awq
```

For unsupported diffusion models, use:

- **TeaCache** or **Cache-DiT** for speed
- **CPU offloading** for memory
- **BF16/FP16** dtype choices

## Video Models (Wan2.2)

No documented diffusion weight quantization support. Use:
- **TeaCache** or **Cache-DiT** for speed
- **CPU offloading** for memory
- **Tensor parallelism** for both

## TTS Models (Qwen3-TTS)

The AR decoder can be quantized. Trade-offs:
- Intelligibility: maintained
- Voice cloning fidelity (CustomVoice): slight degradation at 4-bit
- Recommendation: use AWQ 8-bit instead of 4-bit for TTS to preserve voice quality

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-AWQ --omni --quantization awq
```

## Choosing Based on Modality

| Primary Use Case | Recommended Approach |
|-----------------|---------------------|
| Text or omni AR backbone | AWQ 4-bit or GPTQ 4-bit |
| Qwen-Image diffusion | diffusion `fp8` or `int8`, often with tuned `ignored_layers` |
| Z-Image diffusion | diffusion `fp8`, `int8`, or `gguf` depending on the goal |
| FLUX diffusion | documented `fp8` support for FLUX.1-dev, `gguf` for FLUX.2-klein |
| Qwen3-Omni thinker | pre-quantized ModelOpt FP8 checkpoint |
| Unsupported diffusion models | CPU offloading + cache acceleration |
| TTS (Qwen3-TTS) | AWQ 8-bit for quality |
| Audio understanding (MiMo-Audio) | AWQ 4-bit |
