# Quantization Compatibility by Modality

## Summary

Quantization in vLLM-Omni applies to **autoregressive (AR) language model components** only. Diffusion transformer (DiT) components used in image/video generation are not quantized and remain in BF16/FP16.

## Compatibility Matrix

| Model Type | Weight Quantization | KV Cache FP8 | Notes |
|------------|--------------------|----|-------|
| **Omni models** (Qwen-Omni) | Yes (AR backbone) | Yes | Quantize the thinker/talker; DiT decoder unaffected |
| **AR-only text models** | Yes | Yes | Full support |
| **Multi-stage AR+DiT** (Qwen-Image, BAGEL) | Partial (AR stage only) | AR stage only | DiT stage stays in BF16 |
| **DiT-only image models** (FLUX, SD3, Z-Image) | No | N/A | No quantization support |
| **Video models** (Wan2.2) | No | N/A | DiT architecture, no quantization |
| **TTS models** (Qwen3-TTS) | Yes (AR decoder) | Yes | Quality may degrade for voice cloning |
| **Audio models** (Stable-Audio) | No | N/A | Diffusion architecture |

## Omni Models (Qwen2.5-Omni, Qwen3-Omni)

The AR backbone (thinker/talker stages) can be quantized. This reduces memory for the language model portion:

```bash
vllm serve Qwen/Qwen2.5-Omni-7B-AWQ --omni --quantization awq
```

**Expected impact**:
- Text understanding/generation quality: minimal degradation (~95%+ of BF16)
- Audio output quality: slight degradation, especially for voice characteristics
- Multi-modal understanding: minimal degradation

## Image Generation Models (FLUX, SD3, Qwen-Image, BAGEL)

Diffusion components do not support weight quantization. Use performance techniques instead:
- **TeaCache**: 1.5-2.0x speedup with no quality loss
- **CPU offloading**: fit larger models with `--cpu-offload-gb`
- **Dtype**: ensure BF16 is used (default on Ampere+)

For Qwen-Image and BAGEL (AR+DiT pipelines), quantizing the AR stage still saves memory:

```bash
vllm serve Qwen/Qwen-Image-AWQ --omni --quantization awq
```

## Video Models (Wan2.2)

No weight quantization support. Use:
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
| Text/omni (Qwen-Omni) | AWQ 4-bit, saves ~75% VRAM |
| Image generation (FLUX, SD3) | CPU offloading + TeaCache |
| Image generation (Qwen-Image) | AWQ on AR stage |
| Video generation (Wan2.2) | CPU offloading + Cache-DiT |
| TTS (Qwen3-TTS) | AWQ 8-bit for quality |
| Audio understanding (MiMo-Audio) | AWQ 4-bit |
