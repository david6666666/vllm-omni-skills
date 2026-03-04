# vLLM-Omni Skills Update Log Index

## Quick Navigation

### By Skill
- [Image Generation](image-gen.md) - Image generation model updates
- [Video Generation](video-gen.md) - Video generation model updates
- [Audio & TTS](audio-tts.md) - Audio and TTS updates
- [API](api.md) - API endpoint and interface updates
- [Quantization](quantization.md) - Quantization method updates
- [Performance](performance.md) - Performance optimization updates
- [Distributed](distributed.md) - Distributed inference updates
- [CI/CD](cicd.md) - CI/CD pipeline updates

### By Time
- [2026-03 (Week 1)](./#2026-03) - Latest updates
- [2026-02](archive/2026-02.md) - Archive
- [2026-01](archive/2026-01.md) - Archive

---

## Recent Updates (Last 7 Days)

### 2026-03-04
- **[image-gen]** HunyuanImage3 image editing support ([#1644](https://github.com/vllm-project/vllm-omni/pull/1644))

### 2026-03-03
- **[image-gen]** Fix filepath resolution for GLM-Image ([#1609](https://github.com/vllm-project/vllm-omni/pull/1609))

### 2026-03-02
- **[image-gen]** Fix load_weights error for HunyuanImage3.0 ([#1598](https://github.com/vllm-project/vllm-omni/pull/1598))
- **[video-gen]** Speed up diffusion model startup by multi-thread weight loading ([#1504](https://github.com/vllm-project/vllm-omni/pull/1504))
- **[audio-tts]** Qwen3TTS streaming output ([#1438](https://github.com/vllm-project/vllm-omni/pull/1438))

---

## Update Statistics

| Skill | This Week | This Month | Total |
|-------|-----------|------------|-------|
| image-gen | 3 | 3 | 3 |
| api | 2 | 2 | 2 |
| audio-tts | 2 | 2 | 2 |
| video-gen | 1 | 1 | 1 |
| quantization | 1 | 1 | 1 |
| performance | 5 | 5 | 5 |

---

## Archive Strategy

**Archive every 4 weeks** (aligned with vllm-omni release cycle)

- Current cycle: 2026-03-04 ~ 2026-04-01
- Next archive: 2026-04-01

---

## How to Use

### View updates for a specific skill
```bash
# View all image generation updates
cat docs/updates/image-gen.md

# View recent updates
head -50 docs/updates/api.md
```

### View overall changes
```bash
# View centralized changelog
cat docs/CHANGELOG.md
```

---

*Maintained by vllm-omni-skills auto-update system*
*Archive cycle: 4 weeks (aligned with vllm-omni release)*
