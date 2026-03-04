# vllm-omni-perf Update Log

> Last updated: 2026-03-04
> [View all skills updates](../CHANGELOG.md) | [Back to index](README.md)

---

### 2026-02-26
**[PR #1468](https://github.com/vllm-project/vllm-omni/pull/1468)** - process request.num_cached_tokens if it equals to the initial value

**Fixed**:
- <img width="908" height="214" alt="image" src="https://github.com/user-attachments/assets/c9cd8596-1fe9-405e-bfce-7f1e1b319e93" />
- ERROR:
- iteration_stats.prompt_token_stats=PromptTokenStats(ALL_SOURCES=('local_compute', 'local_cache_hit', 'external_kv_transfer'), computed=3809, **local_cache_hit=-1,** external_kv_transfer=0, **cached_tokens=-1**, recomputed_tokens=0, total=3808)

**Updated in skill**:
- ✅ (auto-marked)

---


### 2026-03-02
**[PR #1504](https://github.com/vllm-project/vllm-omni/pull/1504)** - Speed up diffusion model startup by multi-thread weight loading

**Added**:
- The weight loading time for large diffusion model are large, ~3min for QwenImage, ~5min for Wan2.2-I2V 14B. This PR reduce weight loading time by loading safetensors shards in parallel with a thread pool instead of sequentially.
- Helpful in:
- - Reduce wait time for CI or benchmarking board

**Updated in skill**:
- ✅ (auto-marked)

---


### 2026-02-26
**[PR #1518](https://github.com/vllm-project/vllm-omni/pull/1518)** - Use pull through cache image for H100 pool

**Changed**:
- so they don't run into rate limit

**Updated in skill**:
- ✅ (auto-marked)

---


## 2026-03 - Week 1

---

*Maintained by vllm-omni-skills auto-update system*
*Archived every 4 weeks (aligned with vllm-omni release cycle)*
