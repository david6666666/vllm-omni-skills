# vllm-omni-cicd Update Log

> Last updated: 2026-03-04
> [View all skills updates](../CHANGELOG.md) | [Back to index](README.md)

---

### 2026-02-26
**[PR #1435](https://github.com/vllm-project/vllm-omni/pull/1435)** - ComfyUI test, more screenshot, and code cleaning

**Changed**:
- The commits in this PR do the following:
- - Add integration test for the ComfyUI plugin. It runs the online serving in a subprocess with mocked AsyncOmni to skip actual generation. The purpose is to guard any changes to the API layer and ensures that API editors also remember to update API calls in the ComfyUI plugin.
- - Code cleaning as per #1113 comments after it is merged

**Updated in skill**:
- ✅ (auto-marked)

---


### 2026-02-26
**[PR #1438](https://github.com/vllm-project/vllm-omni/pull/1438)** - Streaming output

**Added**:
- Simple streaming output implementation for Qwene3TTS models for the latest disaggregated inference pipeline.
- one can test it with:
- ```

**Updated in skill**:
- ✅ (auto-marked)

---


### 2026-02-26
**[PR #1448](https://github.com/vllm-project/vllm-omni/pull/1448)** - Race condition in MultiprocExecutor when concurent access to Scheduler

**Fixed**:
- This PR fix the race condition bug in `MultiprocExecutor` when both `collective_rpc` and `add_req` access into `Scheduler`.
- The test can expose the error and code fix is given in the PR
- This is bug description provide in test file

**Updated in skill**:
- ✅ (auto-marked)

---


### 2026-02-26
**[PR #1449](https://github.com/vllm-project/vllm-omni/pull/1449)** - Reduce Perf test case and fix modify stage config

**Fixed**:
- Recover H100 test cases and fix full test
- run in ci
- ---

**Updated in skill**:
- ✅ (auto-marked)

---


### 2026-02-27
**[PR #1488](https://github.com/vllm-project/vllm-omni/pull/1488)** - enable cpu_offloading flag for non_cuda

**Changed**:
- Current cpu_offloading is only open to CUDA. However, CPU offloading is also very useful feature due to memory capacity issue such as intel arc B60.
- This PR aims to provide a new way to decide if certain device should provide cpu-offloading capability or not.
- Verified on XPU

**Updated in skill**:
- ✅ (auto-marked)

---


### 2026-02-26
**[PR #1492](https://github.com/vllm-project/vllm-omni/pull/1492)** - Enable layerwise offload on all hardware

**Changed**:
- This PR replace `torch.cuda.` by `current_omni_platform.`, so that other platforms also can use layerwise offload feature.
- ```
- vllm serve Qwen/Qwen-Image --omni --enable-layerwise-offload

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


### 2026-02-28
**[PR #1534](https://github.com/vllm-project/vllm-omni/pull/1534)** - Merge vllm pull 35368

**Changed**:
- Cherrypick the changes in vllm PR https://github.com/vllm-project/vllm/pull/35368 from @linyueqian .
- It helps #1367 #1519 and also may helps #1496 and #1447.
- Tested with:

**Updated in skill**:
- ✅ (auto-marked)

---


### 2026-02-28
**[PR #1543](https://github.com/vllm-project/vllm-omni/pull/1543)** - Modify some CI test cases to run on L4 environment to reduce H100 resource usage.

**Changed**:
- Modify some CI test cases to run on L4 environment to reduce H100 resource usage.
- 1. test benchmark testcase and abort testcase
- ` /workspace/.venv/bin/python -m pytest -sv tests/benchmarks/test_serve_cli.py tests/engine/test_async_omni_engine_abort.py --html=report.html --self-contained-html`

**Updated in skill**:
- ✅ (auto-marked)

---


## 2026-03 - Week 1

---

*Maintained by vllm-omni-skills auto-update system*
*Archived every 4 weeks (aligned with vllm-omni release cycle)*
