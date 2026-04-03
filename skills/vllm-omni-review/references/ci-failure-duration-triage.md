## CI Failure & Duration Triage

## Quick Start

When a user reports CI is **failing (red) / flaky / noticeably slower** and provides logs:

1. Focus only on the **first error** (the earliest fatal signal). Don’t get distracted by cascading follow-up errors.
2. Extract: job name/level, the stage/step where the first error occurs, exit code/exception stack, whether it’s a timeout, and per-stage durations (if available).
3. Classify into one of five categories: **build/install**, **test failure**, **infrastructure**, **timeout/perf regression**, **config/environment**.
4. Provide **1–3** root-cause hypotheses. Each must include an **exact log evidence excerpt**, and hypotheses must be ordered **from lowest verification cost to highest**.
5. Provide a **≤5-minute** minimal verification for **Hypothesis 1** only. Specify **environment + commands + expected result** (must not require running the full pipeline).

## Log-reading guardrail (must follow)

Do **not** triage by reading only the log tail. Many CI logs end with teardown noise, process-kill cascades, or warning lines that are not the first failure.

Required sequence:

1. Fetch the **full raw log** first (or confirm whether the current log view is truncated).
2. Locate pytest failure anchors before reading linearly:
   - `=========== FAILURES ===========`
   - `short test summary info`
   - first `FAILED tests/...` or first `Traceback` with assertion details
3. If you must inspect by line number, jump to the user-provided line and then read **both directions** (at least ±100 lines).
4. Treat warning-only lines (e.g., ffmpeg options, deprecation warnings) as non-fatal unless they are immediately followed by a fatal exit/assertion.
5. Report the **earliest fatal signal** (assertion/exception/exit reason), not the last error-looking line in teardown.

If only a partial/truncated log is available, explicitly state:
- “cannot attribute from current logs: raw log appears partial/truncated”
- and request the raw/full log or the missing window around the first failure block.

## Required inputs (list what’s missing if not provided)

- Job name and type (deploy / L1–L5 / other)
- Logs: **50–100 lines before and after the first error** (or the full log)
- Change: PR link or commit SHA (if unknown, explicitly mark as “unknown”)
- CI environment info (runner/image/Python/CUDA, etc.; cite from logs if present)

## Decision tree (branch on the first error)

- **Build/install failures**
  - Common signals: `SyntaxError`, compiler `error:`, `ld:`/`linker`, `undefined reference`, `CMake Error`, `nvcc fatal`, `ModuleNotFoundError`, `ImportError`, `pip install` failures, `Failed building wheel`, `No matching distribution found`
- **Test failures**
  - Common signals: pytest `FAILED`, `AssertionError`, `E   assert ...`, traceback pointing to a `tests/...` line, or still failing after flaky retries
  - Sub-classify the test failure as one of:
    - **Functional test** (feature behavior, API contract, workflow correctness, edge/error handling)
    - **Performance test** (latency/throughput/runtime/memory regression, perf guardrails)
    - **Accuracy test** (model output quality, numerics, golden outputs, tolerance)
- **Infrastructure issues**
  - Common signals: `Timeout`/`timed out`, `Killed`/`SIGKILL`, `OOM`/`Out of memory`, `No space left on device`, `Disk quota exceeded`, `Connection reset`, `TLS handshake`, `503`/`429`, image pull failures
- **Timeout/performance regressions**
  - Common signals: a stage/step duration increases significantly in logs/summary; approaches timeout threshold; or long periods with no output
- **Configuration/environment issues**
  - Common signals: `Permission denied`, `AccessDenied`, `401/403`, `missing required env`, `KeyError: <ENV>`, secrets/tokens not injected, `could not read credentials`

## Test failure sub-classification: functional vs performance vs accuracy

When classification is **test failure**, you must add a sub-type: **functional**, **performance**, or **accuracy**.

### Functional test

Use **functional** when the first failure is about expected behavior, API contract, control flow, or error handling correctness.

Common signals (look for these near the first error):

- API/contract behavior mismatches: wrong status/code/message, missing keys/fields, schema mismatches
- Workflow/logic mismatches: branch conditions, retries/fallbacks/tool calls, state transitions not matching expected behavior
- Edge/error path checks: invalid input handling, exception type/message mismatch, timeout/error propagation assertion failures
- Integration behavior checks: mocked dependency interactions, call order/count assertions, side-effect checks (file/db/message)

Evidence pattern:

- The failure line typically includes **behavioral assertions** (state/output/exception/call sequence), not metric thresholds.
- The traceback points into tests validating endpoint behavior, pipeline flow, or business logic expectations.

### Accuracy test

Use **accuracy** when the first failure is an assertion about output quality, score, or numeric tolerance.

Common signals (look for these near the first error):

- Golden / expected output mismatches: `expected`, `got`, `mismatch`, `diff`, `golden`, `baseline`, `reference`, `snapshot`
- Numeric tolerance issues: `atol`, `rtol`, `tolerance`, `allclose`, `max_abs_err`, `cosine`, `psnr`, `ssim`
- Quality metric regressions: `accuracy`, `bleu`, `rouge`, `wer`, `cer`, `mAP`, `F1`, `pass@k` (or similar task metrics)
- Determinism drift often shows up as: `seed`, `random`, `nondeterministic`, `torch.backends.cudnn.deterministic`

Evidence pattern:

- The failure line typically includes **an assertion and compared values/metrics**, not runtime thresholds.
- The traceback points into a test that computes a metric or compares outputs against a known-good reference.

### Performance test

Use **performance** when the first failure is a regression check on runtime/throughput/latency/memory, or when the test is explicitly a perf guardrail.

Common signals:

- Runtime thresholds: `took`, `elapsed`, `timeout`, `exceeded`, `slower`, `regression`, `p95`, `p99`, `latency`, `throughput`, `tok/s`, `tokens/s`, `iters/s`, `qps`
- Benchmark-style naming: `benchmark`, `perf`, `performance`, `microbench`, `speed`, `profile`
- Resource guardrails: `peak memory`, `RSS`, `VRAM`, `cuda memory`, `OOM` *inside the test assertion* (not runner OOM)

Evidence pattern:

- The failure line usually includes a **measured value vs threshold/baseline**, e.g. “X is 1.4× slower than baseline” or “p95 latency > limit”.

### Tie-breakers (when logs are ambiguous)

- If the first error references **time/throughput/memory limits**, treat as **performance**.
- If the first error is about **quality scores, numeric tolerance, or golden/reference output**, treat as **accuracy**.
- If the first error is about **API contract, workflow logic, exception handling, or integration behavior**, treat as **functional**.
- If the job fails by **global timeout** (no assertion, runner killed), classify as `timeout/perf regression` (not test failure).

## Output (must use this template)

Rules:

- Hypotheses must be grounded in log facts. If there’s no evidence, write “cannot attribute from current logs” and list what additional info is needed.
- For duration issues, explicitly state: **stage name + baseline vs current run**.
- If there are multiple hypotheses, mark priority (usually ordered by verification cost from low to high).

```markdown
## CI Triage Report

- **Job**: <job name> / <L1|L2|…> / <trigger>
- **Change**: <PR or SHA or unknown>
- **Classification**: <build/install | test failure | infrastructure | config/environment | timeout/perf regression | other>
- **Test sub-type (only if test failure)**: <functional | performance | accuracy | unknown>

### First error

- **Location**: <stage or step name>
- **Excerpt**:
```
<verbatim log lines>
```

### Duration (if relevant)

- **Anomalous stage**: <stage name>
- **Baseline vs current**: <e.g., 2min → 10min> or <multiplier>
- **Notes**: <log-based only: serialized waiting/retries/cache misses, etc.>

### Root-cause hypotheses (ordered by verification priority)

**Hypothesis 1 (verify first)**
- **Description**: <one sentence>
- **Evidence**:
```
<log excerpt>
```

**Hypothesis 2**
- **Description**: …
- **Evidence**: …

(Optional hypothesis 3)

### Minimal verification plan (for Hypothesis 1)

- **Environment**: <local / specific container image / minimal CI workflow name>
- **Steps**:
  1. `<command or action A>`
  2. `<command or action B>`
- **Expected result**: <success | reproduce an error | duration within a range>

### Recommended actions

- If verification **confirms**: <fix direction or recommend reverting the change>
- If verification **does not confirm**: <move to next hypothesis or expand logs/contact infra>
- If it’s **infra/environment**: <adjust parallelism/timeouts/resources or include key points for an ops ticket>
```

## Linking CI failures to a specific change

When you triage a CI failure from Buildkite (for example
`https://buildkite.com/...`), you should
always first identify **which change** the build is running:

- **Branch / commit**: at the top of the Buildkite job page you will see something like
  `Example:main / xxxx (#xxxx)`.
  - `Example:main` is the branch.
  - `xxxx` is the short commit SHA.
  - `#xxxx` is the PR number.
- Use these links to decide whether the failure is:
  - **Change-induced**: the failure only appears on this PR/commit, and does not
    reproduce on the current `main` (or the base branch).
  - **Pre‑existing or infra**: the same job is already red/flaky on `main` or on
    unrelated PRs.

Recommended checks:

1. Open the **commit link** and quickly scan the diff for files directly related to the
   failing job (e.g., TTS tests vs. engine code).
2. Open the **PR link** to see title, description, and any explicit claims (perf,
   correctness, infra, etc.).
3. Optionally compare with a recent **green run on `main`** for the same job to see
   whether the failure pattern is new.

When you fill in the triage template above, use:

- `Change`: set to the PR or full SHA (e.g. `#xxxx / xxxx`).
- In **Recommended actions**, explicitly say whether the evidence points to “regression
  introduced by this PR/commit” vs. “likely pre‑existing / infra issue”.

## Additional resources

- L1–L5 levels and directory conventions: `https://github.com/vllm-project/vllm-omni/blob/main/docs/contributing/ci/CI_5levels.md`
