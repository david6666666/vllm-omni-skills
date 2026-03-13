---
name: vllm-omni-bug
description: Use when reviewing bug issues or [Bug]/[Bugfix] PRs on vllm-project/vllm-omni. Drives regression reproduction and test coverage analysis via the bug-test-coverage procedure.
---

# vLLM-Omni Bug Review

## Overview

This skill is triggered for PRs with prefix `[Bug]` or `[Bugfix]` (and for bug-issue review). It provides a structured procedure to decide whether regression tests are needed and to design minimal test plans.

## Procedure

**Follow [references/bug-test-coverage.md](references/bug-test-coverage.md)** for:

1. Classifying the bug (code issue, edge case, environment-specific, flake, non-code/config)
2. Mapping to existing coverage (unit, e2e, perf, model-related)
3. Deciding whether new tests are **required**, **recommended**, or **not_needed**
4. Designing a minimal test plan (level, scenario, preconditions, assertions)
5. Specifying missing assertions and assessing CI impact

## When to Use

- Reviewing a GitHub issue labeled or titled as a bug
- Reviewing a PR with title prefix `[Bug]` or `[Bugfix]`
- Deciding if a bugfix PR needs a regression test and what that test should cover

## References

- [Bug Test Coverage](references/bug-test-coverage.md) — Full procedure, output template, examples, and vLLM-Omni test level reference
