# Agent Operating Contract

This repository is an R&D artifact for the Beyond Matmul project. The north
star is one independently reviewed result in an external open-source ML project
showing that preserved tensor-contraction provenance causes an attributable
inference performance improvement. Matrix multiplication is the rank-2 case;
dense GEMM, batched GEMM, grouped GEMM, and generic contraction remain valid
fallbacks. The whitepaper and compact codebase explain, reproduce, and bound
that external result.

## Non-Negotiables

- Start from latest `origin/main`.
- Do all implementation in a fresh git worktree.
- Use TDD for code changes.
- Keep changes tasteful, minimal, and PR-sized.
- Treat GitHub issues as the state of the world.
- Do not self-merge implementation PRs.
- Keep wiki, docs, and whitepaper claims synchronized with executable evidence.
- Do not count metadata-only provenance, a synthetic proxy, an existing
  upstream optimization, or a comparison against a knowingly weak baseline as
  the external performance result.
- Do not generalize the matrix IR into a universal tensor IR before the active
  external target passes its documented validation gate.

## Worktree Discipline

Never implement in a dirty shared checkout. Begin work like this:

```bash
git fetch origin main
git worktree add -b codex/<issue-or-task-slug> ../beyond-matmul-<issue-or-task-slug> origin/main
cd ../beyond-matmul-<issue-or-task-slug>
```

Use branch prefix `codex/` unless the user explicitly asks otherwise. Keep one
issue per branch and one branch per worktree. Remove worktrees only after their
PR is merged or abandoned.

## TDD

For code changes:

1. Write or update the failing test first.
2. Run the narrow test and confirm it fails for the expected reason.
3. Implement the smallest coherent change.
4. Run the narrow test until it passes.
5. Run the full local CI-equivalent gate before opening a PR.

Docs-only changes do not need artificial tests, but they do need careful review
for duplication, stale claims, broken references, and consistency with the
whitepaper and wiki.

## Local CI Parity

Use the local hook script before pushing or opening a ready PR:

```bash
scripts/ci_local
```

This mirrors `.github/workflows/ci.yml`: locked dependency sync, unit tests,
demos, and benchmark smoke. If a future CI step is added, update
`scripts/ci_local` in the same PR.

Install local hooks in a worktree with:

```bash
git config core.hooksPath .githooks
```

## Taste And Minimalism

- Prefer existing IR, planner, frontend, and test patterns.
- Add abstractions only when they remove real duplication or express a stable
  project concept.
- Do not widen scope from one operator family to another in the same PR unless
  the issue explicitly requires it.
- Keep benchmarks honest: distinguish pure-Python proxies from performance
  evidence.
- Preserve dense GEMM, batched GEMM, grouped GEMM, and generic contraction as
  valid fallbacks. The claim is richer operator space, not "matmul bad."

## Issue State

Issues are the coordination layer for agents. Use labels as semaphores:

```text
status:ready status:claimed status:blocked status:review status:done status:stale
priority:p0 priority:p1 priority:p2
risk:low risk:medium risk:high
area:frontend area:ir area:planner area:benchmarks area:whitepaper area:wiki area:infra
kind:code kind:research kind:docs kind:test kind:retrospective
```

Each issue should include acceptance criteria and, when relevant, an explicit
`Blocked by: #...` section. Claim exactly one issue at a time. A claim comment
should include agent/thread id, branch, worktree path, timestamp, and intended
scope.

## PR Management

PRs must be linked to issues and include:

- `Closes #<issue>`.
- Summary of behavior or artifact changes.
- Tests and local CI-equivalent commands run.
- Docs/wiki/whitepaper updates made or explicitly not needed.
- Residual risks and follow-up issue links.

Reviewer agents check correctness, tests, minimality, and alignment with the
north star. They merge only after CI/local parity is green and issue state is
coherent. The implementing agent does not merge its own PR.

## Wiki And Whitepaper

The GitHub wiki is for humans and agents: concise definitions, north-star
criteria, current coverage, and operating loop. Avoid duplicated slop.

The whitepaper in `whitepaper/` is the cumulative research argument. Keep it
accurate to the codebase. Do not make claims that are not backed by tests,
demos, benchmark artifacts, or clearly marked future work.

The project-level north star is not complete merely because unsupported claims
are labeled as future work. Completion requires the external attributable
performance result defined above; a negative target-validation result should
reject or replace the target rather than redefine success after measurement.

## Definition Of Done

A task is done when the issue acceptance criteria are met, tests pass, local CI
parity passes, docs and claims are synchronized, the PR is reviewed by a
separate agent, and the linked issue is updated after merge.
