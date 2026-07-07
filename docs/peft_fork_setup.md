# PEFT Fork Setup And Sync Rules

This note is the capstone setup contract for agents working in the
`alexlopashev/peft` fork for Beyond Matmul. It records repository mechanics and
coordination rules only; it does not implement the provenance-aware LoRA
optimization.

The current benchmark contract lives in
`docs/peft_capstone_benchmark_contract.md`. The active fork target is
`alexlopashev/peft` on branch
`beyond-matmul/provenance-lora-inference`, compared against upstream
`huggingface/peft`.

## Clone And Remotes

Use a dedicated PEFT checkout or worktree outside the Beyond Matmul repository.
Keep the fork and upstream remotes named explicitly:

```bash
git clone git@github.com:alexlopashev/peft.git peft-beyond-matmul
cd peft-beyond-matmul
git remote add upstream git@github.com:huggingface/peft.git
git fetch origin
git fetch upstream
git checkout beyond-matmul/provenance-lora-inference
```

If the checkout already exists, verify the remotes before editing:

```bash
git remote -v
git branch --show-current
git status --short
```

Use `origin` for `alexlopashev/peft` and `upstream` for `huggingface/peft`.
Do not push Beyond Matmul integration commits directly to `upstream`.

## Branch Naming

The long-lived integration branch is:

```text
beyond-matmul/provenance-lora-inference
```

Use that branch as the benchmark fork ref until a later issue replaces it with
a pinned commit. For issue-sized PEFT changes, prefer a short-lived branch under
the same namespace:

```text
beyond-matmul/<beyond-matmul-issue-number>-<short-slug>
```

Examples:

```text
beyond-matmul/77-lora-provenance-design
beyond-matmul/78-provenance-lora-inference
```

When a short-lived branch is reviewed and kept, merge or fast-forward it into
`beyond-matmul/provenance-lora-inference` inside the fork. The benchmark harness
should record the exact PEFT fork commit it measured, not only the branch name.

## Dependency Install

Follow PEFT's source install path for fork work. In a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

If the `test` extra changes upstream, prefer PEFT's current contributing guide
over this note. Keep dependency changes out of Beyond Matmul docs unless they
are required to reproduce the capstone benchmark.

## Test Command Discovery

Before editing PEFT code, discover the current local test commands from the
fork checkout rather than copying stale commands from memory:

```bash
sed -n '1,180p' Makefile
sed -n '1,180p' pyproject.toml
rg -n "pytest|make test|make quality" docs/source/developer_guides README.md
```

At the time this note was written, PEFT exposes these useful commands:

```bash
make quality
make test
python -m pytest tests/<test-file-name> -k <name-of-test>
```

For the first Beyond Matmul LoRA inference work, the narrow test should come
from the touched LoRA or model file, then `make quality` and the relevant PEFT
test target should run before opening a PEFT PR. The Beyond Matmul benchmark
harness remains validated separately with `scripts/ci_local`.

## Upstream Sync

Before starting PEFT fork work:

```bash
git fetch upstream main
git fetch origin
git checkout beyond-matmul/provenance-lora-inference
git merge --ff-only upstream/main
```

If `--ff-only` fails, stop and decide whether to merge or rebase in the PEFT
fork. Prefer a visible merge commit for shared integration branches. Rebase only
short-lived personal branches that have not been reviewed or used by a
benchmark artifact.

After syncing, push the fork branch:

```bash
git push origin beyond-matmul/provenance-lora-inference
```

When a benchmark artifact is produced, record the immutable PEFT commit SHA in
the artifact. Branch names are acceptable for local development only.

## Minimum Likely PEFT Touch Points

The first integration should stay inside the smallest LoRA inference surface
until issue #77 proves a wider design is necessary. The likely PEFT files are:

- `src/peft/tuners/lora/layer.py`: primary LoRA layer runtime, adapter factors,
  merge state, and forward-time application.
- `src/peft/tuners/lora/model.py`: LoRA module replacement and adapter wiring
  around target modules.
- `src/peft/tuners/lora/config.py`: only if the fork needs an explicit
  opt-in flag or metadata field for provenance-aware inference.
- `src/peft/peft_model.py`: only if the benchmark needs model-level metadata
  about active adapters or the lowering path used.
- `src/peft/mixed_model.py`: only if mixed-adapter behavior affects the first
  benchmark case; otherwise keep it out of scope.
- `tests/test_custom_models.py`, `tests/test_tuners_utils.py`, or the relevant
  LoRA test file: narrow output-equivalence and fallback coverage.

Avoid broad PEFT method work, training/autograd behavior, generation loops,
KV-cache behavior, quantization backends, GPU kernels, and universal
Transformer coverage for the first fork integration unless a later issue
explicitly changes the scope.

## Beyond Matmul Issue Mapping

Beyond Matmul issues remain the source of truth for capstone coordination.
Every PEFT fork branch or PR created for this capstone should reference the
Beyond Matmul issue number in its branch name, PR body, and issue comments.

- #74 `Define PEFT capstone benchmark contract`: Beyond Matmul docs only,
  merged in PR #84. No PEFT fork PR.
- #75 `Document PEFT fork setup and sync rules`: Beyond Matmul docs only. No
  PEFT fork PR.
- #76 `Build TorchBench-style PEFT upstream-vs-fork harness`: Beyond Matmul
  harness only, merged in PR #85. It targets
  `alexlopashev/peft@beyond-matmul/provenance-lora-inference` for manual fork
  runs.
- #77 `Design PEFT low-rank provenance integration points`: design issue. If
  it needs a PEFT branch for inspection artifacts, use
  `beyond-matmul/77-lora-provenance-design` and link it from the issue. The
  design note lives in `docs/peft_low_rank_provenance_design.md`.
- #78 `Implement provenance-aware LoRA inference path in PEFT fork`:
  implementation issue. Use a PEFT branch such as
  `beyond-matmul/78-provenance-lora-inference`, then merge or fast-forward the
  reviewed result into `beyond-matmul/provenance-lora-inference`. Start from
  the #77 handoff checklist instead of redoing PEFT integration discovery.
- #79 and later benchmark-result issues: use the exact PEFT commit measured
  from `beyond-matmul/provenance-lora-inference`; link any PEFT PR that changed
  that commit.

If a PEFT PR is opened in `alexlopashev/peft`, add a comment on the Beyond
Matmul issue with the PR URL and the intended fork branch. If an upstream PR to
`huggingface/peft` is later considered, coordinate that under a separate Beyond
Matmul issue; do not treat upstreaming as part of the first benchmark
implementation by default.
