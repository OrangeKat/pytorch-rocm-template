# PyTorch on AMD GPU — uv template

A minimal template for uv-managed projects that run PyTorch on AMD GPUs
(ROCm) on Linux, with stock CUDA/CPU/Metal PyTorch everywhere else.

## Why the vendoring step exists

AMD publishes its ROCm PyTorch wheels
(`https://stable.repo.amd.com/rocm/whl-next/`) with the legacy
`linux_x86_64` platform tag, which `uv lock` never accepts. The workaround
is `scripts/prepare_rocm_wheels.py`: it downloads each required AMD wheel
once, retags it `linux_x86_64` → `manylinux_2_28_x86_64` (the same glibc
floor AMD's own manylinux-tagged wheels carry), rebuilds `RECORD`, and
follows the full dependency closure on the AMD index. The lockable result
is staged in `vendor/pytorch-rocm`, wired into `pyproject.toml` via
`[tool.uv] find-links` (find-links covers transitive deps, unlike
`[tool.uv.sources]`).

## Quick start

```bash
python3 scripts/prepare_rocm_wheels.py      # once; vendors the retagged AMD ROCm closure
uv lock && uv sync --all-groups
uv run python scripts/gpu_check.py          # expect device: cuda (ROCm/HIP build)
```

A ~1.9 GB `.venv` is created; make sure you have disk headroom.

On WSL2, run `bash scripts/setup_wsl_rocm.sh` first (kernel/HIP SDK/permission
checks) rather than trial-and-erroring the first failed GPU probe.

## Repo layout

| Path | Contents |
|------|----------|
| `scripts/prepare_rocm_wheels.py` | retags + vendors AMD's legacy-tagged ROCm wheel closure for `uv lock` |
| `scripts/gpu_check.py` | backend/GPU diagnostics: HIP vs CUDA build, driver, groups, tools |
| `scripts/setup_wsl_rocm.sh` | WSL2 + AMD Radeon setup checks (kernel, HIP SDK, `/dev/kfd`, permissions) |
| `vendor/pytorch-rocm/` | vendored, retagged ROCm wheels (gitignored; ~1.9 GB) |
| `pyproject.toml` | torch pins per OS + `[tool.uv]` find-links comments |

## Switching backend / GPU family

- **CUDA (NVIDIA) or CPU-only**: only the markers matter — remove the
  `==...+rocm*` pins and the `[tool.uv] find-links` entry, keep torch on
  PyPI. `torch.cuda.is_available()` then means an NVIDIA GPU.
- **Different AMD GPU family** (the default targets `gfx1100`, RDNA3):
  change the `torch[device-gfx1100]` extra in `pyproject.toml` and the
  `SEED_WHEELS` comment in `scripts/prepare_rocm_wheels.py` to your GPU
  (e.g. `[device-gfx90a]`, `[device-gfx1200]`), then re-run the vendoring
  step. Find your family via the ROCm compatibility docs for your GPU.
- **torch bump**: update `SEED_WHEELS` in the vendor script and the pins in
  `pyproject.toml`, re-run `prepare_rocm_wheels.py`, then `uv lock`.

## Sanity check

The canonical check is that torch exposes your AMD GPU through the standard
`torch.cuda` API:

```python
import torch
print(torch.version.hip)          # e.g. "10.1.0" on a ROCm build; None on CUDA
print(torch.cuda.is_available())  # True when the GPU is visible
print(torch.cuda.get_device_name(0))
```

Full diagnostics (driver groups, missing tools, permission hints) live in
`scripts/gpu_check.py`.

Re-run `prepare_rocm_wheels.py` after any torch version bump (update both
`[project.dependencies]` and `SEED_WHEELS`).

## License

MIT — see `LICENSE`.
