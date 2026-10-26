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

## Installing ROCm drivers — WSL2, Ubuntu 24.04.5, Radeon RX 7900 GRE (gfx1100)

Adapted from AMD's [official ROCm install guide](https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=radeon&w=compute&gpu=amd-radeon-rx-7900-gre&os=wsl&ubuntu-ver=24.04.5&i=pkgman&gfx=gfx1100).
Note: WSL support in ROCm 10.1.0 is a **Technical Preview** — expect it to
improve in later releases.

### On Windows (do this first)

1. **Remove conflicting AMD software**: uninstall any existing HIP SDK for
   Windows or other AMD graphics packages (*Control Panel > Programs >
   Uninstall a program*).
2. **Install WSL2 and Ubuntu 24.04** — follow
   [Microsoft's guide](https://learn.microsoft.com/en-us/windows/wsl/install),
   e.g. `wsl --install -d Ubuntu-24.04`.
3. **Install AMD Software: Adrenalin Edition** on Windows — this provides the
   kernel driver half that WSL reaches through. Grab it from AMD's
   [ROCm 10.1 release notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-ROCM-10-01.html#Downloads),
   then reboot your Windows host.

### Inside WSL2 (Ubuntu 24.04.5)

No kernel driver or permission configuration is needed here — WSL2 sees the
GPU through the Windows Adrenalin driver.

```bash
# 1. Additional packages required by some ROCm tools
sudo apt update
sudo apt install libatomic1 libquadmath0

# 2. Register the AMD ROCm 10.1.0 apt repository
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://stable.repo.amd.com/rocm/gpg/packages.gpg -O - | \
gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg > /dev/null

sudo tee /etc/apt/sources.list.d/amdrocm-stable.sources << EOF
X-Repo-Id: amdrocm-stable
Types: deb
URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2404/
Suites: stable
Components: main
Architectures: amd64
Signed-By: /etc/apt/keyrings/amdrocm.gpg
Enabled: yes
EOF

sudo apt update

# 3. Install the ROCm meta package for gfx1100 (RX 7900 GRE family)
sudo apt install amdrocm10.1-gfx1100
```

### Verify

```bash
rocminfo   # should print "WSL2 environment detected." and list the Radeon GPU
hipinfo    # lists detected AMD GPUs with runtime properties
```

Then continue with [Quick start](#quick-start) — `scripts/setup_wsl_rocm.sh`
will confirm the kernel/HIP SDK/permission setup before the torch probe.

### ROCm-on-WSL2 caveats

- Do **not** copy the ROCm compiler/runtime DLLs into `C:\Windows\System32` —
  this is explicitly warned against and causes conflicts.
- `rocgdb` and `rocm-debug-agent` are *not* supported on WSL2 (no KFD debug
  interface).
- To remove the ROCm stack again: `sudo apt autoremove amdrocm10.1-gfx1100`.
- Native Linux (non-WSL) instead installs the kernel driver via
  `amdgpu-install` (brand-new Radeon cards need ROCm ≥7.2.4 uninstalled first)
  — see the [full guide](https://rocm.docs.amd.com/en/latest/install/rocm.html)
  for your distro.

## Repo layout

| Path | Contents |
|------|----------|
| `scripts/prepare_rocm_wheels.py` | retags + vendors AMD's legacy-tagged ROCm wheel closure for `uv lock` |
| `scripts/gpu_check.py` | backend/GPU diagnostics: HIP vs CUDA build, driver, groups, tools |
| `main.py` | end-to-end GPU verification: alloc, matmul correctness vs CPU, autograd, tiny training loop |
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
`scripts/gpu_check.py`; end-to-end correctness/compute checks run with
`uv run python main.py`.

Re-run `prepare_rocm_wheels.py` after any torch version bump (update both
`[project.dependencies]` and `SEED_WHEELS`).

## License

MIT — see `LICENSE`.
