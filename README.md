# PyTorch on AMD GPU — uv template

A minimal [uv](https://docs.astral.sh/uv/) project that runs PyTorch on AMD
GPUs (ROCm) on Linux, and on stock CUDA/CPU/Metal PyTorch everywhere else.

## Setup

**Requirements:** Linux (or WSL2), an AMD GPU supported by ROCm, and `uv`.
If `rocminfo` already lists your GPU, skip to step 2.

```bash
# 1. Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Vendor AMD's ROCm wheels, then create the environment (once)
python3 scripts/prepare_rocm_wheels.py
uv lock && uv sync --all-groups

# 3. Verify
uv run python scripts/gpu_check.py   # expect: device : cuda
uv run python main.py                # full correctness + training smoke test
```

No ROCm driver yet? Use AMD's official
[native Linux guide](https://rocm.docs.amd.com/en/latest/install/rocm.html) or
the [WSL2 steps below](#wsl2-driver-install--ubuntu-2404-radeon-rx-7900-gre-gfx1100).

Disk usage: ~1.9 GB for the vendored wheels and ~6.4 GB for `.venv`.

On WSL2, run `bash scripts/setup_wsl_rocm.sh` before step 3 — it checks the
driver, `/dev/kfd`, and group permissions instead of leaving you to debug the
first failed GPU probe.

## WSL2 driver install — Ubuntu 24.04, Radeon RX 7900 GRE (gfx1100)

<details>
<summary>Adapted from AMD's official ROCm 10.1 WSL2 guide (Technical Preview)</summary>

### On Windows

1. Uninstall any existing AMD HIP SDK or graphics packages.
2. Install WSL2 with Ubuntu 24.04:
   `wsl --install -d Ubuntu-24.04`.
3. Install **AMD Software: Adrenalin Edition** from the
   [ROCm 10.1 release notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-ROCM-10-01.html#Downloads),
   then reboot.

### Inside WSL2

```bash
sudo apt update
sudo apt install libatomic1 libquadmath0

# Register the AMD ROCm 10.1 apt repository
sudo mkdir -p /etc/apt/keyrings
wget https://stable.repo.amd.com/rocm/gpg/packages.gpg -O - | \
  gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg > /dev/null

sudo tee /etc/apt/sources.list.d/amdrocm-stable.sources > /dev/null <<'EOF'
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
sudo apt install amdrocm10.1-gfx1100
```

No kernel driver or permission setup is needed on WSL2 — the GPU is reached
through the Windows Adrenalin driver.

### Verify

```bash
rocminfo   # should print "WSL2 environment detected." and list the GPU
hipinfo    # lists detected AMD GPUs
```

### Caveats

- Do **not** copy ROCm compiler/runtime DLLs into `C:\Windows\System32`.
- `rocgdb` and `rocm-debug-agent` are not supported on WSL2.
- Remove the stack with `sudo apt autoremove amdrocm10.1-gfx1100`.
- Native Linux installs the driver with `amdgpu-install` instead (brand-new
  Radeon cards need ROCm ≥ 7.2.4 uninstalled first) — see the
  [official guide](https://rocm.docs.amd.com/en/latest/install/rocm.html).

</details>

## Repo layout

| Path | Contents |
|------|----------|
| `main.py` | end-to-end GPU verification: allocation, matmul vs CPU, autograd, tiny training loop |
| `scripts/gpu_check.py` | backend/GPU diagnostics and a matmul smoke test |
| `scripts/prepare_rocm_wheels.py` | retags + vendors AMD's ROCm wheel closure for `uv lock` |
| `scripts/setup_wsl_rocm.sh` | WSL2 checks: kernel, HIP SDK, `/dev/kfd`, permissions |
| `vendor/pytorch-rocm/` | vendored, retagged ROCm wheels (gitignored, ~1.9 GB) |
| `pyproject.toml` | torch pins per OS + `[tool.uv]` find-links |

## Why the wheels are vendored

AMD publishes its ROCm PyTorch wheels
(`https://stable.repo.amd.com/rocm/whl-next/`) with the legacy
`linux_x86_64` platform tag, which `uv lock` refuses. The workaround is
`scripts/prepare_rocm_wheels.py`: it downloads each AMD wheel once, retags it
`linux_x86_64` → `manylinux_2_28_x86_64`, rebuilds `RECORD`, follows the full
dependency closure on the AMD index, and stages the result in
`vendor/pytorch-rocm`. `pyproject.toml` points uv at that directory with
`[tool.uv] find-links`, so the lock is reproducible.

## Switching backend, GPU family, or torch version

- **CUDA (NVIDIA) or CPU-only:** remove the `==...+rocm*` pins and the
  `[tool.uv] find-links` entry, keep torch on PyPI. On Linux,
  `torch.cuda.is_available()` then means an NVIDIA GPU.
- **Different AMD GPU family** (the default targets `gfx1100`, RDNA3): change
  the `torch[device-gfx1100]` extra in `pyproject.toml` and `SEED_WHEELS` in
  `scripts/prepare_rocm_wheels.py` (e.g. `gfx90a`, `gfx1200`), then re-run the
  vendoring step.
- **torch bump:** update `SEED_WHEELS` and the pins in `pyproject.toml`,
  re-run `prepare_rocm_wheels.py`, then `uv lock`.

## License

MIT — see `LICENSE`.
