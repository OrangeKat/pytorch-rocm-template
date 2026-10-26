"""Check the PyTorch GPU backend and print actionable diagnostics.

Usage:
    uv run python scripts/gpu_check.py

On Linux+AMD this setup installs ROCm (HIP) torch wheels; PyTorch exposes the
AMD GPU through the standard ``torch.cuda`` API, so an available Radeon shows
up as device "cuda" and your training code uses it automatically.
"""

from __future__ import annotations

import platform


def main() -> None:
    import torch

    print(f"torch            : {torch.__version__}")
    print(f"torch.version.hip: {torch.version.hip}   (set => ROCm/HIP build, None => CUDA build)")
    print(f"platform         : {platform.system()} / {platform.machine()}")
    print(f"device           : {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if not torch.cuda.is_available():
        if torch.version.hip:
            print()
            print("ROCm build detected but NO GPU visible. Usual causes on Linux:")
            print("  1. amdgpu driver not installed: run `apt install amdgpu-dkms`")
            print("     (AMD's ROCm driver stack, matching the wheel's ROCm version)")
            print("  2. no permission: your user must be in the `video` and `render`")
            print("     groups: `sudo usermod -aG video,render $USER` (re-login)")
            print("  3. unsupported consumer GPU: set e.g.")
            print("       export HSA_OVERRIDE_GFX_VERSION=11.0.0   # RDNA3 (7900...")
            print("     (correct GFX depends on the card; see ROCm docs)")
            print("  4. check driver-side with `rocminfo` / `clinfo`")
        else:
            print()
            print("CUDA build (NVIDIA) with no GPU visible. If you have an AMD")
            print("GPU, this environment was synced without ROCm wheels — run")
            print("`uv sync` in this repo on Linux to get them.")
        return

    n = torch.cuda.device_count()
    print(f"visible GPUs     : {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        vram = props.total_memory / (1024 ** 3)
        print(f"  [{i}] {props.name}  ({vram:.1f} GiB, SM{props.multi_processor_count if hasattr(props, 'multi_processor_count') else '?'}, "
              f"gfx={getattr(props, 'gcnArchName', 'n/a')})")

    # Bring up runtime + one matmul on device 0.
    device = "cuda:0"
    print(f"  a 4096x4096 matmul on {torch.cuda.get_device_name(0)}...", end="", flush=True)
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    torch.cuda.synchronize()
    import time

    t0 = time.perf_counter()
    _ = a @ b
    torch.cuda.synchronize()
    print(f" {time.perf_counter() - t0 * 1_000:.1f} ms  OK")


if __name__ == "__main__":
    main()
