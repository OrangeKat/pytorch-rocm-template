"""Verify that PyTorch works correctly on this machine's GPU.

Usage:
    uv run python main.py

Runs a series of checks on the GPU (via the standard ``torch.cuda`` API,
which also covers AMD ROCm/HIP) and compares results against the CPU
reference, printing a PASS/FAIL summary at the end.
"""

from __future__ import annotations

import sys
import time

CHECKS: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    CHECKS.append((name, ok))


def main() -> None:
    import torch

    print(f"torch {torch.__version__}")
    print(f"hip  : {torch.version.hip}   (ROCm/HIP build if not None)")
    print(f"cuda : {torch.version.cuda}")
    print()

    if not torch.cuda.is_available():
        print("NO GPU visible to torch.cuda — nothing to test.")
        if torch.version.hip:
            print("ROCm build installed. If you have an AMD GPU, check the")
            print("driver / permissions (see scripts/gpu_check.py).")
        else:
            print("This looks like a CUDA or CPU build of torch.")
        sys.exit(1)

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / (1024**3)
    print(f"GPU  : {props.name}  ({vram:.1f} GiB)  [{device}]")
    print()

    # --- 1. Device info -----------------------------------------------------
    check("device count", torch.cuda.device_count() >= 1)

    # --- 2. Tensor allocation on GPU ---------------------------------------
    x = torch.randn(1024, 1024, device=device)
    check("tensor allocation", x.device.type == "cuda", f"{x.device}")

    # --- 3. Compute correctness: CPU vs GPU matmul --------------------------
    torch.manual_seed(0)
    a_cpu = torch.randn(512, 512)
    b_cpu = torch.randn(512, 512)
    ref = a_cpu @ b_cpu
    got = (a_cpu.to(device) @ b_cpu.to(device)).cpu()
    max_err = (ref - got).abs().max().item()
    check("matmul matches CPU", max_err < 1e-3, f"max abs err {max_err:.2e}")

    # --- 4. Elementwise + reductions on GPU ---------------------------------
    y = torch.arange(0, 1000, device=device, dtype=torch.float32)
    check("sum matches CPU", y.sum().item() == y.cpu().sum().item())
    check(
        "softmax matches CPU",
        torch.allclose(
            torch.softmax(y, dim=0).cpu(), torch.softmax(y.cpu(), dim=0)
        ),
    )

    # --- 5. Autograd on GPU -------------------------------------------------
    w = torch.randn(64, 32, device=device, requires_grad=True)
    out = (w @ torch.randn(32, 16, device=device)).sum()
    out.backward()
    check("backward pass", w.grad is not None and w.grad.shape == w.shape)

    # --- 6. Small training loop converges -----------------------------------
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
    ).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    xs = torch.randn(256, 16, device=device)
    ys = (xs.sum(dim=1, keepdim=True) * 2.0)
    initial = None
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(xs), ys)
        if initial is None:
            initial = loss.item()
        loss.backward()
        opt.step()
    check(
        "training loop reduces loss",
        loss.item() < initial * 0.1,
        f"{initial:.4f} -> {loss.item():.4f}",
    )

    # --- 7. Sanity: output stays on device, dtype preserved -----------------
    big = torch.randn(2048, 2048, device=device, dtype=torch.float16)
    check("half precision matmul", (big @ big).dtype == torch.float16)

    # --- 8. Performance smoke test ------------------------------------------
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = torch.randn(4096, 4096, device=device) @ torch.randn(4096, 4096, device=device)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    check("4096x4096 matmul", dt < 5.0, f"{dt * 1000:.1f} ms")

    print()
    if all(ok for _, ok in CHECKS):
        print(f"All {len(CHECKS)} checks PASSED — PyTorch works on {props.name}.")
    else:
        failed = [n for n, ok in CHECKS if not ok]
        print(f"{len(failed)} check(s) FAILED: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
