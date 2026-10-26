#!/usr/bin/env bash
# WSL2 + AMD Radeon helper for this template.
#
# Run INSIDE WSL:   bash scripts/setup_wsl_rocm.sh
# Needs sudo (you will be prompted) only when groups must be fixed.
#
# Checks, in order:
#   A. WSL kernel version — AMD GPU compute (/dev/kfd) requires a WSL kernel
#      >= 6.6 (Windows 11 24H2 + `wsl --update`). Older 5.15 kernels never
#      expose /dev/kfd, no matter what is installed on Windows.
#   B. HIP SDK on the Windows host (visible under /mnt/c/Program Files/AMD/
#      ROCm/...) and the Windows GPU driver.
#   C. /dev/dxg or /dev/kfd — the actual compute passthrough devices.
#   D. Group permissions for the current user.
#   E. The project's GPU check (uv run scripts/gpu_check.py).
set -euo pipefail

msg() { printf '\n\033[1;33m==> %s\033[0m\n' "$*"; }
ok()  { printf '  \033[1;32mOK\033[0m %s\n' "$*"; }
warn(){ printf '  \033[1;31m!!\033[0m %s\n' "$*"; }

IN_WSL=false
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    IN_WSL=true
fi

msg "Environment"
echo "    kernel: $(uname -r)"
if ! $IN_WSL; then
    echo "    (not a WSL kernel — native Linux path; skipping Windows checks)"
fi

# --- A. kernel -------------------------------------------------------------
KERNEL_MAJOR_MINOR="$(uname -r | cut -d. -f1,2)"
if $IN_WSL; then
    if dpkg --compare-versions "$KERNEL_MAJOR_MINOR" ge 6.6 2>/dev/null || \
       [[ "$KERNEL_MAJOR_MINOR" == "6.6"* || "$KERNEL_MAJOR_MINOR" == "6."* ]]; then
        ok "WSL kernel >= 6.6 — AMD KFD passthrough is supported"
    else
        warn "WSL kernel $KERNEL_MAJOR_MINOR is TOO OLD for AMD GPU compute."
        echo "    /dev/kfd only exists on WSL kernels >= 6.6 (AMD KFD-in-WSL)."
        echo "    This is an OS-level limitation — the HIP SDK cannot help"
        echo "    until the kernel is updated:"
        echo '      - Windows must be Windows 11 24H2 (build 26100+)'
        echo '        (PowerShell: [System.Environment]::OSVersion.Version, or just winver)'
        echo "      - Update WSL from PowerShell:   wsl --update    then: wsl --shutdown"
        echo "         (verify afterwards:  wsl --version  → kernel >= 6.6)"
    fi
fi

# --- B. HIP SDK on the Windows host ----------------------------------------
if $IN_WSL && [[ -d /mnt/c ]]; then
    msg "Checking HIP SDK on the Windows host"
    SDK_DIR=""
    for d in "/mnt/c/Program Files/AMD/ROCm"/*/; do
        [[ -d "$d" ]] && SDK_DIR="${d%/}"
    done
    if [[ -n "$SDK_DIR" ]]; then
        ok "HIP SDK found: $SDK_DIR"
        for exe in bin/hipInfo.exe bin/rocm-smi.exe bin/hipcc.cmd; do
            [[ -f "$SDK_DIR/$exe" ]] && echo "    present: ${exe##$SDK_DIR/}"
        done
        echo "    (version: $(basename "$SDK_DIR"))"
    else
        ls "/mnt/c/Program Files/AMD/" 2>/dev/null | sed 's/^/    AMD dir: /' || true
        warn "No HIP SDK found under C:\\Program Files\\AMD\\ROCm\\"
        echo "    Install 'HIP SDK for Windows' (or Adrenalin with SDK) from"
        echo "    https://www.amd.com/en/developer-resources/rocm-hub/hip-sdk.html"
        echo "    Verify on Windows (PowerShell):"
        echo "      ls 'C:\\Program Files\\AMD\\ROCm'"
        echo '      & "C:\Program Files\AMD\ROCm\<ver>\bin\hipInfo.exe"'
    fi
    # Windows GPU driver check
    cmd.exe /c wmic path win32_VideoController get name /value 2>/dev/null |
        grep "Name=" | head -4 | sed 's/^/    GPU: /' || true
fi

# --- C. compute device nodes ------------------------------------------------
msg "GPU device nodes"
[[ -e /dev/dxg ]]   && ok "/dev/dxg (GPU paravirtualization) present" || warn "/dev/dxg missing"
[[ -e /dev/kfd ]]   && ok "/dev/kfd (AMD KFD compute — what torch/HIP needs) present" || warn "/dev/kfd missing"
[[ -e /dev/dri/renderD128 ]] && ok "/dev/dri/renderD128 present" || warn "/dev/dri/renderD128 missing"

if [[ ! -e /dev/kfd ]]; then
    if $IN_WSL && [[ "$KERNEL_MAJOR_MINOR" != 6.* ]]; then
        echo
        warn "/dev/kfd cannot appear on this kernel — do the steps above (wsl --update)."
    else
        echo
        warn "/dev/kfd still missing on a new kernel; HIP SDK + driver may not"
        echo "    actually be the WSL-capable one, or WSL hasn't been restarted."
        echo "    From PowerShell: wsl --shutdown ; verify HIP SDK is the WSL build;"
        echo "    and ensure the WSL settings don't block device passthrough."
    fi
    echo
    echo "Re-run this script after fixing the Windows side."
    exit 1
fi

# --- D. permissions ----------------------------------------------------------
if ! id | grep -qw video || ! id | grep -qw render; then
    msg "Adding your user to the video and render groups (sudo password required)"
    sudo usermod -aG video,render "$USER"
    echo "    Group changes need a fresh WSL session: close ALL WSL terminals"
    echo "    and run 'wsl --shutdown' from PowerShell, then reopen."
    exit 0
else
    ok "video/render group membership"
fi

# --- E. the project's GPU check ----------------------------------------------
msg "Running the project's GPU check"
if command -v uv >/dev/null 2>&1; then
    uv run python scripts/gpu_check.py || true
else
    warn "uv not found — install it:  curl -LsSf https://astral.sh/uv/install.sh | sh"
fi
