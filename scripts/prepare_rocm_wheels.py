#!/usr/bin/env python3
"""Vendor + re-tag the AMD ROCm wheel stack so `uv lock` can pin it.

AMD's ROCm PyTorch wheels (https://stable.repo.amd.com/rocm/whl-next/) are
published with the legacy ``linux_x86_64`` platform tag. ``uv pip`` edits
those fine, but ``uv lock`` never considers them eligible — which makes a
uv-managed, reproducible install impossible as-is.

Fix: download each required AMD wheel once, rewrite its ``WHEEL`` tag
``linux_x86_64`` -> ``manylinux_2_28_x86_64`` (AMD's stack targets glibc
>= 2.28 anyway, the same floor AMD's own manylinux-tagged wheels carry),
rebuild ``RECORD``, read the wheel's METADATA, and follow every dependency
that exists on the AMD index transitively. The result is a complete,
lockable closure staged in ``vendor/pytorch-rocm`` (a find-links directory
wired into pyproject.toml); packages not on the AMD index stay on PyPI.

Usage:
    python3 scripts/prepare_rocm_wheels.py          # download + vendor
    python3 scripts/prepare_rocm_wheels.py --plan   # resolve, no download
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import email.parser as email_parser
from pathlib import Path

AMD_INDEX = "https://stable.repo.amd.com/rocm/whl-next/"

# Seed pins: the AMD ROCm PyTorch line for gfx1100 (RDNA3), as given for
# this GPU. Note torchvision/torchaudio builds were realigned to the ones
# that actually exist on the stable index for rocm10.1.0.
SEED_WHEELS = [
    ("torch", "2.14.0+rocm10.1.0"),
    ("torchvision", "0.27.0+rocm10.1.0"),
    ("torchaudio", "2.11.0.2+rocm10.1.0"),
]

PY_TAG = "cp312"
LEGACY_TAG = "linux_x86_64"
RETAGGED_TAG = "manylinux_2_28_x86_64"

VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "pytorch-rocm"
TMP_DIR = VENDOR_DIR / "_downloads"

_INDEX_CACHE: dict[str, dict[str, str]] = {}   # norm pkg -> filename -> href


def norm(name: str) -> str:
    return name.lower().replace("_", "-")


def _get_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "prepare-rocm-wheels/2"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _index(pkg: str) -> dict[str, str]:
    """PEP 503 page for one package on the AMD index: encoded fname -> href.

    Keys keep the URL-encoded form from the listing page (e.g. `%2B` for
    `+`); building URLs from them is required, because S3 decodes a literal
    `+` in a path segment as a space and 404s.
    """
    pkg = norm(pkg)
    if pkg in _INDEX_CACHE:
        return _INDEX_CACHE[pkg]
    html = _get_bytes(AMD_INDEX + pkg + "/").decode()
    out: dict[str, str] = {}
    for href in re.findall(r'href="([^"#]+)"', html):
        fname = href.rsplit("/", 1)[-1]
        out[fname] = href
    _INDEX_CACHE[pkg] = out
    return out


def _on_amd_index(dep: str) -> bool:
    try:
        _index(dep)
        return True
    except Exception:
        return False


def _pick_wheel(pkg: str, version_spec: str | None) -> tuple[str, str] | None:
    """Pick a wheel for pkg on the AMD index (respecting == pin or newest).

    Accepts any linux_x86_64-tagged wheel (any Python/ABI tag; retagged
    later) and py3-none-any mirrors of generic deps (staged as-is).
    Prefers cp/cp linux builds over py3-none linux builds.
    Handles PEP 625 dist names: index filenames may use underscores while
    the normalized package name uses dashes.
    """
    cand: list[tuple[str, str, bool]] = []   # (version, fname, is_linux)
    for f in _index(pkg):
        parts = f[:-4].split("-")
        if len(parts) < 5:
            continue
        dist, ver = parts[0], parts[1]
        abi, plat = parts[-2], parts[-1]
        if norm(dist) != norm(pkg):
            continue
        if plat == LEGACY_TAG:                       # e.g. cp312-cp312-…, py3-none-…
            cand.append((urllib.parse.unquote(ver), f, True))
        elif abi == "none" and plat == "any":        # portable mirror
            cand.append((urllib.parse.unquote(ver), f, False))
    if not cand:
        return None
    files = [f for _, f, _ in cand]
    linux = {f for _, f, is_linux in cand if is_linux}

    def matched_version(f: str) -> str | None:
        for ver, fname, _ in cand:
            if fname == f:
                return ver
        return None

    if version_spec:
        spec = urllib.parse.unquote(version_spec)
        exact = [f for f in files if matched_version(f) == spec]
        if exact:
            files = exact
        else:
            print(f"  [warn] {pkg}=={version_spec} not found on AMD index; "
                  "falling back to newest")
    def vkey(f: str) -> tuple:
        ver = matched_version(f)
        head, *local = ver.split("+")
        nums = tuple(int(x) for x in re.findall(r"\d+", head)[:3])
        loc_nums = tuple(int(x) for x in re.findall(r"\d+", ".".join(local)) or (0,))
        return (nums, loc_nums)  # prefer local builds (+rocm…), higher first
    files.sort(key=vkey, reverse=True)
    # Preference: binary-for-cp312 > other-py linux builds > portable.
    def abi_rank(f: str) -> tuple[bool, bool]:
        parts = f[:-4].split("-")
        abi, plat = parts[-2], parts[-1]
        is_linux = plat == LEGACY_TAG
        cp312 = abi == PY_TAG and parts[-3] == PY_TAG
        return (cp312, is_linux)
    files_by_pref = sorted(files, key=abi_rank, reverse=True)
    chosen = files_by_pref[0]
    return chosen, _index(pkg)[chosen]


def _b64(sha: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(sha).rstrip(b"=").decode()


def _read_metadata_bytes(wheel_path: Path) -> bytes:
    zf = zipfile.ZipFile(wheel_path)
    try:
        meta_name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        return zf.read(meta_name)
    finally:
        zf.close()


def _resolve_store_url(claimed: str) -> str:
    """Resolve an index-path URL to its concrete store location.

    `/rocm/whl-next/...` is only a listing namespace: AMD's own wheels
    redirect into `/rocm/pytorch/whl-next/...`, while mirrors of generic
    packages redirect into `/rocm/core/whl-next/...`. urllib's opener can
    404 on these multi-hop redirects, so probe candidates with a 1-byte
    ranged GET and return the first that answers.
    """
    head, sep, tail = claimed.partition("/rocm/")
    if not sep:
        return claimed
    tails = [tail]
    if tail.startswith("whl-next/"):
        body = tail[len("whl-next/"):]
        tails = [f"whl-next/{body}", f"pytorch/whl-next/{body}", f"core/whl-next/{body}"]
    for t in tails:
        cand = head + "/rocm/" + t
        try:
            req = urllib.request.Request(
                cand, headers={"Range": "bytes=0-0", "User-Agent": "prepare-rocm-wheels/2"}
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                if 200 <= r.status < 300:
                    return cand
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            continue
    return claimed


def _synthesize_rocm(version: str, device_family: str) -> Path:
    """Build a deterministic stand-in for AMD's sdist-only ``rocm`` meta wheel.

    The real ``rocm`` package is published on the index only as an sdist whose
    ``setup.py`` inspects the *build machine's* GPU to pick its device deps —
    nondeterministic under ``uv lock/sync``. This creates an equivalent pure
    wheel that embeds the sdist's ``rocm_sdk`` python code and pins the
    dependency set for exactly one device family, so the whole closure is
    reproducible.
    """
    import tarfile

    out_name = f"rocm-{version}-py3-none-any.whl"
    out_path = VENDOR_DIR / out_name
    if out_path.exists():
        print(f"[skip] {out_name} already synthesized")
        return out_path

    sdist_url = _resolve_store_url(
        urllib.parse.urljoin(AMD_INDEX, f"rocm/rocm-{version}.tar.gz")
    )
    print(f"[get ] rocm {version} meta (sdist-only on the AMD index)")
    blob = _get_bytes(sdist_url)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        members = [
            m for m in tf.getmembers() if "/src/rocm_sdk/" in m.name and m.isfile()
        ]
        payload: dict[str, bytes] = {}
        for m in members:
            rel = m.name.split("/src/", 1)[1]
            payload[rel] = tf.extractfile(m).read()

    metadata_lines = [
        "Metadata-Version: 2.1",
        "Name: rocm",
        f"Version: {version}",
        f"Summary: AMD ROCm SDK meta package (deterministic vendor build for"
        f" {device_family})",
        "Requires-Python: >=3",
        f"Requires-Dist: rocm-sdk-core ~= {version}",
        f'Requires-Dist: rocm-sdk-libraries ~= {version}; extra == "libraries"',
        f'Requires-Dist: rocm-sdk-device-{device_family} ~= {version};'
        f' extra == "device-{device_family}"',
        "Provides-Extra: libraries",
        f"Provides-Extra: device-{device_family}",
        "",
        "Replaces AMD's sdist-only `rocm` meta package with a pinned, "
        "deterministic wheel generated by scripts/prepare_rocm_wheels.py.",
    ]
    _write_minimal_wheel(
        out_path,
        "rocm",
        version,
        payload=payload,
        metadata="\n".join(metadata_lines).encode(),
        entry_points="[console_scripts]\nrocm-sdk = rocm_sdk.__main__:main\n",
    )
    print(f"[ok  ] {out_name} (synthesized; sdist replaced)")
    return out_path


def _write_minimal_wheel(
    out_path: Path,
    dist_name: str,
    version: str,
    payload: dict[str, bytes],
    metadata: bytes,
    entry_points: str | None = None,
) -> None:
    dist_dir = (
        dist_name.replace("-", "_").replace(".", "_").replace("(", "")
        + "-"
        + version
        + ".dist-info"
    )
    files: dict[str, bytes] = {**payload, f"{dist_dir}/METADATA": metadata}
    files[f"{dist_dir}/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: prepare-rocm-wheels\n"
        "Root-Is-Purelib: true\nTag: py3-none-any\n"
    ).encode()
    if entry_points is not None:
        files[f"{dist_dir}/entry_points.txt"] = entry_points.encode()
    records = []
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in sorted(files.items()):
            zf.writestr(name, data)
            records.append(
                (name, _b64(hashlib.sha256(data).digest()), len(data))
            )
        record = io.StringIO()
        writer = csv.writer(record, lineterminator="\n")
        for row in sorted(records):
            writer.writerow(row)
        writer.writerow([f"{dist_dir}/RECORD", "", ""])
        zf.writestr(f"{dist_dir}/RECORD", record.getvalue())


def _process(pkg: str, version_spec: str, dry: bool) -> list[tuple[str, str]] | None:
    """Download one wheel, retag it (legacy tags only), stage it; return deps.

    Returns None when the AMD index has no usable wheel for the package —
    the caller then leaves it to PyPI.
    """
    picked = _pick_wheel(pkg, version_spec)
    if picked is None:
        return None
    fname, _href = picked
    # hrefs on the AMD index are bare filenames relative to the package dir;
    # wheels live in store trees the listing path redirects into — resolve
    # that (see _resolve_store_url) rather than hard-coding one.
    remote = _resolve_store_url(urllib.parse.urljoin(AMD_INDEX + norm(pkg) + "/", fname))
    needs_retag = fname.endswith(f"-{LEGACY_TAG}.whl")
    if needs_retag:
        out_name = fname[:-len(f"-{LEGACY_TAG}.whl")] + f"-{RETAGGED_TAG}.whl"
    else:
        out_name = fname
    out_name = out_name.replace("%2B", "+")
    out_path = VENDOR_DIR / out_name

    if out_path.exists():
        print(f"[skip] {out_name} already vendored")
        meta_bytes = _read_metadata_bytes(out_path)
    else:
        print(f"[get ] {out_name}")
        if dry:
            meta_bytes = _read_metadata_bytes_from_url(remote)
        else:
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            tmp = TMP_DIR / (out_name + ".dl")
            print(f"[dl  ] {remote}")
            with urllib.request.urlopen(remote, timeout=120) as r, open(tmp, "wb") as fdst:
                shutil.copyfileobj(r, fdst)
            if needs_retag:
                _retag(tmp, out_path)
                tmp.unlink()
            else:
                tmp.rename(out_path)  # py3-none-any: already portable
            print(f"[ok  ] {out_path.name} ({out_path.stat().st_size / 1e6:.2f} MB)")
            meta_bytes = _read_metadata_bytes(out_path)

    deps: list[tuple[str, str]] = []
    msg = email_parser.BytesParser().parsebytes(meta_bytes)
    for line in msg.get_all("Requires-Dist", []):
        head, _sep, marker = line.partition(";")
        # Skip deps gated on install extras (e.g. torch's `extra ==
        # "device-gfx1010"` for every GPU family); only the base graph is
        # followed. The extras this project needs are vendored via seeds.
        if marker and "extra" in marker:
            continue
        head = head.strip()
        name_req = re.split(r"[>\[<(~=; ]", head, maxsplit=1)[0].strip()
        if not name_req:
            continue
        m = re.search(r"==\s*([^\s;,]+)", head)
        ver = m.group(1) if m else None
        deps.append((name_req, ver))
    return deps


def _read_metadata_bytes_from_url(remote: str) -> bytes:
    blob = _get_bytes(remote)
    zf = zipfile.ZipFile(io.BytesIO(blob))
    try:
        meta_name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        return zf.read(meta_name)
    finally:
        zf.close()


def _retag(src: Path, dst: Path) -> None:
    """Stream-copy the wheel zip, patching WHEEL Tags and rewriting RECORD."""
    zf_src = zipfile.ZipFile(src)
    try:
        dist_dir = next(n.split("/")[0] for n in zf_src.namelist() if ".dist-info/" in n)
        wheel_meta = f"{dist_dir}/WHEEL"

        # Patch WHEEL in memory (already-patched digest recorded below).
        with zf_src.open(wheel_meta) as f:
            wheel_patched = (
                f.read().decode().replace(f"-{LEGACY_TAG}", f"-{RETAGGED_TAG}").encode()
            )

        hashes: dict[str, str] = {}
        zf_dst = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
        try:
            for info in zf_src.infolist():
                if info.filename.endswith("/RECORD"):  # rewritten at the end
                    continue
                if info.filename == wheel_meta:
                    body = wheel_patched
                    h = hashlib.sha256(body)
                    size = len(body)
                    zf_dst.writestr(info, body)
                else:
                    h = hashlib.sha256()
                    size = 0
                    with zf_src.open(info) as fsrc, zf_dst.open(info, "w") as fdst:
                        for chunk in iter(lambda: fsrc.read(1 << 20), b""):
                            size += len(chunk)
                            h.update(chunk)
                            fdst.write(chunk)
                hashes[info.filename] = (_b64(h.digest()), size)

            record_body = io.StringIO()
            writer = csv.writer(record_body, lineterminator="\n")
            for member, (digest, size) in sorted(hashes.items()):
                writer.writerow([member, digest, size])
            writer.writerow([f"{dist_dir}/RECORD", "", ""])
            zf_dst.writestr(f"{dist_dir}/RECORD", record_body.getvalue())
        finally:
            zf_dst.close()
    finally:
        zf_src.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="resolve closure, no downloads")
    args = ap.parse_args()

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    pending: list[tuple[str, str]] = []
    for pkg, ver in SEED_WHEELS:
        pending.append((pkg, ver))
    # The synthesized rocm meta (below) pins the SDK wheels explicitly,
    # covering torch's `rocm[libraries]~=…` base requirement.
    for sdk_pkg in ["rocm-sdk-core", "rocm-sdk-libraries",
                    "rocm-sdk-device-gfx1100", "rocm-bootstrap"]:
        pending.append((sdk_pkg, None))
    # torch's device extras are gated with `extra == "device-gfx1100"` and
    # not followed by the base-graph closure — vendor them explicitly, as
    # the official `pip install torch[device-gfx1100]` line does.
    for dev_ver in SEED_WHEELS:
        if dev_ver[0] == "torch":
            pending.append(("amd-torch-device-gfx1100", dev_ver[1]))
            pending.append(("amd-torch-device-gfx110x", dev_ver[1]))

    staged: list[str] = []
    while pending:
        pkg, ver = pending.pop(0)
        key = norm(pkg)
        if key in done:
            continue
        done.add(key)
        if not _on_amd_index(pkg):
            print(f"[pypi] {pkg} is not on the AMD index — left to PyPI")
            continue
        picked_deps = _process(pkg, ver, dry=args.plan)
        if picked_deps is None:
            print(f"[pypi] {pkg}: no usable wheel on the AMD index — left to PyPI")
            continue
        staged.append(pkg)
        for dep_name, dep_ver in picked_deps:
            if norm(dep_name) in done:
                continue
            pending.append((dep_name, dep_ver))

    print(f"\n{'plan' if args.plan else 'closure'} complete: {len(staged)} AMD packages staged in {VENDOR_DIR}")
    if args.plan:
        return 0
    _synthesize_rocm("10.1.0", "gfx1100")
    print("next: uv lock && uv sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
