"""Download the public CGMacros dataset from PhysioNet.

CGMacros is a published nutrition/CGM dataset (~627 MB zip). It is not stored
in git or Git LFS — fetch a local copy with::

    uv run download-cgmacros

By default meal photos (~630 MB uncompressed) are skipped on extract; pass
``--photos`` to keep them. Source: https://physionet.org/content/cgmacros/1.0.0/
(open S3 bucket ``physionet-open``, no PhysioNet login required).
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import typer
from eliot import start_action

PHYSIONET_PAGE: str = "https://physionet.org/content/cgmacros/1.0.0/"
S3_BASE: str = "https://physionet-open.s3.amazonaws.com/cgmacros/1.0.0"
ZIP_NAME: str = "CGMacros_dateshifted365.zip"
USER_AGENT: str = "sugar-sugar-cgmacros-downloader"

# SHA256SUMS.txt from PhysioNet cgmacros 1.0.0.
FILE_SHA256: Dict[str, str] = {
    "CGMacros_dateshifted365.zip": (
        "05c8b0e6f1a2757050aced55ce4bf6ab2ac9b30f2fd8ca193056812d9c621d4d"
    ),
    "DataDictionary.pdf": (
        "f161ff8df4bd95efa79d3daa8aba5eab754df287ab0f4dfc74bce156f6862c1a"
    ),
    "DataDictionary_Bio.csv": (
        "2d5b4c27873284ddbf218df51172b69ffa9caf6e97a3a122977b7a2761da4c61"
    ),
    "DataDictionary_CGMacros-00X.csv": (
        "db7bf114d61b90fc5de43fed5e5cd6ba33af37e900a706e1b5747d7b0ffac3f8"
    ),
    "DataDictionary_Gut_Health_Test.csv": (
        "fbc7555aacb93b2fb107b65db8d4a3d88d3546e7818fe5378fd2840cdceeb1b0"
    ),
    "DataDictionary_Microbes.csv": (
        "eb75be859489f59f6b81c3f8d29c307f078498655b4ae0878e1aa49a57d61f76"
    ),
    "LICENSE.txt": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
}

SIDECARS: Tuple[str, ...] = (
    "SHA256SUMS.txt",
    "LICENSE.txt",
    "DataDictionary.pdf",
    "DataDictionary_Bio.csv",
    "DataDictionary_CGMacros-00X.csv",
    "DataDictionary_Gut_Health_Test.csv",
    "DataDictionary_Microbes.csv",
)

_CHUNK_BYTES: int = 1024 * 1024


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_dest() -> Path:
    return project_root() / "data" / "cgmmacros"


def s3_url(filename: str) -> str:
    return f"{S3_BASE}/{filename}"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    digest = sha256_file(path)
    if digest != expected.lower():
        raise ValueError(
            f"SHA256 mismatch for {path.name}: expected {expected}, got {digest}"
        )


def dataset_is_present(dest: Path) -> bool:
    cgmacros = dest / "CGMacros"
    if not cgmacros.is_dir():
        return False
    return any(cgmacros.glob("CGMacros-*/CGMacros-*.csv"))


def is_photo_member(name: str) -> bool:
    lowered = name.replace("\\", "/").lower()
    if "/photos/" in lowered or lowered.startswith("photos/"):
        return True
    return lowered.endswith((".jpg", ".jpeg", ".png", ".heic", ".webp"))


def _safe_extract_path(root: Path, member_name: str) -> Path:
    parts = [
        part
        for part in member_name.replace("\\", "/").split("/")
        if part and part != "."
    ]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"unsafe zip path: {member_name}")
    target = (root.joinpath(*parts)).resolve()
    root_resolved = root.resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise ValueError(f"unsafe zip path: {member_name}")
    return target


def extract_zip(
    zip_path: Path,
    dest: Path,
    *,
    include_photos: bool = False,
) -> int:
    """Extract ``zip_path`` into ``dest``. Returns the number of files written."""
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = info.filename
            if not name or name.endswith("/") or info.is_dir():
                continue
            if not include_photos and is_photo_member(name):
                continue
            target = _safe_extract_path(dest, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as out:
                for chunk in iter(lambda: src.read(_CHUNK_BYTES), b""):
                    out.write(chunk)
            written += 1
    return written


def download_url(
    url: str,
    dest: Path,
    *,
    expected_sha256: Optional[str] = None,
    show_progress: bool = False,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    hasher = hashlib.sha256()
    with urllib.request.urlopen(request) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        read = 0
        while True:
            chunk = resp.read(_CHUNK_BYTES)
            if not chunk:
                break
            out.write(chunk)
            hasher.update(chunk)
            read += len(chunk)
            if show_progress and total:
                pct = 100.0 * read / total
                print(
                    f"\r{dest.name}: {read / 1_048_576:.1f}/"
                    f"{total / 1_048_576:.1f} MB ({pct:.0f}%)",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
        if show_progress and total:
            print(file=sys.stderr)
    digest = hasher.hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"SHA256 mismatch for {dest.name}: expected {expected_sha256}, got {digest}"
        )
    tmp.replace(dest)


def _file_is_current(path: Path, expected_sha256: Optional[str]) -> bool:
    if not path.is_file():
        return False
    if expected_sha256 is None:
        return True
    return sha256_file(path) == expected_sha256.lower()


def fetch_cgmacros(
    dest: Path,
    *,
    include_photos: bool = False,
    force: bool = False,
    keep_zip: bool = False,
    sidecar_names: Iterable[str] = SIDECARS,
) -> Path:
    """Download + extract CGMacros into ``dest``. Returns ``dest``."""
    dest = dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with start_action(
        action_type=u"download_cgmacros",
        dest=str(dest),
        include_photos=include_photos,
        force=force,
    ) as action:
        for name in sidecar_names:
            sidecar = dest / name
            expected = FILE_SHA256.get(name)
            if not force and _file_is_current(sidecar, expected):
                continue
            download_url(s3_url(name), sidecar, expected_sha256=expected)

        if dataset_is_present(dest) and not force:
            action.add_success_fields(skipped=True, reason="already_present")
            print(f"CGMacros already present at {dest} (use --force to re-download)")
            return dest

        zip_path = dest / ZIP_NAME
        zip_sha = FILE_SHA256[ZIP_NAME]
        if force or not _file_is_current(zip_path, zip_sha):
            print(f"Downloading {ZIP_NAME} from PhysioNet open S3 ({PHYSIONET_PAGE})")
            download_url(
                s3_url(ZIP_NAME),
                zip_path,
                expected_sha256=zip_sha,
                show_progress=True,
            )
        else:
            verify_sha256(zip_path, zip_sha)

        written = extract_zip(zip_path, dest, include_photos=include_photos)
        action.add_success_fields(extracted_files=written)
        print(f"Extracted {written} files into {dest}")
        if not keep_zip and zip_path.is_file():
            zip_path.unlink()
        return dest


def download_cgmacros(
    dest: Optional[Path] = typer.Option(
        None,
        "--dest",
        help="Extract directory (default: data/cgmmacros).",
    ),
    photos: bool = typer.Option(
        False,
        "--photos",
        help="Also extract meal photos (~630 MB extra on disk).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-download even if the dataset is already present.",
    ),
    keep_zip: bool = typer.Option(
        False,
        "--keep-zip",
        help="Keep the 627 MB zip after extracting.",
    ),
) -> None:
    """Download the public CGMacros dataset from PhysioNet into data/cgmmacros/."""
    target = dest if dest is not None else default_dest()
    fetch_cgmacros(
        target,
        include_photos=photos,
        force=force,
        keep_zip=keep_zip,
    )


def main() -> None:
    typer.run(download_cgmacros)
