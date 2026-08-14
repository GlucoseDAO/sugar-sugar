"""Download the public D1NAMO (Dubosson) T1D subset from Zenodo.

D1NAMO is not a vendor CGM export and is not stored in git. Fetch a local copy
with::

    uv run download-d1namo

Only the diabetes pictures/glucose/food/insulin zip is fetched (~252 MB). The
ECG and chest-belt zips (several GB) are skipped on purpose.

Source: https://zenodo.org/records/5651217
Paper: Dubosson et al., Informatics in Medicine Unlocked, 2018.
License: CC BY-SA 4.0.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import typer
from eliot import start_action

from sugar_sugar.download_cgmacros import download_url, extract_zip

ZENODO_PAGE: str = "https://zenodo.org/records/5651217"
ZENODO_RECORD: str = "5651217"
ZIP_NAME: str = "diabetes_subset_pictures-glucose-food-insulin.zip"
ZIP_URL: str = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{ZIP_NAME}/content"
)
ZIP_MD5: str = "104a2810050e2c6a6be698f683d862b5"
USER_AGENT: str = "sugar-sugar-d1namo-downloader"
_CHUNK_BYTES: int = 1024 * 1024


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_dest() -> Path:
    return project_root() / "data" / "d1namo"


def dataset_is_present(dest: Path) -> bool:
    if not dest.is_dir():
        return False
    return any(path.name.lower() == "glucose.csv" for path in dest.rglob("glucose.csv"))


def md5_file(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_md5(path: Path, expected: str) -> None:
    digest = md5_file(path)
    if digest != expected.lower():
        raise ValueError(
            f"MD5 mismatch for {path.name}: expected {expected}, got {digest}"
        )


def fetch_d1namo(
    dest: Path,
    *,
    include_photos: bool = True,
    force: bool = False,
    keep_zip: bool = False,
) -> Path:
    """Download + extract the D1NAMO T1D subset into ``dest``. Returns ``dest``."""
    dest = dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with start_action(
        action_type=u"download_d1namo",
        dest=str(dest),
        include_photos=include_photos,
        force=force,
    ) as action:
        if dataset_is_present(dest) and not force:
            action.add_success_fields(skipped=True, reason="already_present")
            print(f"D1NAMO already present at {dest} (use --force to re-download)")
            return dest

        zip_path = dest / ZIP_NAME
        if force or not zip_path.is_file() or md5_file(zip_path) != ZIP_MD5:
            print(f"Downloading {ZIP_NAME} from Zenodo ({ZENODO_PAGE})")
            download_url(
                ZIP_URL,
                zip_path,
                show_progress=True,
                user_agent=USER_AGENT,
            )
        verify_md5(zip_path, ZIP_MD5)

        written = extract_zip(zip_path, dest, include_photos=include_photos)
        action.add_success_fields(extracted_files=written)
        print(f"Extracted {written} files into {dest}")
        if not keep_zip and zip_path.is_file():
            zip_path.unlink()
        return dest


def download_d1namo(
    dest: Optional[Path] = typer.Option(
        None,
        "--dest",
        help="Extract directory (default: data/d1namo).",
    ),
    no_photos: bool = typer.Option(
        False,
        "--no-photos",
        help="Skip meal JPEGs and keep only glucose/insulin/food CSVs.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-download even if the dataset is already present.",
    ),
    keep_zip: bool = typer.Option(
        False,
        "--keep-zip",
        help="Keep the 252 MB zip after extracting.",
    ),
) -> None:
    """Download the public D1NAMO T1D subset from Zenodo into data/d1namo/."""
    target = dest if dest is not None else default_dest()
    fetch_d1namo(
        target,
        include_photos=not no_photos,
        force=force,
        keep_zip=keep_zip,
    )


def main() -> None:
    typer.run(download_d1namo)
