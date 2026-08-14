"""Download the public BIG IDEAs Dexcom + food-log subset from PhysioNet.

The full PhysioNet zip is ~4.7 GB because it includes Empatica ACC/BVP streams.
This downloader fetches only Demographics.csv, Dexcom_NNN.csv, and
Food_Log_NNN.csv (~3 MB). Empatica files are never downloaded.

Source: https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/
Paper: Bent et al., npj Digital Medicine, 2021.
License: ODC-By 1.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from eliot import start_action

from sugar_sugar.download_cgmacros import download_url

PHYSIONET_PAGE: str = "https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/"
S3_BASE: str = (
    "https://physionet-open.s3.amazonaws.com/big-ideas-glycemic-wearable/1.1.3"
)
FILES_BASE: str = "https://physionet.org/files/big-ideas-glycemic-wearable/1.1.3"
USER_AGENT: str = "sugar-sugar-bigideas-downloader"
SUBJECT_IDS: tuple[int, ...] = tuple(range(1, 17))


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_dest() -> Path:
    return project_root() / "data" / "bigideas"


def dataset_is_present(dest: Path) -> bool:
    if not dest.is_dir():
        return False
    return any(path.name.lower().startswith("dexcom_") for path in dest.rglob("Dexcom_*.csv"))


def s3_url(rel_path: str) -> str:
    return f"{S3_BASE}/{rel_path}"


def files_url(rel_path: str) -> str:
    return f"{FILES_BASE}/{rel_path}"


def _download_optional(url: str, dest: Path) -> bool:
    try:
        download_url(url, dest, show_progress=True, user_agent=USER_AGENT)
        return True
    except Exception as exc:
        print(f"Skip {dest.name}: {exc}")
        dest.unlink(missing_ok=True)
        return False


def fetch_bigideas(dest: Path, *, force: bool = False) -> Path:
    """Download Dexcom + food logs into ``dest``. Returns ``dest``."""
    dest = dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with start_action(action_type=u"download_bigideas", dest=str(dest), force=force) as action:
        if dataset_is_present(dest) and not force:
            action.add_success_fields(skipped=True, reason="already_present")
            print(f"BIG IDEAs already present at {dest} (use --force to re-download)")
            return dest

        print(f"Downloading BIG IDEAs Dexcom + food logs from PhysioNet ({PHYSIONET_PAGE})")
        written = 0
        demo = dest / "Demographics.csv"
        if force or not demo.is_file():
            if _download_optional(s3_url("Demographics.csv"), demo) or _download_optional(
                files_url("Demographics.csv"), demo
            ):
                written += 1

        for subject_id in SUBJECT_IDS:
            folder = f"{subject_id:03d}"
            subject_dir = dest / folder
            subject_dir.mkdir(parents=True, exist_ok=True)
            for name in (f"Dexcom_{folder}.csv", f"Food_Log_{folder}.csv"):
                target = subject_dir / name
                if target.is_file() and not force:
                    written += 1
                    continue
                rel = f"{folder}/{name}"
                if _download_optional(s3_url(rel), target) or _download_optional(
                    files_url(rel), target
                ):
                    written += 1

        action.add_success_fields(written_files=written)
        print(f"Wrote {written} files into {dest}")
        if not dataset_is_present(dest):
            raise FileNotFoundError(
                f"BIG IDEAs Dexcom tables were not downloaded into {dest}"
            )
        return dest


def download_bigideas(
    dest: Optional[Path] = typer.Option(
        None,
        "--dest",
        help="Extract directory (default: data/bigideas).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-download even if the dataset is already present.",
    ),
) -> None:
    """Download BIG IDEAs Dexcom + food logs from PhysioNet into data/bigideas/."""
    target = dest if dest is not None else default_dest()
    fetch_bigideas(target, force=force)


def main() -> None:
    typer.run(download_bigideas)
