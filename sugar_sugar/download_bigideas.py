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


def _mirror_urls(rel_path: str) -> tuple[str, ...]:
    """Mirrors for *rel_path*, best first.

    ``physionet.org/files`` leads: the open S3 bucket mirrors this dataset only
    up to 1.1.2, so every ``S3_BASE`` request for the pinned 1.1.3 is a 404.
    S3 stays as a fallback for the day PhysioNet publishes 1.1.3 there — the
    version is never mixed, both constants point at the same release.
    """
    return (files_url(rel_path), s3_url(rel_path))


def _download_optional(rel_path: str, dest: Path) -> bool:
    """Fetch *rel_path* from the first mirror that serves it.

    A per-mirror 404 is expected traffic, not news; only a file that no mirror
    serves is reported, so the log says how many files are missing rather than
    how many requests were made.
    """
    errors: list[str] = []
    for url in _mirror_urls(rel_path):
        try:
            download_url(url, dest, show_progress=True, user_agent=USER_AGENT)
            return True
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            dest.unlink(missing_ok=True)
    print(f"MISSING {dest.name} — no mirror served it:")
    for error in errors:
        print(f"    {error}")
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
        expected = 1 + 2 * len(SUBJECT_IDS)
        written = 0
        demo = dest / "Demographics.csv"
        if demo.is_file() and not force:
            written += 1
        elif _download_optional("Demographics.csv", demo):
            written += 1

        incomplete: list[str] = []
        for subject_id in SUBJECT_IDS:
            folder = f"{subject_id:03d}"
            subject_dir = dest / folder
            subject_dir.mkdir(parents=True, exist_ok=True)
            subject_files = 0
            for name in (f"Dexcom_{folder}.csv", f"Food_Log_{folder}.csv"):
                target = subject_dir / name
                if target.is_file() and not force:
                    subject_files += 1
                    continue
                if _download_optional(f"{folder}/{name}", target):
                    subject_files += 1
            written += subject_files
            # A subject is parseable only as a Dexcom export *plus* its food log.
            if subject_files < 2:
                incomplete.append(folder)

        complete_subjects = len(SUBJECT_IDS) - len(incomplete)
        action.add_success_fields(
            written_files=written,
            expected_files=expected,
            complete_subjects=complete_subjects,
            incomplete_subjects=incomplete,
        )
        print(
            f"Wrote {written}/{expected} files into {dest} "
            f"({complete_subjects}/{len(SUBJECT_IDS)} subjects complete)"
        )
        if not demo.is_file():
            print("WARNING: Demographics.csv is missing — subjects will have no age/gender.")
        if incomplete:
            raise FileNotFoundError(
                f"BIG IDEAs is incomplete in {dest}: subjects {', '.join(incomplete)} "
                "are missing a Dexcom export or a food log and cannot be played. "
                "Re-run with --force once the mirror recovers."
            )
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
