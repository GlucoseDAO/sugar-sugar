from typing import Any, Optional
from dash import html, dcc, Dash, Output, Input, State
import dash_bootstrap_components as dbc
import polars as pl
from datetime import datetime
import uuid
import csv
import shutil
from pathlib import Path
from eliot import start_action
from sugar_sugar.config import PREDICTION_HOUR_OFFSET, STORAGE_TYPE
from sugar_sugar.components.metrics import MetricsComponent
from sugar_sugar.data import load_glucose_data
from sugar_sugar.i18n import t, normalize_locale
from sugar_sugar.nickname import email_key, normalize_nickname

# Dataset used for the example/generic format and the format-C even rounds.
_EXAMPLE_DATASET_PATH: Path = Path("data/example.csv")

_MOBILE_UA_KEYWORDS: tuple[str, ...] = (
    'iphone', 'android', 'ipad', 'mobile', 'mobi', 'opera mini',
)


def _is_mobile_ua(ua: Optional[str]) -> bool:
    if not ua:
        return False
    lc = ua.lower()
    return any(keyword in lc for keyword in _MOBILE_UA_KEYWORDS)


def hidden_area_is_complete(df: pl.DataFrame) -> bool:
    """True when the hidden hour is drawn through to its last point.

    Same rule as the Submit button: the last ``PREDICTION_HOUR_OFFSET`` rows
    must include a non-zero prediction on the final slot (drawing interpolates
    the points in between). Used both to enable Submit and to decide whether
    Finish/Exit may persist the in-progress round.
    """
    if df.height < PREDICTION_HOUR_OFFSET:
        return False
    hidden = df.slice(df.height - PREDICTION_HOUR_OFFSET, PREDICTION_HOUR_OFFSET)
    predictions = hidden.get_column("prediction")
    nonzero = [i for i, value in enumerate(predictions.to_list()) if value != 0.0]
    if not nonzero:
        return False
    return max(nonzero) >= hidden.height - 1


# `SubmitComponent()` is also constructed per prediction-page render
# (create_prediction_layout), so the one-shot ranking migration is gated to run
# once per process rather than on every render -- it reads five CSVs.
_IDENTITY_BACKFILL_DONE: bool = False

# Suffix for the pristine copy taken before a ranking CSV is first converted.
# Gitignored via `*.pre-nickname.bak`.
_BACKUP_SUFFIX: str = ".pre-nickname.bak"


def _backup_before_conversion(path: Path) -> Optional[Path]:
    """Copy ``path`` aside once, before it is first converted.

    Deliberately **never overwrites** an existing backup: a later boot would copy
    already-converted content over it and destroy the only pristine record.  So the
    first run wins and the original survives every subsequent restart.
    """
    backup = path.with_name(path.name + _BACKUP_SUFFIX)
    if backup.exists():
        return None
    shutil.copy2(path, backup)
    return backup

class SubmitComponent(html.Div):
    def __init__(self, *, locale: str = "en") -> None:
        self._locale: str = normalize_locale(locale)
        self._stats_csv_path = (
            Path(__file__).resolve().parents[2]
            / 'data'
            / 'input'
            / 'prediction_statistics.csv'
        )
        # Overall ranking: written only after participant completed all eligible formats.
        self._ranking_csv_path = (
            Path(__file__).resolve().parents[2]
            / 'data'
            / 'input'
            / 'prediction_ranking.csv'
        )
        ranking_dir = self._stats_csv_path.parent
        self._ranking_by_format_paths: dict[str, Path] = {
            "A": ranking_dir / "prediction_ranking_A.csv",
            "B": ranking_dir / "prediction_ranking_B.csv",
            "C": ranking_dir / "prediction_ranking_C.csv",
        }
        self._stats_csv_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_csv_path = Path(__file__).resolve().parents[2] / 'prediction_statistics.csv'
        if legacy_csv_path.exists() and not self._stats_csv_path.exists():
            legacy_csv_path.replace(self._stats_csv_path)
        self._repair_misaligned_csv_rows()
        global _IDENTITY_BACKFILL_DONE
        if not _IDENTITY_BACKFILL_DONE:
            # Set first: a failure must surface loudly at boot, not retry per request.
            _IDENTITY_BACKFILL_DONE = True
            self.backfill_leaderboard_identity()
        super().__init__([
            html.Div(
                id="prediction-progress-label",
                children=t("ui.submit.progress_no_data", locale=self._locale),
                style={
                    'textAlign': 'center',
                    'marginBottom': '10px',
                    'fontSize': '16px',
                    'color': '#6c757d',
                    'fontStyle': 'italic'
                }
            ),
            html.Div(
                [
                    html.Button(
                        t("ui.common.finish_exit", locale=self._locale),
                        id="finish-study-button",
                        className="ui primary button finish-study-exit",
                        title=t("ui.common.finish_exit", locale=self._locale),
                        style={
                            'width': '48px',
                            'minWidth': '48px',
                            'fontSize': '18px',
                            'padding': '0',
                            'textAlign': 'center',
                            'display': 'inline-flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'lineHeight': '1',
                            'height': '48px',
                            'flexShrink': '0',
                        }
                    ),
                    html.Button(
                        t("ui.submit.submit", locale=self._locale),
                        id="submit-button",
                        className="ui green button",
                        disabled=True,  # Start disabled
                        style={
                            'width': '300px',
                            'fontSize': '25px',
                            'padding': '15px 0',
                            'textAlign': 'center',
                            'display': 'inline-flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'lineHeight': '1.2',
                            'height': '60px',
                        }
                    ),
                ],
                id="prediction-submit-row",
                disable_n_clicks=True,
                style={
                    'display': 'flex',
                    'flexDirection': 'row',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'gap': '10px',
                    'flexWrap': 'nowrap',
                },
            ),
            dcc.Store(id='prediction-stats-store', data=None, storage_type=STORAGE_TYPE)
        ], id="prediction-actions", style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center'})

    def _repair_misaligned_csv_rows(self) -> None:
        """Repair CSV rows whose columns were written in desired_fieldnames order instead of
        the file's actual header order.  This happened when run_id and format were inserted
        near the start of the dict but appended to files that already had run_id/format at
        the end of the header.

        General algorithm: if the file's header is H and the values were written in order D
        (desired_names), then corrupted_row[H[i]] = actual value of D[i].
        To recover: correct_value_for_header_col H[k] = corrupted_row[ H[ D.index(H[k]) ] ].
        """
        # The desired dict key order that was used when the bug was present.
        ranking_desired: list[str] = [
            'study_id', 'run_id', 'number', 'timestamp', 'format', 'rounds_played',
            'is_example_data', 'data_source_name', 'overall_mae_mgdl', 'overall_mse_mgdl',
            'overall_rmse_mgdl', 'overall_mape_pct',
        ]
        stats_desired: list[str] = [
            'study_id', 'run_id', 'number', 'timestamp', 'email', 'format',
            'is_example_data', 'data_source_name', 'age', 'user_id', 'gender',
            'uses_cgm', 'cgm_duration_years', 'diabetic', 'diabetic_type',
            'diabetes_duration', 'location', 'rounds_played', 'predicted_values',
            'real_values', 'prediction_times', 'overall_mae_mgdl', 'overall_mse_mgdl',
            'overall_rmse_mgdl', 'overall_mape_pct', 'per_round_metrics',
        ]
        self._repair_csv(
            self._ranking_csv_path,
            desired_order=ranking_desired,
            corrupt_check_col='overall_mae_mgdl',
            corrupt_check=lambda v: not self._is_numeric(v),
        )
        for path in self._ranking_by_format_paths.values():
            self._repair_csv(
                path,
                desired_order=ranking_desired,
                corrupt_check_col='overall_mae_mgdl',
                corrupt_check=lambda v: not self._is_numeric(v),
            )
        self._repair_csv(
            self._stats_csv_path,
            desired_order=stats_desired,
            corrupt_check_col='timestamp',
            corrupt_check=lambda v: self._is_integer_string(v),
        )

    @staticmethod
    def _is_numeric(value: str) -> bool:
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _is_integer_string(value: str) -> bool:
        try:
            int(str(value).strip())
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _repair_csv(
        path: Path,
        desired_order: list[str],
        corrupt_check_col: str,
        corrupt_check: Any,
    ) -> None:
        """Rewrite corrupted rows in-place using the known desired write order."""
        if not path.exists():
            return
        with path.open('r', newline='', encoding='utf-8', errors='replace') as fh:
            reader = csv.DictReader(fh)
            header = list(reader.fieldnames or [])
            rows = list(reader)
        if not header:
            return
        # Only attempt repair when the file already has the upgraded schema
        # (run_id and format present somewhere after the first column).
        if 'run_id' not in header or 'format' not in header:
            return

        # Build a lookup: desired column name → index in desired_order
        desired_index: dict[str, int] = {name: i for i, name in enumerate(desired_order)}

        repaired: list[dict[str, Any]] = []
        changed = False
        for row in rows:
            if not corrupt_check(row.get(corrupt_check_col, '')):
                repaired.append(row)
                continue

            # Recover correct values: correct_row[H[k]] = row[ H[ D.index(H[k]) ] ]
            fixed: dict[str, str] = {}
            for k, h_col in enumerate(header):
                d_idx = desired_index.get(h_col)
                if d_idx is not None and d_idx < len(header):
                    fixed[h_col] = row.get(header[d_idx], '')
                else:
                    fixed[h_col] = row.get(h_col, '')
            repaired.append(fixed)
            changed = True

        if not changed:
            return
        tmp_path = path.with_suffix('.tmp')
        with tmp_path.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            for row in repaired:
                writer.writerow({k: row.get(k, '') for k in header})
        tmp_path.replace(path)

    def _email_keys_by_study(self) -> dict[str, str]:
        """``study_id`` -> derived ``email_key``, read from the statistics CSV.

        `prediction_statistics.csv` has always stored the address, so this only
        derives the hash the ranking CSVs need -- no new information is written
        anywhere, and the address itself never moves.  When one study_id has
        several rows the most recent non-empty address wins, since that is the
        one a future run would hash.
        """
        if not self._stats_csv_path.exists():
            return {}
        with self._stats_csv_path.open('r', newline='', encoding='utf-8', errors='replace') as file_handle:
            reader = csv.DictReader(file_handle)
            fieldnames = reader.fieldnames or []
            if 'study_id' not in fieldnames or 'email' not in fieldnames:
                return {}
            keys: dict[str, str] = {}
            for row in reader:
                study_id = str(row.get('study_id') or '').strip()
                key = email_key(row.get('email'))
                if study_id and key:
                    keys[study_id] = key
        return keys

    def backfill_leaderboard_identity(self) -> int:
        """Convert pre-nickname ranking CSVs on first run. Idempotent.

        Two things happen:

        1. Every ranking CSV gains the ``email_key`` / ``nickname`` columns, so the
           schema is uniform from boot rather than only after the next finished game.
        2. Rows with a blank ``email_key`` get theirs derived from the address the
           statistics CSV already holds for that ``study_id``.

        (2) is what lets a player recognise their own history.  Board placement is
        arcade-style -- every finished game keeps its own slot and nothing is ever
        merged -- but a historical row carries only a ``study_id``, so on a new
        device (fresh localStorage, new study_id, same email) the player's own past
        scores would read as somebody else's: not highlighted, not counted as their
        placement, and invisible to the `/final` nickname suggestion.  Backfilling
        links them without moving, merging or removing a single slot.

        A row that already has an ``email_key`` is never relabelled, and a study_id
        with no recorded address keeps a blank key (its slots then belong to that
        session alone, exactly as an anonymous player's do).  Returns the number of
        rows stamped.
        """
        with start_action(action_type=u"backfill_leaderboard_identity") as action:
            email_keys = self._email_keys_by_study()
            stamped = 0
            upgraded_files = 0
            backups: list[str] = []
            for path in [self._ranking_csv_path, *self._ranking_by_format_paths.values()]:
                rows_changed, schema_changed, backup = self._backfill_identity_csv(
                    path, email_keys
                )
                stamped += rows_changed
                upgraded_files += int(schema_changed)
                if backup is not None:
                    backups.append(backup.name)
            action.log(
                message_type=u"ranking_identity_backfilled",
                known_study_ids=len(email_keys),
                rows_stamped=stamped,
                files_upgraded=upgraded_files,
                backups=backups,
            )
        return stamped

    @staticmethod
    def _backfill_identity_csv(
        path: Path, email_keys: dict[str, str]
    ) -> tuple[int, bool, Optional[Path]]:
        """Add the identity columns to one ranking CSV and fill blank keys.

        Returns ``(rows_stamped, schema_changed, backup_path)``.  Writes nothing when
        there is nothing to do, so this is safe to run on every boot -- and takes a
        pristine ``*.pre-nickname.bak`` copy before its first write, so a botched
        conversion can be undone by hand.
        """
        if not path.exists():
            return 0, False, None
        with path.open('r', newline='', encoding='utf-8', errors='replace') as file_handle:
            reader = csv.DictReader(file_handle)
            header = list(reader.fieldnames or [])
            rows = list(reader)
        if not header:
            return 0, False, None

        upgraded_header = header + [c for c in ('email_key', 'nickname') if c not in header]
        schema_changed = upgraded_header != header

        stamped = 0
        for row in rows:
            if str(row.get('email_key') or '').strip():
                continue  # already has an identity -- never relabel it
            key = email_keys.get(str(row.get('study_id') or '').strip(), '')
            if key:
                row['email_key'] = key
                stamped += 1

        if stamped == 0 and not schema_changed:
            return 0, False, None

        # Snapshot the original before the first conversion touches it.
        backup = _backup_before_conversion(path)

        tmp_path = path.with_suffix('.tmp')
        with tmp_path.open('w', newline='', encoding='utf-8') as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=upgraded_header)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, '') for column in upgraded_header})
        tmp_path.replace(path)
        return stamped, schema_changed, backup

    def set_study_nickname(self, *, study_id: str, key: str, nickname: str) -> int:
        """Stamp `nickname` onto this study's ranking rows only; backfill their `email_key`.

        Matches on ``study_id`` alone -- deliberately **never** on ``email_key`` -- so a
        returning player who picks a different name does not rewrite the rows of their
        earlier study entries.  One nickname per study; the previous one survives in the
        CSV and is what `stored_nickname` later offers as a suggestion.

        Needed because the `/final` box is edited *after* `save_statistics` has already
        written the run's rows.  Returns the number of rows changed.
        """
        cleaned = normalize_nickname(nickname)
        if not study_id:
            return 0

        changed_total = 0
        paths = [self._ranking_csv_path, *self._ranking_by_format_paths.values()]
        for path in paths:
            if not path.exists():
                continue
            with path.open('r', newline='', encoding='utf-8', errors='replace') as file_handle:
                reader = csv.DictReader(file_handle)
                header = list(reader.fieldnames or [])
                rows = list(reader)
            if not header:
                continue

            # Files written before nicknames existed lack the columns entirely.
            upgraded_header = header + [c for c in ('email_key', 'nickname') if c not in header]

            changed = 0
            for row in rows:
                if str(row.get('study_id') or '') != study_id:
                    continue
                row['nickname'] = cleaned
                if key and not str(row.get('email_key') or ''):
                    row['email_key'] = key
                changed += 1

            if changed == 0 and upgraded_header == header:
                continue

            tmp_path = path.with_suffix('.tmp')
            with tmp_path.open('w', newline='', encoding='utf-8') as out_handle:
                writer = csv.DictWriter(out_handle, fieldnames=upgraded_header)
                writer.writeheader()
                for row in rows:
                    writer.writerow({column: row.get(column, '') for column in upgraded_header})
            tmp_path.replace(path)
            changed_total += changed

        return changed_total

    def _get_next_number(self) -> int:
        """Get the next number for the prediction statistics."""
        csv_file_path = self._stats_csv_path
        if not csv_file_path.exists():
            return 0
        
        try:
            with csv_file_path.open('r', newline='', encoding='utf-8', errors='replace') as file_handle:
                reader = csv.DictReader(file_handle)
                numbers = [int(row['number']) for row in reader if row['number'].isdigit()]
                return max(numbers) + 1 if numbers else 0
        except Exception:
            return 0

    def save_statistics(self, user_info: dict[str, Any], *, write_ranking: bool = True) -> None:
        """Save prediction statistics to CSV file.

        This writes one statistics row per format run (``study_id`` + ``run_id``).
        Switching A→B must not erase the A row. Archived ``runs_by_format``
        runs are rewritten on later saves so a mid-session format switch still
        lands every field, even at 2 rounds. If `user_info["rounds"]` is
        present, the current run is aggregated across those rounds.

        The full dataset is NOT passed in (it no longer lives in a client store).
        Per-round prediction times come from `round_info['window_times']` captured
        at submit; for legacy rounds missing that, the dataset is reloaded
        server-side by the round's own identity and sliced. age/user_id come from
        `user_info` (age) and the fixed adapter default (user_id=1).

        ``write_ranking`` is True for Submit and for Finish from ``/ending``.
        Every submitted round is stored in ``prediction_statistics.csv``, one
        row per ``study_id`` + ``run_id`` (so format A/B/C runs do not overwrite
        each other). Ranking CSVs get the same rows for bookkeeping; the public
        board still hides runs below ``MIN_USEFUL_ROUNDS``. Finish/Exit from
        the chart stores the study row but skips ranking.
        """
        # Consent gate (defense-in-depth): never persist study data for a session
        # that has not completed mandatory consent. The display_page guard already
        # blocks unconsented navigation to the game, but enforcing it here -- at the
        # single write boundary -- also protects against a crafted client that
        # fabricates user_info and triggers a finish callback directly.
        if not user_info.get('consent_completed'):
            with start_action(
                action_type=u"save_statistics_skipped_no_consent",
                study_id=str(user_info.get('study_id') or ''),
            ):
                pass
            return

        csv_file_path = self._stats_csv_path
        
        rounds: list[dict[str, Any]] = user_info.get('rounds') or []
        parameters: list[dict[str, Any]] = []
        actual_values: list[dict[str, Any]] = []
        prediction_times: list[dict[str, Any]] = []
        version = str(user_info.get('run_format') or user_info.get('format') or '')

        # Stable ID across derived outputs (stats + ranking)
        study_id = user_info.get('study_id')
        if not study_id:
            study_id = str(uuid.uuid4())
            user_info['study_id'] = study_id

        metrics_component = MetricsComponent()

        def _metrics_from_table(table_data: list[dict[str, str]]) -> dict[str, Optional[float]]:
            metrics = metrics_component._calculate_metrics_from_table_data(table_data) if len(table_data) >= 2 else {}
            def _val(name: str) -> Optional[float]:
                item = metrics.get(name)
                if not item:
                    return None
                v = item.get('value')
                return float(v) if v is not None else None
            return {
                'mae': _val('MAE'),
                'mse': _val('MSE'),
                'rmse': _val('RMSE'),
                'mape': _val('MAPE'),
            }

        def _build_aggregate_table_data(rounds_in: list[dict[str, Any]]) -> list[dict[str, str]]:
            actual_row: dict[str, str] = {'metric': 'Actual Glucose'}
            pred_row: dict[str, str] = {'metric': 'Predicted'}
            out_idx = 0
            for round_info in rounds_in:
                table_data = round_info.get('prediction_table_data') or []
                if len(table_data) < 2:
                    continue
                round_actual = table_data[0]
                round_pred = table_data[1]
                i = 0
                while True:
                    key = f"t{i}"
                    if key not in round_actual or key not in round_pred:
                        break
                    actual_row[f"t{out_idx}"] = round_actual.get(key, "-")
                    pred_row[f"t{out_idx}"] = round_pred.get(key, "-")
                    out_idx += 1
                    i += 1
            return [actual_row, pred_row]

        def _time_list(window_df: pl.DataFrame) -> list[str]:
            time_col = window_df.get_column('time')
            if time_col.dtype == pl.String:
                return [str(t) for t in time_col.to_list()]
            return time_col.dt.strftime('%Y-%m-%d %H:%M:%S').to_list()

        def _resolve_round_times(round_info: dict[str, Any], window_size: int) -> list[str]:
            """Window times for a round: prefer the times captured at submit time
            (`window_times`); fall back to reloading the round's own dataset and
            slicing it (legacy rounds saved before window_times existed)."""
            stored = round_info.get('window_times')
            if stored:
                return [str(t) for t in stored]
            window_start = int(round_info.get('prediction_window_start') or 0)
            is_example = bool(round_info.get('is_example_data', user_info.get('is_example_data', True)))
            uploaded = user_info.get('uploaded_data_path')
            path = _EXAMPLE_DATASET_PATH if (is_example or not uploaded) else Path(str(uploaded))
            if not path.exists():
                return []
            glucose_df, _ = load_glucose_data(path)
            return _time_list(glucose_df.slice(max(0, window_start), window_size))

        # Per-round + overall metrics (computed in mg/dL, regardless of UI unit)
        per_round_metrics: list[dict[str, Any]] = []
        if rounds:
            for round_info in rounds:
                table_data = round_info.get('prediction_table_data') or []
                round_number = int(round_info.get('round_number') or (len(per_round_metrics) + 1))
                m = _metrics_from_table(table_data)
                per_round_metrics.append({
                    'round_number': round_number,
                    'mae': m['mae'],
                    'mse': m['mse'],
                    'rmse': m['rmse'],
                    'mape': m['mape'],
                })
        elif user_info.get('prediction_table_data'):
            table_data = user_info.get('prediction_table_data', []) or []
            m = _metrics_from_table(table_data)
            per_round_metrics.append({
                'round_number': 1,
                'mae': m['mae'],
                'mse': m['mse'],
                'rmse': m['rmse'],
                'mape': m['mape'],
            })

        overall_table_data = _build_aggregate_table_data(rounds) if rounds else (user_info.get('prediction_table_data', []) or [])
        overall = _metrics_from_table(overall_table_data)

        if rounds:
            # Aggregate across played rounds
            for round_idx, round_info in enumerate(rounds, start=1):
                table_data = round_info.get('prediction_table_data') or []
                if len(table_data) < 2:
                    continue

                window_size = int(round_info.get('prediction_window_size') or 0)
                if window_size <= 0:
                    continue

                actual_row = table_data[0]
                prediction_row = table_data[1]
                times = _resolve_round_times(round_info, window_size)

                for i in range(window_size):
                    time_key = f"t{i}"
                    pred_str = prediction_row.get(time_key, "-")
                    act_str = actual_row.get(time_key, "-")
                    if pred_str != "-" and act_str != "-" and i < len(times):
                        parameters.append({"version": version, "round": round_idx, "value": pred_str})
                        actual_values.append({"version": version, "round": round_idx, "value": act_str})
                        prediction_times.append({"version": version, "round": round_idx, "value": times[i]})
        else:
            # Backwards-compatible single-round behavior (still a single row)
            table_data = user_info.get('prediction_table_data', []) or []
            if len(table_data) >= 2:
                actual_row = table_data[0]
                prediction_row = table_data[1]
                single_size = int(user_info.get('prediction_window_size') or 0)
                times = _resolve_round_times(user_info, single_size) if single_size else [
                    str(t) for t in (user_info.get('window_times') or [])
                ]

                for i in range(max(single_size, len(times))):
                    time_key = f"t{i}"
                    pred_str = prediction_row.get(time_key, "-")
                    act_str = actual_row.get(time_key, "-")
                    if pred_str != "-" and act_str != "-" and i < len(times):
                        parameters.append({"version": version, "round": 1, "value": pred_str})
                        actual_values.append({"version": version, "round": 1, "value": act_str})
                        prediction_times.append({"version": version, "round": 1, "value": times[i]})

        # age comes from user_info; user_id matches the adapter default (data.py).
        age = int(user_info.get('age') or 0)
        user_id = 1
        
        # 0 = started the game but never submitted a round (still stored).
        # Legacy single-round sessions have table data but no `rounds` list.
        if rounds:
            rounds_played = len(rounds)
        elif user_info.get('prediction_table_data'):
            rounds_played = 1
        else:
            rounds_played = 0
        number = user_info.get("number")
        if number is None or (isinstance(number, str) and number.strip() == ""):
            number = self._get_next_number()
            user_info["number"] = number
        data = {
            'study_id': study_id,
            'run_id': str(user_info.get('run_id') or ''),
            'number': number,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'email': user_info.get('email', ''),
            'format': version,
            'is_example_data': bool(user_info.get('is_example_data', True)),
            'data_source_name': str(user_info.get('data_source_name', 'example.csv')),
            'age': age,
            'user_id': user_id,
            'gender': user_info.get('gender', ''),
            'uses_cgm': bool(user_info.get('uses_cgm', False)),
            'cgm_duration_years': user_info.get('cgm_duration_years', ''),
            'diabetic': user_info.get('diabetic', ''),
            'diabetic_type': user_info.get('diabetic_type', ''),
            'diabetes_duration': user_info.get('diabetes_duration', ''),
            'location': user_info.get('location', ''),
            'rounds_played': rounds_played,
            # Clear naming: "real" == ground truth, "predicted" == user prediction
            'predicted_values': str(parameters),
            'real_values': str(actual_values),
            'prediction_times': str(prediction_times),
            # Metrics
            'overall_mae_mgdl': overall['mae'],
            'overall_mse_mgdl': overall['mse'],
            'overall_rmse_mgdl': overall['rmse'],
            'overall_mape_pct': overall['mape'],
            'per_round_metrics': str(per_round_metrics),
        }
        
        def _upgrade_and_upsert_csv(
            path: Path,
            row: dict[str, Any],
            legacy_to_new: dict[str, str],
            match_keys: tuple[str, ...] = ("study_id",),
        ) -> None:
            """Insert or replace the row matching all ``match_keys``.

            Incremental saves (Start, each submit, Exit) of the *same* run
            must not duplicate. Different format runs of the same player
            (A then B) must not overwrite each other — match ``study_id`` +
            ``run_id`` for those files.
            """
            desired_fieldnames = list(row.keys())
            match_values = tuple(str(row.get(key, "") or "") for key in match_keys)
            can_match = all(match_values)
            existing_fieldnames: list[str] = []
            existing_rows: list[dict[str, Any]] = []
            if path.exists():
                with path.open("r", newline="", encoding="utf-8", errors="replace") as file_handle:
                    reader = csv.DictReader(file_handle)
                    existing_fieldnames = list(reader.fieldnames or [])
                    existing_rows = list(reader)

            needs_upgrade = (
                any(field in existing_fieldnames for field in legacy_to_new.keys())
                or any(field not in existing_fieldnames for field in desired_fieldnames)
                or not existing_fieldnames
            )
            if needs_upgrade:
                preserved_existing = [f for f in existing_fieldnames if f and f not in legacy_to_new.keys()]
                fieldnames: list[str] = []
                for name in preserved_existing + desired_fieldnames:
                    if name not in fieldnames:
                        fieldnames.append(name)
            else:
                fieldnames = list(existing_fieldnames)

            out_rows: list[dict[str, Any]] = []
            replaced = False
            for old_row in existing_rows:
                upgraded_row: dict[str, Any] = {key: old_row.get(key, "") for key in fieldnames}
                for old_key, new_key in legacy_to_new.items():
                    if upgraded_row.get(new_key, "") in ("", None) and old_key in old_row:
                        upgraded_row[new_key] = old_row.get(old_key, "")
                old_values = tuple(str(upgraded_row.get(key, "") or "") for key in match_keys)
                if can_match and old_values == match_values:
                    out_rows.append({key: row.get(key, upgraded_row.get(key, "")) for key in fieldnames})
                    replaced = True
                else:
                    out_rows.append(upgraded_row)
            if not replaced:
                out_rows.append({key: row.get(key, "") for key in fieldnames})

            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with tmp_path.open("w", newline="", encoding="utf-8") as out_handle:
                writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
                writer.writeheader()
                for out_row in out_rows:
                    writer.writerow(out_row)
            tmp_path.replace(path)

        def _match_keys_for(row: dict[str, Any]) -> tuple[str, ...]:
            if str(row.get("run_id") or "").strip():
                return ("study_id", "run_id")
            return ("study_id", "format")

        _stats_legacy = {
            'id': 'study_id',
            'parameters': 'predicted_values',
            'actual_values': 'real_values',
            'prediction_time': 'prediction_times',
        }

        def _stats_row_for_run(
            *,
            run_format: str,
            run_id: str,
            run_rounds: list[dict[str, Any]],
            is_example: bool,
            source_name: str,
        ) -> tuple[dict[str, Any], dict[str, Optional[float]], int]:
            """Build a statistics row for one archived (or current) format run."""
            parameters_run: list[dict[str, Any]] = []
            actual_values_run: list[dict[str, Any]] = []
            prediction_times_run: list[dict[str, Any]] = []
            per_round: list[dict[str, Any]] = []
            for round_idx, round_info in enumerate(run_rounds, start=1):
                table_data = round_info.get('prediction_table_data') or []
                m = _metrics_from_table(table_data)
                per_round.append({
                    'round_number': int(round_info.get('round_number') or round_idx),
                    'mae': m['mae'],
                    'mse': m['mse'],
                    'rmse': m['rmse'],
                    'mape': m['mape'],
                })
                if len(table_data) < 2:
                    continue
                window_size = int(round_info.get('prediction_window_size') or 0)
                if window_size <= 0:
                    continue
                actual_row = table_data[0]
                prediction_row = table_data[1]
                times = _resolve_round_times(round_info, window_size)
                for i in range(window_size):
                    time_key = f"t{i}"
                    pred_str = prediction_row.get(time_key, "-")
                    act_str = actual_row.get(time_key, "-")
                    if pred_str != "-" and act_str != "-" and i < len(times):
                        parameters_run.append({"version": run_format, "round": round_idx, "value": pred_str})
                        actual_values_run.append({"version": run_format, "round": round_idx, "value": act_str})
                        prediction_times_run.append({"version": run_format, "round": round_idx, "value": times[i]})
            overall_m = _metrics_from_table(
                _build_aggregate_table_data(run_rounds) if run_rounds else []
            )
            rounds_n = len(run_rounds)
            row = {
                'study_id': study_id,
                'run_id': run_id,
                'number': number,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'email': user_info.get('email', ''),
                'format': run_format,
                'is_example_data': is_example,
                'data_source_name': source_name,
                'age': age,
                'user_id': user_id,
                'gender': user_info.get('gender', ''),
                'uses_cgm': bool(user_info.get('uses_cgm', False)),
                'cgm_duration_years': user_info.get('cgm_duration_years', ''),
                'diabetic': user_info.get('diabetic', ''),
                'diabetic_type': user_info.get('diabetic_type', ''),
                'diabetes_duration': user_info.get('diabetes_duration', ''),
                'location': user_info.get('location', ''),
                'rounds_played': rounds_n,
                'predicted_values': str(parameters_run),
                'real_values': str(actual_values_run),
                'prediction_times': str(prediction_times_run),
                'overall_mae_mgdl': overall_m['mae'],
                'overall_mse_mgdl': overall_m['mse'],
                'overall_rmse_mgdl': overall_m['rmse'],
                'overall_mape_pct': overall_m['mape'],
                'per_round_metrics': str(per_round),
            }
            return row, overall_m, rounds_n

        # Archived format runs first, then the current run, so a later save of
        # the same format (same run_id) wins. Different run_ids stay as rows.
        runs_by_format: dict[str, list[dict[str, Any]]] = dict(user_info.get("runs_by_format") or {})
        archived_payloads: list[tuple[dict[str, Any], dict[str, Optional[float]], int]] = []
        for fmt, runs in runs_by_format.items():
            for run in runs:
                archived_rounds = list(run.get("rounds") or [])
                if not archived_rounds:
                    continue
                archive_row, archive_overall, archive_n = _stats_row_for_run(
                    run_format=str(run.get("format") or fmt),
                    run_id=str(run.get("active_run_id") or run.get("run_id") or ""),
                    run_rounds=archived_rounds,
                    is_example=bool(run.get("is_example_data", True)),
                    source_name=str(run.get("data_source_name") or ""),
                )
                archived_payloads.append((archive_row, archive_overall, archive_n))
                _upgrade_and_upsert_csv(
                    csv_file_path,
                    archive_row,
                    legacy_to_new=_stats_legacy,
                    match_keys=_match_keys_for(archive_row),
                )

        _upgrade_and_upsert_csv(
            csv_file_path,
            data,
            legacy_to_new=_stats_legacy,
            match_keys=_match_keys_for(data),
        )

        # Write ranking row for fast leaderboard lookups.
        # `email_key` is a one-way hash used only to merge one player's rows across
        # devices; `nickname` is an optional public display label. Neither the address
        # nor the nickname belongs to the study record -- see sugar_sugar/nickname.py.
        leaderboard_identity = email_key(user_info.get('email'))
        leaderboard_nickname = normalize_nickname(user_info.get('nickname'))
        ranking_row = {
            'study_id': study_id,
            'run_id': str(user_info.get('run_id') or ''),
            'number': data['number'],
            'timestamp': data['timestamp'],
            'email_key': leaderboard_identity,
            'nickname': leaderboard_nickname,
            'format': data['format'],
            'rounds_played': rounds_played,
            'is_example_data': data['is_example_data'],
            'data_source_name': data['data_source_name'],
            'overall_mae_mgdl': overall['mae'],
            'overall_mse_mgdl': overall['mse'],
            'overall_rmse_mgdl': overall['rmse'],
            'overall_mape_pct': overall['mape'],
        }
        # Chart Exit skips ranking (write_ranking=False). Start-only stubs
        # (0 rounds) stay in statistics. Every submitted round is written
        # here for bookkeeping; the public board still hides short runs.
        if not write_ranking:
            return

        def _ranking_from_stats(
            stats_row: dict[str, Any],
            metrics: dict[str, Optional[float]],
        ) -> dict[str, Any]:
            return {
                'study_id': study_id,
                'run_id': str(stats_row.get('run_id') or ''),
                'number': data['number'],
                'timestamp': stats_row.get('timestamp', data['timestamp']),
                'email_key': leaderboard_identity,
                'nickname': leaderboard_nickname,
                'format': stats_row.get('format', ''),
                'rounds_played': stats_row.get('rounds_played', 0),
                'is_example_data': stats_row.get('is_example_data', True),
                'data_source_name': stats_row.get('data_source_name', ''),
                'overall_mae_mgdl': metrics['mae'],
                'overall_mse_mgdl': metrics['mse'],
                'overall_rmse_mgdl': metrics['rmse'],
                'overall_mape_pct': metrics['mape'],
            }

        # Per-format ranking: current run plus every archived format run.
        if version in self._ranking_by_format_paths and rounds_played >= 1:
            _upgrade_and_upsert_csv(
                self._ranking_by_format_paths[version],
                ranking_row,
                legacy_to_new={},
                match_keys=_match_keys_for(ranking_row),
            )
        for archive_row, archive_overall, archive_n in archived_payloads:
            archive_fmt = str(archive_row.get('format') or '')
            if archive_fmt not in self._ranking_by_format_paths or archive_n < 1:
                continue
            archive_ranking = _ranking_from_stats(archive_row, archive_overall)
            _upgrade_and_upsert_csv(
                self._ranking_by_format_paths[archive_fmt],
                archive_ranking,
                legacy_to_new={},
                match_keys=_match_keys_for(archive_ranking),
            )

        # Overall (cumulative) ranking across formats played so far.
        played_formats: set[str] = {str(fmt) for fmt, runs in runs_by_format.items() if runs}
        if version:
            played_formats.add(version)
        all_rounds: list[dict[str, Any]] = []
        for fmt in sorted(played_formats):
            for run in (runs_by_format.get(fmt) or []):
                all_rounds.extend(list(run.get("rounds") or []))
        all_rounds.extend(rounds if rounds else [])

        overall_all_table = _build_aggregate_table_data(all_rounds) if all_rounds else overall_table_data
        overall_all = _metrics_from_table(overall_all_table)

        source_names: set[str] = set()
        any_uploaded = False
        for fmt in played_formats:
            for run in (runs_by_format.get(fmt) or []):
                any_uploaded = any_uploaded or (not bool(run.get("is_example_data", True)))
                name = str(run.get("data_source_name") or "")
                if name:
                    source_names.add(name)
        any_uploaded = any_uploaded or (not bool(user_info.get("is_example_data", True)))
        current_source = str(user_info.get("data_source_name") or "")
        if current_source:
            source_names.add(current_source)

        data_source_name = next(iter(source_names)) if len(source_names) == 1 else "multiple"
        total_rounds_played = int(
            sum(len(list(run.get("rounds") or [])) for fmt in played_formats for run in (runs_by_format.get(fmt) or []))
            + rounds_played
        )

        if total_rounds_played < 1:
            return

        overall_ranking_row = {
            'study_id': study_id,
            'run_id': str(user_info.get('run_id') or uuid.uuid4()),
            'number': data['number'],
            'timestamp': data['timestamp'],
            'email_key': leaderboard_identity,
            'nickname': leaderboard_nickname,
            'format': "ALL",
            'rounds_played': total_rounds_played,
            'is_example_data': (not any_uploaded),
            'data_source_name': data_source_name,
            'overall_mae_mgdl': overall_all['mae'],
            'overall_mse_mgdl': overall_all['mse'],
            'overall_rmse_mgdl': overall_all['rmse'],
            'overall_mape_pct': overall_all['mape'],
        }
        _upgrade_and_upsert_csv(
            self._ranking_csv_path,
            overall_ranking_row,
            legacy_to_new={},
            match_keys=("study_id",),
        )

    def register_callbacks(self, app: Dash) -> None:
        """Register callbacks for the submit component"""
        
        @app.callback(
            [Output('submit-button', 'disabled'),
             Output('submit-button', 'children'),
             Output('submit-button', 'style'),
             Output('prediction-progress-label', 'children'),
             Output('prediction-progress-label', 'style')],
            [Input('current-window-df', 'data'),
             Input('interface-language', 'data'),
             Input('user-agent', 'data')],
            prevent_initial_call=False
        )
        def update_submit_button_state(
            df_data: Optional[dict[str, Any]],
            interface_language: Optional[str],
            user_agent: Optional[str],
        ) -> tuple[bool, str, dict[str, Any], str, dict[str, Any]]:
            """Enable submit button only when there are predictions to the end of the hidden area"""
            locale = normalize_locale(interface_language)
            ready_text = (
                f"✓ {t('ui.submit.submit', locale=locale)}"
                if _is_mobile_ua(user_agent)
                else t("ui.submit.progress_ready", locale=locale)
            )
            base_style = {
                'width': '300px',
                'fontSize': '25px',
                'padding': '15px 0',
                'textAlign': 'center',
                'display': 'inline-flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'lineHeight': '1.2',
                'height': '60px',
            }
            
            base_label_style = {
                'textAlign': 'center',
                'marginBottom': '10px',
                'fontSize': '16px',
                'fontStyle': 'italic'
            }
            
            if not df_data:
                disabled_style = {**base_style, 'backgroundColor': '#cccccc', 'color': '#666666', 'cursor': 'not-allowed'}
                label_style = {**base_label_style, 'color': '#6c757d'}
                return True, t("ui.submit.submit", locale=locale), disabled_style, t("ui.submit.progress_no_data", locale=locale), label_style
            
            # Reconstruct DataFrame to check for predictions
            df = self._reconstruct_dataframe_from_dict(df_data)
            
            # Check predictions in the hidden area (last PREDICTION_HOUR_OFFSET points)
            visible_points = len(df) - PREDICTION_HOUR_OFFSET
            hidden_area_df = df.slice(visible_points, PREDICTION_HOUR_OFFSET)
            
            # Find the last time point with a prediction
            predictions_mask = hidden_area_df.get_column("prediction") != 0.0
            if predictions_mask.any():
                # Get indices of predictions in hidden area
                prediction_indices = [i for i, has_pred in enumerate(predictions_mask) if has_pred]
                last_prediction_idx = max(prediction_indices)
                total_hidden_points = len(hidden_area_df)
                
                predictions_to_end = hidden_area_is_complete(df)
                
                # Check if first point is auto-snapped to ground truth
                first_point_is_snapped = False
                if len(prediction_indices) > 0 and prediction_indices[0] == 0:
                    # First prediction point exists - check if it matches ground truth (auto-snapped)
                    first_prediction_value = hidden_area_df.get_column("prediction")[0]
                    first_ground_truth_value = hidden_area_df.get_column("gl")[0]
                    # Allow small floating point tolerance
                    if abs(first_prediction_value - first_ground_truth_value) < 0.01:
                        first_point_is_snapped = True
                
                # Calculate user-made predictions (excluding auto-snapped first point)
                user_predictions_count = len(prediction_indices)
                if first_point_is_snapped:
                    user_predictions_count -= 1
                
                # Required predictions is always the full hidden area
                required_user_predictions = total_hidden_points
                
                # Debug output
                print(f"DEBUG: Prediction count - total_hidden_points: {total_hidden_points}, prediction_indices: {prediction_indices}")
                print(f"DEBUG: first_point_is_snapped: {first_point_is_snapped}, user_predictions_count: {user_predictions_count}, required: {required_user_predictions}")
                
                if predictions_to_end:
                    enabled_style = {**base_style, 'backgroundColor': '#4CBB17', 'color': 'white', 'cursor': 'pointer'}
                    label_style = {**base_label_style, 'display': 'none'}
                    return False, ready_text, enabled_style, "", label_style
                else:
                    disabled_style = {**base_style, 'backgroundColor': '#999999', 'color': 'white', 'cursor': 'not-allowed'}
                    label_style = {**base_label_style, 'color': '#6c757d'}
                    status_text = t(
                        "ui.submit.progress_some",
                        locale=locale,
                        done=user_predictions_count,
                        total=required_user_predictions,
                    )
                    return True, t("ui.submit.submit", locale=locale), disabled_style, status_text, label_style
            else:
                disabled_style = {**base_style, 'backgroundColor': '#cccccc', 'color': '#666666', 'cursor': 'not-allowed'}
                label_style = {**base_label_style, 'color': '#6c757d'}
                return True, t("ui.submit.submit", locale=locale), disabled_style, t("ui.submit.progress_hidden_area", locale=locale), label_style

    def _reconstruct_dataframe_from_dict(self, df_data: dict[str, list[Any]]) -> pl.DataFrame:
        """Reconstruct a Polars DataFrame from stored dictionary data"""
        return pl.DataFrame({
            'time': pl.Series(df_data['time']).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
            'gl': pl.Series(df_data['gl'], dtype=pl.Float64),
            'prediction': pl.Series(df_data['prediction'], dtype=pl.Float64),
            'age': pl.Series([int(float(x)) for x in df_data['age']], dtype=pl.Int64),
            'user_id': pl.Series([int(float(x)) for x in df_data['user_id']], dtype=pl.Int64)
        }) 