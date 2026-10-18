"""Dedicated Data Analyst Agent.

The Planner supplies an approved task. This role discovers and profiles only the
approved datasets, uses the shared read-only SQL tools, checks returned results,
and emits reproducible evidence. It never mutates source data.
"""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable

from analysis_methods import AnalysisMethodError, welch_tests_from_summaries
from planner import PlannerTask


class AnalystError(RuntimeError):
    """The Data Analyst could not safely complete its approved task."""


AnalysisStepGenerator = Callable[[str, PlannerTask, dict[str, Any]], dict[str, Any]]
QueryExecutor = Callable[[str, PlannerTask], dict[str, Any]]

MAX_ANALYSIS_QUERIES = 8
MAX_QUERY_REPAIRS = 2
MAX_ANALYSIS_ACTIONS = MAX_ANALYSIS_QUERIES * (MAX_QUERY_REPAIRS + 1) + 1
MAX_RAG_RETRIEVALS = 3
MAX_CLEAN_ROWS = 50000
MAX_CLEAN_CSV_BYTES = 15_000_000


def _sql_words_with_depth(sql: str) -> tuple[list[tuple[str, int, int, int]], list[tuple[int, int]]]:
    """Tokenize enough SQLite syntax to safely relocate a misplaced compound ORDER BY."""
    words: list[tuple[str, int, int, int]] = []
    closes: list[tuple[int, int]] = []
    depth = 0
    i = 0
    while i < len(sql):
        ch = sql[i]
        if ch == "'" or ch == '"' or ch == "`":
            quote = ch
            i += 1
            while i < len(sql):
                if sql[i] == quote:
                    if i + 1 < len(sql) and sql[i + 1] == quote:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if ch == "[":
            close = sql.find("]", i + 1)
            i = len(sql) if close < 0 else close + 1
            continue
        if sql.startswith("--", i):
            end = sql.find("\n", i + 2)
            i = len(sql) if end < 0 else end + 1
            continue
        if sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            i = len(sql) if end < 0 else end + 2
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            closes.append((depth, i))
            depth = max(0, depth - 1)
            i += 1
            continue
        if ch.isalpha() or ch == "_":
            start = i
            i += 1
            while i < len(sql) and (sql[i].isalnum() or sql[i] in "_$"):
                i += 1
            words.append((sql[start:i].upper(), start, i, depth))
            continue
        i += 1
    return words, closes


def _repair_misplaced_compound_order_by(sql: str, error: str) -> str | None:
    """Move a query-block ORDER BY that SQLite rejected before UNION to the block tail.

    Sorting an unbounded SELECT arm has no useful guarantee; in a compound SELECT,
    the requested ordering belongs to the combined result. LIMIT/OFFSET is left to
    the model repair path because moving those operators could change query meaning.
    """
    if "order by clause should come after union all not before" not in error.casefold():
        return None
    words, closes = _sql_words_with_depth(sql)
    for index, token in enumerate(words[:-1]):
        word, start, _end, depth = token
        if word != "ORDER" or words[index + 1][0] != "BY" or words[index + 1][3] != depth:
            continue
        unions = [
            (j, item) for j, item in enumerate(words)
            if j > index and item[3] == depth and item[0] in {"UNION", "INTERSECT", "EXCEPT"}
        ]
        if not unions:
            continue
        first_union_index, first_union = unions[0]
        if any(item[0] in {"LIMIT", "OFFSET", "FETCH"} and item[3] == depth
               for item in words[index + 2:first_union_index]):
            continue
        last_union_index, last_union = unions[-1]
        block_close = min(
            (position for close_depth, position in closes
             if close_depth == depth and position > last_union[1]),
            default=len(sql),
        )
        if depth == 0:
            semicolon = sql.find(";", last_union[1])
            if semicolon >= 0:
                block_close = min(block_close, semicolon)
        final_order_exists = any(
            item[0] == "ORDER" and item[3] == depth and item[1] > last_union[1]
            and j + 1 < len(words) and words[j + 1][0] == "BY" and words[j + 1][3] == depth
            for j, item in enumerate(words)
        )
        clause = sql[start:first_union[1]].strip()
        if not clause:
            continue
        before = sql[:start].rstrip()
        combined = sql[first_union[1]:block_close].strip()
        after = sql[block_close:]
        if final_order_exists:
            return (before + "\n" + combined + after).strip()
        return (before + "\n" + combined + "\n" + clause + after).strip()
    return None


class DataAnalystAgent:
    """Data discovery, profiling, SQL analysis, and evidence packaging."""

    def __init__(
        self,
        action_generator: AnalysisStepGenerator,
        query_executor: QueryExecutor,
        db_path: str | Path | None = None,
        catalog_path: str | Path | None = None,
        retrieval_tool: Callable[[str, int], list[dict[str, Any]]] | None = None,
        agent_name: str = "data_analyst",
        allow_cleaning: bool = True,
    ) -> None:
        root = Path(__file__).resolve().parent
        self.db_path = Path(db_path) if db_path else root / "demo_company" / "data" / "business_demo.sqlite3"
        self.catalog_path = Path(catalog_path) if catalog_path else root / "demo_company" / "docs" / "source_catalog.json"
        self.action_generator = action_generator
        self.query_executor = query_executor
        self.retrieval_tool = retrieval_tool
        self.agent_name = agent_name
        self.allow_cleaning = allow_cleaning

    def _uri(self) -> str:
        if not self.db_path.is_file():
            raise AnalystError("The business SQLite database is missing.")
        return self.db_path.resolve().as_uri() + "?mode=ro"

    def _catalog(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise AnalystError("The source catalog could not be read.") from None
        if not isinstance(data, list):
            raise AnalystError("The source catalog is invalid.")
        return {row.get("id"): row for row in data if isinstance(row, dict) and isinstance(row.get("id"), str)}

    @staticmethod
    def _identifier(value: Any) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise AnalystError("The source catalog contains an invalid SQL identifier.")
        return '"' + value.replace('"', '""') + '"'

    def discover(self, source_ids: list[str]) -> list[dict[str, Any]]:
        """Inspect schemas and actual row/date coverage for the plan's data sources."""
        catalog = self._catalog()
        if not source_ids:
            raise AnalystError("The approved task has no data sources.")
        result = []
        try:
            with sqlite3.connect(self._uri(), uri=True, timeout=2) as connection:
                for source_id in source_ids:
                    entry = catalog.get(source_id)
                    if not entry:
                        raise AnalystError("The task names a source absent from the catalog.")
                    table_name = entry.get("sqlite_table")
                    table = self._identifier(table_name)
                    info = list(connection.execute("PRAGMA table_info(" + table + ")"))
                    if not info:
                        raise AnalystError("The catalog table is missing from the SQLite database.")
                    columns = [
                        {"name": row[1], "type": (row[2] or "UNDECLARED").upper(), "primary_key": bool(row[5])}
                        for row in info
                    ]
                    ranges = {}
                    for column in columns:
                        name = column["name"]
                        if (
                            "DATE" in column["type"] or "TIME" in column["type"]
                            or name.lower().endswith("_date") or name.lower() in {"date", "month"}
                        ):
                            quoted = self._identifier(name)
                            span = connection.execute(
                                f"SELECT MIN({quoted}), MAX({quoted}) FROM {table}"
                            ).fetchone()
                            ranges[name] = {"min": span[0], "max": span[1]}
                    actual_count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    result.append({
                        "dataset_id": source_id,
                        "table": table_name,
                        "description": entry.get("description", ""),
                        "actual_row_count": actual_count,
                        "catalog_row_count": entry.get("row_count"),
                        "columns": columns,
                        "date_ranges": ranges,
                    })
        except sqlite3.Error as exc:
            raise AnalystError("Schema discovery failed: " + " ".join(str(exc).split())[:180]) from None
        return result

    @staticmethod
    def _quantile(values: list[float], q: float) -> float:
        ordered = sorted(values)
        pos = (len(ordered) - 1) * q
        lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
        fraction = pos - lo
        return ordered[lo] * (1 - fraction) + ordered[hi] * fraction

    def profile(self, sources: list[dict[str, Any]]) -> dict[str, Any]:
        """Report missingness, uniqueness, distributions, and numeric outlier flags."""
        profiles = []
        flags = []
        try:
            with sqlite3.connect(self._uri(), uri=True, timeout=2) as connection:
                for source in sources:
                    table = self._identifier(source["table"])
                    columns = source["columns"]
                    count = source["actual_row_count"]
                    primary_keys = [c["name"] for c in columns if c["primary_key"]]
                    duplicate_keys = None
                    if len(primary_keys) == 1:
                        key = self._identifier(primary_keys[0])
                        duplicate_keys = connection.execute(
                            f"SELECT COUNT(*) - COUNT(DISTINCT {key}) FROM {table}"
                        ).fetchone()[0]
                        if duplicate_keys:
                            flags.append(f"{source['dataset_id']}: {duplicate_keys} duplicate primary-key values.")
                    column_results = []
                    for column in columns:
                        name = column["name"]
                        quoted = self._identifier(name)
                        non_null = connection.execute(
                            f"SELECT COUNT({quoted}) FROM {table}"
                        ).fetchone()[0]
                        distinct = connection.execute(
                            f"SELECT COUNT(DISTINCT {quoted}) FROM {table}"
                        ).fetchone()[0]
                        info = {
                            "column": name,
                            "type": column["type"],
                            "null_count": count - non_null,
                            "distinct_non_null": distinct,
                        }
                        is_text = any(t in column["type"] for t in ("TEXT", "CHAR", "CLOB"))
                        if is_text:
                            blank = connection.execute(
                                f"SELECT COUNT(*) FROM {table} WHERE {quoted} IS NOT NULL AND TRIM(CAST({quoted} AS TEXT)) = ''"
                            ).fetchone()[0]
                            info["blank_count"] = blank
                            if blank:
                                flags.append(f"{source['dataset_id']}.{name}: {blank} blank values.")
                            if distinct <= 20 and not any(
                                token in name.lower() for token in ("id", "name", "text", "review", "comment")
                            ) and distinct:
                                info["top_values"] = [
                                    {"value": row[0], "count": row[1]}
                                    for row in connection.execute(
                                        f"SELECT {quoted}, COUNT(*) FROM {table} WHERE {quoted} IS NOT NULL "
                                        f"GROUP BY {quoted} ORDER BY COUNT(*) DESC LIMIT 10"
                                    )
                                ]
                        if any(t in column["type"] for t in ("INT", "REAL", "NUM", "DEC", "FLOAT", "DOUBLE")):
                            sample = [
                                float(row[0]) for row in connection.execute(
                                    f"SELECT {quoted} FROM {table} WHERE {quoted} IS NOT NULL LIMIT 5001"
                                )
                                if isinstance(row[0], (int, float)) and math.isfinite(float(row[0]))
                            ]
                            info["numeric_sample_size"] = len(sample)
                            if sample:
                                q1, q3 = self._quantile(sample, .25), self._quantile(sample, .75)
                                iqr = q3 - q1
                                outliers = sum(x < q1 - 1.5 * iqr or x > q3 + 1.5 * iqr for x in sample)
                                info.update({
                                    "min": min(sample),
                                    "max": max(sample),
                                    "mean": sum(sample) / len(sample),
                                    "iqr_outlier_count_in_sample": outliers,
                                    "outlier_policy": "Flag only; no values removed",
                                })
                                if outliers:
                                    flags.append(
                                        f"{source['dataset_id']}.{name}: {outliers} possible IQR outliers in sample of {len(sample)}."
                                    )
                        if info["null_count"]:
                            flags.append(f"{source['dataset_id']}.{name}: {info['null_count']} NULL values.")
                        column_results.append(info)
                    profiles.append({
                        "dataset_id": source["dataset_id"],
                        "row_count": count,
                        "column_count": len(columns),
                        "duplicate_primary_key_values": duplicate_keys,
                        "columns": column_results,
                    })
        except sqlite3.Error as exc:
            raise AnalystError("Data profiling failed: " + " ".join(str(exc).split())[:180]) from None
        return {
            "datasets": profiles,
            "quality_flags": flags[:40],
            "cleaning_policy": "Source data is unchanged. Missing values and possible outliers are surfaced, never silently removed.",
        }

    def clean_source(self, source_id: str) -> dict[str, Any]:
        """Create an audited, in-memory cleaned CSV for one approved source.

        The SQLite source is opened read-only and never modified. Blank strings are
        normalized to NULL, text is trimmed, declared numeric/date fields are
        parsed conservatively, exact duplicate rows with duplicate primary keys
        are removed, and conflicting primary-key records remain for review.
        Missing values are never imputed and outliers are never deleted.
        """
        catalog = self._catalog()
        entry = catalog.get(source_id)
        if not entry:
            raise AnalystError("The cleaning request names a source absent from the catalog.")
        table_name = entry.get("sqlite_table")
        table = self._identifier(table_name)
        try:
            with sqlite3.connect(self._uri(), uri=True, timeout=2) as connection:
                schema = list(connection.execute("PRAGMA table_info(" + table + ")"))
                if not schema:
                    raise AnalystError("The requested cleaning source has no usable schema.")
                column_names = [row[1] for row in schema]
                types = {row[1]: (row[2] or "").upper() for row in schema}
                primary_keys = [row[1] for row in sorted(schema, key=lambda row: row[5]) if row[5]]
                cursor = connection.execute(
                    "SELECT * FROM " + table + " LIMIT " + str(MAX_CLEAN_ROWS + 1)
                )
                raw_rows = [dict(zip(column_names, row)) for row in cursor.fetchall()]
        except sqlite3.Error as exc:
            raise AnalystError("Read-only data cleaning failed: " + " ".join(str(exc).split())[:180]) from None
        if len(raw_rows) > MAX_CLEAN_ROWS:
            raise AnalystError(
                f"The cleaning preview is limited to {MAX_CLEAN_ROWS:,} rows; this source is larger."
            )

        transformations: dict[str, dict[str, int]] = {
            name: {"trimmed_text": 0, "blank_to_null": 0, "numeric_converted": 0,
                  "invalid_numeric_to_null": 0, "date_normalized": 0,
                  "invalid_date_to_null": 0}
            for name in column_names
        }
        before_nulls = {name: sum(row.get(name) is None for row in raw_rows) for name in column_names}
        cleaned_rows: list[dict[str, Any]] = []
        for original in raw_rows:
            row = dict(original)
            for name in column_names:
                value = row.get(name)
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped != value:
                        transformations[name]["trimmed_text"] += 1
                    value = stripped
                    if value == "":
                        transformations[name]["blank_to_null"] += 1
                        value = None
                declared_type = types[name]
                lower_name = name.lower()
                if value is not None and any(token in declared_type for token in ("INT", "REAL", "NUM", "DEC", "FLOAT", "DOUBLE")):
                    if isinstance(value, bool):
                        value = None
                        transformations[name]["invalid_numeric_to_null"] += 1
                    elif isinstance(value, str):
                        try:
                            numeric_value = float(value)
                            if not math.isfinite(numeric_value):
                                raise ValueError("non-finite")
                            if "INT" in declared_type:
                                if not numeric_value.is_integer():
                                    raise ValueError("not an integer")
                                value = int(numeric_value)
                            else:
                                value = numeric_value
                            transformations[name]["numeric_converted"] += 1
                        except (ValueError, OverflowError):
                            value = None
                            transformations[name]["invalid_numeric_to_null"] += 1
                    elif isinstance(value, float) and not math.isfinite(value):
                        value = None
                        transformations[name]["invalid_numeric_to_null"] += 1
                is_date = (
                    "DATE" in declared_type or "TIME" in declared_type
                    or lower_name.endswith("_date") or lower_name in {"date", "month"}
                )
                if value is not None and is_date and isinstance(value, str):
                    try:
                        normalized = value.replace("Z", "+00:00")
                        if lower_name == "month" and re.fullmatch(r"\d{4}-\d{2}", normalized):
                            year, month_number = (int(part) for part in normalized.split("-"))
                            if not 1 <= month_number <= 12:
                                raise ValueError("invalid month")
                            replacement = f"{year:04d}-{month_number:02d}"
                        elif "T" in normalized or (" " in normalized and ":" in normalized):
                            from datetime import datetime
                            parsed = datetime.fromisoformat(normalized)
                            replacement = parsed.isoformat()
                        else:
                            from datetime import date
                            replacement = date.fromisoformat(normalized).isoformat()
                        if replacement != value:
                            transformations[name]["date_normalized"] += 1
                        value = replacement
                    except ValueError:
                        value = None
                        transformations[name]["invalid_date_to_null"] += 1
                row[name] = value
            cleaned_rows.append(row)

        exact_duplicates_removed = 0
        conflicting_primary_key_rows = 0
        if primary_keys:
            seen_by_key: dict[tuple[Any, ...], set[str]] = {}
            deduplicated = []
            for row in cleaned_rows:
                key = tuple(row.get(name) for name in primary_keys)
                signature = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
                if any(part is None for part in key):
                    deduplicated.append(row)
                    continue
                variants = seen_by_key.setdefault(key, set())
                if signature in variants:
                    exact_duplicates_removed += 1
                    continue
                if variants:
                    conflicting_primary_key_rows += 1
                variants.add(signature)
                deduplicated.append(row)
            cleaned_rows = deduplicated

        after_nulls = {name: sum(row.get(name) is None for row in cleaned_rows) for name in column_names}
        outlier_flags = {}
        for name in column_names:
            if not any(token in types[name] for token in ("INT", "REAL", "NUM", "DEC", "FLOAT", "DOUBLE")):
                continue
            values = [
                float(row[name]) for row in cleaned_rows
                if isinstance(row.get(name), (int, float))
                and not isinstance(row.get(name), bool)
                and math.isfinite(float(row[name]))
            ]
            if len(values) >= 4:
                q1, q3 = self._quantile(values, .25), self._quantile(values, .75)
                iqr = q3 - q1
                outlier_flags[name] = {
                    "count": sum(value < q1 - 1.5 * iqr or value > q3 + 1.5 * iqr for value in values),
                    "method": "1.5*IQR flag only; no values removed",
                    "n": len(values),
                }
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=column_names, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in cleaned_rows:
            csv_row = {}
            for name in column_names:
                value = row.get(name)
                text = "" if value is None else str(value)
                if isinstance(value, str) and text.startswith(("=", "+", "-", "@")):
                    text = "'" + text
                csv_row[name] = text
            writer.writerow(csv_row)
        csv_text = output.getvalue()
        if len(csv_text.encode("utf-8")) > MAX_CLEAN_CSV_BYTES:
            raise AnalystError("The cleaned CSV exceeds the 15 MB download limit.")

        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(entry.get("name", source_id)))
        filename = filename.rsplit(".", 1)[0] + "_cleaned.csv"
        return {
            "dataset_id": source_id,
            "table": table_name,
            "columns": column_names,
            "rows": cleaned_rows[:25],
            "row_count_input": len(raw_rows),
            "row_count_output": len(cleaned_rows),
            "exact_duplicate_rows_removed": exact_duplicates_removed,
            "conflicting_primary_key_rows_preserved": conflicting_primary_key_rows,
            "nulls_by_column_before": before_nulls,
            "nulls_by_column_after": after_nulls,
            "transformations_by_column": transformations,
            "outlier_flags_by_column": outlier_flags,
            "imputation": "None. Missing values remain missing.",
            "outlier_policy": "Possible outliers are flagged for review; no rows or values are deleted as outliers.",
            "duplicate_policy": "Only exact duplicate rows with the same declared primary key are removed. Conflicting rows sharing a primary key are preserved and flagged.",
            "source_policy": "Source SQLite database is read-only and unchanged.",
            "download_filename": filename,
            "cleaned_csv": csv_text,
        }

    @staticmethod
    def validate_result(result: dict[str, Any]) -> dict[str, Any]:
        rows = result.get("rows", [])
        columns = result.get("columns", [])
        if (
            not isinstance(rows, list)
            or not isinstance(columns, list)
            or not columns
            or any(not isinstance(row, dict) for row in rows)
        ):
            raise AnalystError("SQL result has an invalid shape.")
        nulls = {column: sum(row.get(column) is None for row in rows) for column in columns}
        distinct_rows = {json.dumps(row, sort_keys=True, default=str) for row in rows}
        numeric = {}
        for column in columns:
            values = [
                float(row[column]) for row in rows
                if isinstance(row.get(column), (int, float))
                and not isinstance(row.get(column), bool)
                and math.isfinite(float(row[column]))
            ]
            if values:
                q1, q3 = DataAnalystAgent._quantile(values, .25), DataAnalystAgent._quantile(values, .75)
                iqr = q3 - q1
                numeric[column] = {
                    "n": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "median": DataAnalystAgent._quantile(values, .5),
                    "iqr_outlier_count": sum(x < q1 - 1.5 * iqr or x > q3 + 1.5 * iqr for x in values),
                }

        checks: list[dict[str, Any]] = []
        failures: list[str] = []

        def record(name: str, passed: bool, detail: str) -> None:
            checks.append({"name": name, "passed": passed, "detail": detail})
            if not passed:
                failures.append(name + ": " + detail)

        def number(row: dict[str, Any], key: str) -> float | None:
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            value = float(value)
            return value if math.isfinite(value) else None

        # Recognize explicitly named common period-comparison result contracts.
        arithmetic_specs = (
            ("period_1_sales", "period_2_sales", "change_vs_period_2_usd", "change_vs_period_2_pct"),
            ("period_1_sales", "period_3_sales", "change_vs_period_3_usd", "change_vs_period_3_pct"),
            ("current_value", "comparison_value", "change_usd", "change_pct"),
            ("current_value", "prior_value", "change_usd", "change_pct"),
            ("current_sales", "prior_sales", "change_usd", "change_pct"),
            ("nov_sales", "oct_sales", "dollar_change", "percent_change"),
        )
        for current_key, baseline_key, delta_key, percent_key in arithmetic_specs:
            if not {current_key, baseline_key, delta_key}.issubset(columns):
                continue
            for row_index, row in enumerate(rows):
                current, baseline, reported_delta = (
                    number(row, current_key), number(row, baseline_key), number(row, delta_key)
                )
                if current is None or baseline is None or reported_delta is None:
                    record(
                        f"arithmetic:{delta_key}:row{row_index + 1}", False,
                        "comparison inputs or reported dollar change are missing/non-numeric",
                    )
                    continue
                expected_delta = round(current - baseline, 2)
                if not math.isclose(reported_delta, expected_delta, abs_tol=0.011, rel_tol=0):
                    record(
                        f"arithmetic:{delta_key}:row{row_index + 1}", False,
                        f"reported {reported_delta:.2f}, recalculated {expected_delta:.2f}",
                    )
                    continue
                if percent_key in columns:
                    reported_percent = row.get(percent_key)
                    expected_percent = None if baseline == 0 else round(100 * (current - baseline) / baseline, 2)
                    if expected_percent is None:
                        passed = reported_percent is None
                        detail = "percentage must be NULL when the comparison baseline is zero"
                    else:
                        numeric_percent = number(row, percent_key)
                        passed = numeric_percent is not None and math.isclose(
                            numeric_percent, expected_percent, abs_tol=0.021, rel_tol=0
                        )
                        detail = (
                            f"reported {numeric_percent!r}, recalculated {expected_percent:.2f}%"
                        )
                    record(f"arithmetic:{percent_key}:row{row_index + 1}", passed, detail)
                else:
                    record(
                        f"arithmetic:{delta_key}:row{row_index + 1}", True,
                        f"reported and recalculated change are {expected_delta:.2f}",
                    )

        # If an explicit total row is returned, check it against each complete
        # dimension rollup. Skip intentionally limited/truncated result sets.
        dimension_column = next(
            (name for name in ("dimension_type", "group_type") if name in columns), None
        )
        specialist_sql = str(result.get("specialist_sql", ""))
        if dimension_column is not None:
            rank_columns = [name for name in columns if "rank" in name.casefold()]
            rollup_types = {"total", "grand_total", "all", "subtotal", "rollup"}
            for row_index, row in enumerate(rows):
                row_type = str(row.get(dimension_column, "")).strip().casefold()
                if row_type in rollup_types:
                    for rank_column in rank_columns:
                        if row.get(rank_column) is not None:
                            record(
                                f"ranking:rollup:{rank_column}:row{row_index + 1}", False,
                                "total and rollup rows must not receive a detail-group rank",
                            )
        if result.get("truncated") or re.search(r"\bLIMIT\b", specialist_sql, flags=re.IGNORECASE):
            reconciliation_status = "skipped: query is truncated or explicitly limited"
        elif dimension_column is None:
            reconciliation_status = "not_applicable: no explicit dimension type column"
        else:
            total_rows = [
                row for row in rows
                if str(row.get(dimension_column, "")).strip().lower() in {"total", "grand_total", "all"}
            ]
            if len(total_rows) != 1:
                reconciliation_status = "not_applicable: result has no unique explicit total row"
            else:
                total_row = total_rows[0]
                detail_types = {
                    str(row.get(dimension_column, "")) for row in rows
                    if row is not total_row and row.get(dimension_column) is not None
                }
                effect_columns: dict[str, list[str]] = {}
                measure_columns = []
                for name in columns:
                    lowered = name.lower()
                    if "effect" in lowered and lowered.endswith("_usd"):
                        match = re.search(r"_effect_(vs_.+)_usd$", lowered)
                        if match:
                            effect_columns.setdefault(match.group(1), []).append(name)
                        continue
                    if (
                        lowered.startswith("period_") and lowered.endswith("_sales")
                        or lowered.endswith("_usd") and "rank" not in lowered
                        or lowered.endswith("_amount")
                    ):
                        measure_columns.append(name)
                reconciliation_count = 0
                for detail_type in sorted(detail_types):
                    detail_rows = [
                        row for row in rows
                        if row is not total_row and str(row.get(dimension_column, "")) == detail_type
                    ]
                    for measure in measure_columns:
                        expected_values = [number(row, measure) for row in detail_rows]
                        expected_total = number(total_row, measure)
                        if expected_total is None or not expected_values or any(value is None for value in expected_values):
                            continue
                        summed = round(sum(value for value in expected_values if value is not None), 2)
                        passed = math.isclose(summed, expected_total, abs_tol=0.021, rel_tol=0)
                        record(
                            f"reconciliation:{detail_type}:{measure}", passed,
                            f"detail sum {summed:.2f}, total row {expected_total:.2f}",
                        )
                        reconciliation_count += 1
                    # Decomposition components are not additive one by one across
                    # segments; their combined amount must reconcile to the total.
                    for comparison, effects in sorted(effect_columns.items()):
                        if len(effects) < 2:
                            continue
                        detail_components = [
                            number(row, effect)
                            for row in detail_rows for effect in effects
                        ]
                        total_components = [number(total_row, effect) for effect in effects]
                        if (
                            not detail_components or any(value is None for value in detail_components)
                            or any(value is None for value in total_components)
                        ):
                            continue
                        summed_effects = round(sum(value for value in detail_components if value is not None), 2)
                        total_effects = round(sum(value for value in total_components if value is not None), 2)
                        passed = math.isclose(summed_effects, total_effects, abs_tol=0.021, rel_tol=0)
                        record(
                            f"reconciliation:{detail_type}:decomposition_{comparison}_usd", passed,
                            f"combined detail effects {summed_effects:.2f}, total row effects {total_effects:.2f}",
                        )
                        reconciliation_count += 1
                reconciliation_status = (
                    f"checked {reconciliation_count} dimension/measure totals"
                    if reconciliation_count else "not_applicable: no comparable numeric rollups"
                )

        return {
            "nulls_by_result_column": nulls,
            "duplicate_result_rows": len(rows) - len(distinct_rows),
            "numeric_result_summary": numeric,
            "deterministic_checks": checks,
            "failed_checks": failures,
            "all_checks_passed": not failures,
            "reconciliation_status": reconciliation_status,
            "truncated": bool(result.get("truncated")),
        }

    @staticmethod
    def chart_hint(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, Any] | None:
        if not 2 <= len(rows) <= 50:
            return None
        measures = [
            name for name in columns
            if any(isinstance(row.get(name), (int, float)) for row in rows)
        ]
        dimensions = [name for name in columns if name not in measures]
        if not measures or not dimensions:
            return None
        categorical = next(
            (name for name in dimensions if name.lower() in {
                "dimension_value", "category", "product_category", "region", "channel", "carrier", "segment", "product_name"
            }),
            None,
        )
        # For wide period-comparison results, put the category on the x-axis and
        # plot the period measures as separate series. Period label columns are
        # metadata for the series, not the category axis.
        period_measures = [
            name for name in measures
            if re.fullmatch(
                r"period_\d+_(?:net_)?(?:sales|revenue|orders|customers|quantity|units|profit|margin)",
                name.lower(),
            )
        ]

        def measure_family(name: str) -> str:
            lowered = name.lower()
            if any(token in lowered for token in ("pct", "percent", "rate", "margin")):
                return "rate"
            if any(token in lowered for token in ("sales", "revenue", "profit", "usd", "amount", "cost", "refund", "discount", "aov")):
                return "currency"
            if any(token in lowered for token in ("order", "customer", "count", "unit", "quantity", "number")):
                return "count"
            return "other"

        has_period_labels = any(
            re.fullmatch(r"period_\d+_label", name.lower()) for name in dimensions
        )
        if categorical and has_period_labels and len(period_measures) >= 2:
            families = {measure_family(name) for name in period_measures}
            selected_family = next(
                (family for family in ("currency", "count", "rate", "other") if family in families),
                "other",
            )
            selected = [name for name in period_measures if measure_family(name) == selected_family]
            return {"type": "bar", "x": categorical, "y": selected[:4]}

        temporal = next(
            (name for name in dimensions if any(t in name.lower() for t in ("date", "month", "week", "period"))),
            None,
        )
        x = temporal or categorical or dimensions[0]
        families = {measure_family(name) for name in measures}
        selected_family = next(
            (family for family in ("currency", "count", "rate", "other") if family in families),
            "other",
        )
        selected_measures = [name for name in measures if measure_family(name) == selected_family]
        return {
            "type": "line" if any(t in x.lower() for t in ("date", "month", "week", "period")) else "bar",
            "x": x,
            "y": selected_measures[:3],
        }

    def run_task(self, question: str, task: PlannerTask) -> list[dict[str, Any]]:
        """Iteratively analyze an approved task and return query-by-query evidence."""
        if task.agent != self.agent_name:
            raise AnalystError(f"{self.agent_name} cannot accept a {task.agent} task.")
        discovered = self.discover(task.data_sources)
        profile = self.profile(discovered)
        query_evidence: list[dict[str, Any]] = []
        previous_error: dict[str, str] | None = None
        retrieval_history: list[dict[str, Any]] = []
        rag_unavailable = False
        finished = False

        def record_rag_limit() -> None:
            evidence_id = f"{task.task_id}:rag_limit"
            if any(item.get("evidence_id") == evidence_id for item in query_evidence):
                return
            query_evidence.append({
                "evidence_id": evidence_id,
                "task_id": task.task_id,
                "agent": self.agent_name,
                "objective": task.objective,
                "action": "rag_retrieval",
                "evidence_type": "retrieval_status",
                "source_name": "Local PostgreSQL/pgvector document store",
                "source_ref": "per-task retrieval limit",
                "status": "limit_reached",
                "message": (
                    f"The per-task limit of {MAX_RAG_RETRIEVALS} semantic searches was reached. "
                    "No additional passages were fetched. Any text findings must rely only on the "
                    "retrieved passages in this evidence set and must not be presented as a full-corpus count."
                ),
            })

        for _action_number in range(MAX_ANALYSIS_ACTIONS):
            history = []
            for item in query_evidence:
                if item.get("evidence_type") == "document_context":
                    continue
                preview_columns = item.get("columns", [])[:24]
                history.append({
                    "evidence_id": item["evidence_id"],
                    "sql": item.get("specialist_sql", "")[:12000],
                    "tables_read": item.get("tables_read", []),
                    "columns": preview_columns,
                    "rows_preview": [
                        {column: row.get(column) for column in preview_columns}
                        for row in item.get("rows", [])[:25]
                    ],
                    "row_count_returned": item.get("row_count_returned"),
                    "truncated": item.get("truncated", False),
                    "result_validation": item.get("result_validation", {}),
                    "cleaning_summary": item.get("cleaning_summary"),
                    "statistical_tests": item.get("statistical_tests"),
                    "observation_grain": item.get("observation_grain"),
                })
            state = {
                "source_discovery": discovered,
                "quality_flags": profile.get("quality_flags", [])[:30],
                "profile_summary": [
                    {
                        "dataset_id": dataset.get("dataset_id"),
                        "row_count": dataset.get("row_count"),
                        "column_count": dataset.get("column_count"),
                        "duplicate_primary_key_values": dataset.get("duplicate_primary_key_values"),
                        "columns": [
                            {
                                key: column[key]
                                for key in ("column", "type", "null_count", "distinct_non_null", "min", "max", "mean")
                                if key in column
                            }
                            for column in dataset.get("columns", [])
                        ],
                    }
                    for dataset in profile.get("datasets", [])
                ],
                "query_history": history,
                "retrieval_history": retrieval_history[-MAX_RAG_RETRIEVALS:],
                "rag_calls_remaining": 0 if rag_unavailable else MAX_RAG_RETRIEVALS - len(retrieval_history),
                "rag_unavailable": rag_unavailable,
                "previous_error": previous_error,
                "successful_query_count": sum(bool(item.get("specialist_sql")) for item in query_evidence),
                "completed_cleaning_sources": [
                    item.get("dataset_id") for item in query_evidence if item.get("action") == "clean_dataset"
                ],
                "max_queries": MAX_ANALYSIS_QUERIES,
            }
            action = self.action_generator(question, task, state)
            if not isinstance(action, dict):
                raise AnalystError("The analyst returned an invalid analysis action.")
            action_type = action.get("action")
            if action_type == "done":
                if action.get("retrieval_limit_reached") is True:
                    record_rag_limit()
                if previous_error is not None:
                    raise AnalystError("The analyst stopped after a failed follow-up query; no report was released.")
                has_sql = any(item.get("specialist_sql") for item in query_evidence)
                has_documents = any(item.get("evidence_type") == "document_context" for item in query_evidence)
                if not has_sql and not has_documents:
                    raise AnalystError("The analyst stopped without executing a query or retrieving usable document evidence.")
                finished = True
                break
            if action_type == "retrieve":
                if self.retrieval_tool is None:
                    raise AnalystError("The RAG retrieval tool is not configured.")
                retrieval_query = action.get("query")
                if not isinstance(retrieval_query, str) or not retrieval_query.strip() or len(retrieval_query) > 500:
                    raise AnalystError("The analyst returned an invalid RAG search query.")
                retrieval_number = len(retrieval_history) + 1
                if retrieval_number > MAX_RAG_RETRIEVALS:
                    record_rag_limit()
                    has_sql = any(item.get("specialist_sql") for item in query_evidence)
                    has_documents = any(item.get("evidence_type") == "document_context" for item in query_evidence)
                    if not has_sql and not has_documents:
                        raise AnalystError("The RAG search limit was reached without usable SQL or retrieved passages; no report was released.")
                    finished = True
                    break
                try:
                    results = self.retrieval_tool(retrieval_query.strip(), 5)
                except Exception:
                    # Semantic retrieval is an optional evidence source. Keep a
                    # traceable limitation and let approved SQL analysis continue.
                    rag_unavailable = True
                    evidence_id = f"{task.task_id}:rag{retrieval_number}:unavailable"
                    message = (
                        "Semantic document retrieval was unavailable; no passages were returned. "
                        "Any requested document definitions or review themes that require retrieval "
                        "must be reported as unavailable."
                    )
                    query_evidence.append({
                        "evidence_id": evidence_id,
                        "task_id": task.task_id,
                        "agent": self.agent_name,
                        "objective": task.objective,
                        "action": "rag_retrieval",
                        "evidence_type": "retrieval_status",
                        "retrieval_query": retrieval_query.strip(),
                        "source_name": "Local PostgreSQL/pgvector document store",
                        "source_ref": "RAG service availability",
                        "status": "unavailable",
                        "message": message,
                    })
                    retrieval_history.append({
                        "query": retrieval_query.strip(),
                        "result_count": 0,
                        "evidence_ids": [evidence_id],
                        "results": [],
                        "status": "unavailable",
                    })
                    previous_error = None
                    continue
                if not isinstance(results, list) or len(results) > 5:
                    raise AnalystError("The RAG tool returned an invalid result list.")
                evidence_ids = []
                compact_results = []
                for result_number, result in enumerate(results, 1):
                    if not isinstance(result, dict):
                        continue
                    source_ref = result.get("source_ref")
                    excerpt = result.get("content")
                    if not isinstance(source_ref, str) or not isinstance(excerpt, str) or not excerpt.strip():
                        continue
                    evidence_id = f"{task.task_id}:rag{retrieval_number}:{result_number}"
                    evidence_ids.append(evidence_id)
                    safe_excerpt = excerpt[:1500]
                    document_evidence = {
                        "evidence_id": evidence_id,
                        "task_id": task.task_id,
                        "agent": self.agent_name,
                        "objective": task.objective,
                        "action": "rag_retrieval",
                        "evidence_type": "document_context",
                        "retrieval_query": retrieval_query.strip(),
                        "source_name": str(result.get("source_name", ""))[:200],
                        "source_ref": source_ref[:300],
                        "title": str(result.get("title", ""))[:300],
                        "section": str(result.get("section", ""))[:300],
                        "excerpt": safe_excerpt,
                    }
                    query_evidence.append(document_evidence)
                    compact_results.append({
                        "evidence_id": evidence_id,
                        "source_ref": document_evidence["source_ref"],
                        "section": document_evidence["section"],
                        "excerpt": safe_excerpt[:600],
                    })
                retrieval_history.append({
                    "query": retrieval_query.strip(),
                    "result_count": len(compact_results),
                    "evidence_ids": evidence_ids,
                    "results": compact_results,
                })
                previous_error = None
                continue
            if action_type == "clean":
                if not self.allow_cleaning:
                    raise AnalystError(f"{self.agent_name} does not support dataset-cleaning actions.")
                source_id = action.get("dataset_id")
                if not isinstance(source_id, str) or source_id not in task.data_sources:
                    raise AnalystError("The cleaning action must name a source approved in the plan.")
                if source_id in state["completed_cleaning_sources"]:
                    raise AnalystError("The analyst repeated a completed cleaning action.")
                cleaned = self.clean_source(source_id)
                table = self._identifier(cleaned["table"])
                cleaning_summary = {
                    key: value for key, value in cleaned.items()
                    if key not in {"cleaned_csv", "rows", "columns", "download_filename"}
                }
                clean_number = sum(item.get("action") == "clean_dataset" for item in query_evidence) + 1
                evidence = {
                    "evidence_id": f"{task.task_id}:clean{clean_number}",
                    "task_id": task.task_id,
                    "agent": self.agent_name,
                    "objective": task.objective,
                    "action": "clean_dataset",
                    "analysis_step": clean_number,
                    "dataset_id": source_id,
                    "sql": f"SELECT * FROM {table}",
                    "specialist_sql": f"SELECT * FROM {table}",
                    "tables_read": [cleaned["table"]],
                    "columns": cleaned["columns"],
                    "rows": cleaned["rows"],
                    "row_count_returned": cleaned["row_count_output"],
                    "preview_row_count": len(cleaned["rows"]),
                    "total_cleaned_rows": cleaned["row_count_output"],
                    "preview_truncated": cleaned["row_count_output"] > len(cleaned["rows"]),
                    "truncated": False,
                    "cleaning_summary": cleaning_summary,
                    "cleaned_csv": cleaned["cleaned_csv"],
                    "download_filename": cleaned["download_filename"],
                    "analysis_complete": False,
                }
                if not query_evidence:
                    evidence["source_discovery"] = discovered
                    evidence["source_profile"] = profile
                query_evidence.append(evidence)
                previous_error = None
                continue

            if action_type not in {"query", "statistical_test"}:
                raise AnalystError(
                    "The analyst must return a query, statistical_test, retrieve, clean, or done."
                )
            query_count = sum(bool(item.get("specialist_sql")) for item in query_evidence)
            if query_count >= MAX_ANALYSIS_QUERIES:
                raise AnalystError(
                    f"The analysis reached its {MAX_ANALYSIS_QUERIES}-query limit before the analyst marked it complete."
                )
            sql = action.get("sql")
            if not isinstance(sql, str) or not sql.strip() or len(sql) > 20000:
                raise AnalystError("The analyst returned an invalid or oversized SQL query.")

            repair_attempts = (
                int(previous_error.get("repair_attempts", 0))
                if isinstance(previous_error, dict) else 0
            )
            repairing = repair_attempts > 0
            locally_repaired = False
            query_error: Exception | None = None
            try:
                result = self.query_executor(sql, task)
            except Exception as exc:
                query_error = exc
                error_text = " ".join(str(exc).split())[:240]
                repaired_sql = _repair_misplaced_compound_order_by(sql, error_text)
                if repaired_sql:
                    try:
                        result = self.query_executor(repaired_sql, task)
                    except Exception as repair_exc:
                        sql = repaired_sql
                        query_error = repair_exc
                        error_text = " ".join(str(repair_exc).split())[:240]
                    else:
                        sql = repaired_sql
                        query_error = None
                        locally_repaired = True
            if query_error is not None:
                error_text = " ".join(str(query_error).split())[:240]
                if repair_attempts >= MAX_QUERY_REPAIRS:
                    order_match = re.search(r"\bORDER\s+BY\b[\s\S]*", sql, flags=re.IGNORECASE)
                    order_preview = " ".join(order_match.group(0).split())[:240] if order_match else ""
                    diagnostic = f" Failed ORDER BY clause: {order_preview}" if order_preview else ""
                    raise AnalystError(
                        f"The analyst could not repair its query after {MAX_QUERY_REPAIRS} attempts: "
                        + error_text + diagnostic
                    ) from None
                previous_error = {
                    "sql": sql[:12000],
                    "error": " ".join(str(query_error).split())[:300],
                    "repair_attempts": repair_attempts + 1,
                }
                continue

            previous_error = None
            checks = self.validate_result(result)
            if checks["failed_checks"]:
                raise AnalystError(
                    "The query result failed deterministic validation; no report was released: "
                    + "; ".join(checks["failed_checks"][:4])
                )
            if action_type == "statistical_test":
                if result.get("truncated"):
                    raise AnalystError("Statistical testing cannot use a truncated set of group summaries.")
                try:
                    statistical_tests = welch_tests_from_summaries(result["rows"])
                except AnalysisMethodError as exc:
                    raise AnalystError("The requested statistical test could not be applied: " + str(exc)) from None
                grain = action.get("observation_grain")
                if not isinstance(grain, str) or not grain.strip():
                    raise AnalystError("A statistical test must state its observation grain.")
                result["observation_grain"] = grain.strip()
                result["statistical_tests"] = statistical_tests
                result["analysis_method"] = statistical_tests["test"]
                result["chart_rows"] = [
                    {"group_label": group["group_label"], "mean": group["mean"]}
                    for group in statistical_tests["groups"]
                ]
                chart_suggestion = {"type": "bar", "x": "group_label", "y": ["mean"]}
            else:
                if action.get("template_id") == "demo_customer_segment_period_comparison_v1":
                    result["chart_rows"] = [
                        {
                            "period_customer_segment": (
                                f"{row.get('dimension_type', '')} / {row.get('dimension_value', '')} / "
                                f"{row.get('period_label', '')}"
                            ),
                            "active_customers": row.get("active_customers", 0),
                        }
                        for row in result["rows"]
                        if row.get("dimension_type") != "total"
                    ]
                    chart_suggestion = {
                        "type": "bar", "x": "period_customer_segment", "y": ["active_customers"]
                    }
                elif action.get("template_id") == "demo_customer_cohort_retention_period_comparison_v1":
                    result["chart_rows"] = [
                        {
                            "cohort_offset": f"{row.get('cohort_label', '')} / M+{row.get('month_number', '')}",
                            "retention_pct": row.get("retention_pct"),
                        }
                        for row in result["rows"]
                    ]
                    chart_suggestion = {
                        "type": "bar", "x": "cohort_offset", "y": ["retention_pct"]
                    }
                elif action.get("template_id") == "demo_customer_lifecycle_period_comparison_v1":
                    result["chart_rows"] = [
                        {
                            "period_customer_type": f"{row.get('period_label', '')} / {row.get('customer_type', '')}",
                            "active_customers": row.get("active_customers", 0),
                        }
                        for row in result["rows"]
                        if row.get("customer_type") not in {"all", None}
                    ]
                    chart_suggestion = {
                        "type": "bar", "x": "period_customer_type", "y": ["active_customers"]
                    }
                elif action.get("template_id") == "demo_product_performance_period_comparison_v1":
                    result["chart_rows"] = [
                        {
                            "category_period": f"{row.get('dimension_value', '')} / {row.get('period_label', '')}",
                            "net_product_sales_usd": row.get("net_product_sales_usd"),
                        }
                        for row in result["rows"]
                        if row.get("dimension_type") == "product_category"
                    ]
                    chart_suggestion = {
                        "type": "bar", "x": "category_period", "y": ["net_product_sales_usd"]
                    }
                elif action.get("template_id") == "demo_operations_period_comparison_v1":
                    result["chart_rows"] = [
                        {
                            "measure_group_period": (
                                f"{row.get('metric_type', '')} / {row.get('dimension_value', '')} / "
                                f"{row.get('period_label', '')}"
                            ),
                            "rate_pct": row.get("rate_pct"),
                        }
                        for row in result["rows"]
                    ]
                    chart_suggestion = {
                        "type": "bar", "x": "measure_group_period", "y": ["rate_pct"]
                    }
                else:
                    chart_suggestion = self.chart_hint(result["rows"], result["columns"])
            query_number = sum(bool(item.get("specialist_sql")) for item in query_evidence) + 1
            evidence = {
                **result,
                "evidence_id": f"{task.task_id}:query{query_number}",
                "task_id": task.task_id,
                "agent": self.agent_name,
                "objective": task.objective,
                "action": action_type,
                "analysis_step": query_number,
                "result_validation": checks,
                "chart_suggestion": chart_suggestion,
                "sql_repair_attempted": repairing or locally_repaired,
                "sql_repair": (
                    "Relocated an ORDER BY clause after the compound SELECT following SQLite's syntax error."
                    if locally_repaired else None
                ),
                "query_template": action.get("template_id"),
                "template_limitations": action.get("template_limitations", []),
                "analysis_complete": False,
            }
            if query_number == 1:
                evidence["source_discovery"] = discovered
                evidence["source_profile"] = profile
                evidence["cleaning_policy"] = profile["cleaning_policy"]
            query_evidence.append(evidence)
            if action.get("complete_task") is True:
                finished = True
                break

        if not finished:
            raise AnalystError("The analyst did not finish its bounded analysis loop; no report was released.")
        query_evidence[-1]["analysis_complete"] = True
        return query_evidence

