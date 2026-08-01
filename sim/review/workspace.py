from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ReviewStoreNotFoundError(FileNotFoundError):
    """Raised when a review SQLite store cannot be found."""


class ReviewQueryError(ValueError):
    """Raised when a review query is unsafe or cannot be executed."""


@dataclass(frozen=True)
class ReviewQueryResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool = False


@dataclass(frozen=True)
class ReviewWorkspace:
    output_dir: Path
    db_path: Path
    schema_path: Path
    saved_views_path: Path

    @classmethod
    def open(cls, path: str | Path) -> ReviewWorkspace:
        root = Path(path).expanduser().resolve()
        if root.is_file() and root.name == "run.sqlite":
            db_path = root
            output_dir = root.parent.parent
        else:
            output_dir = root
            db_path = output_dir / "review" / "run.sqlite"
        if not db_path.is_file():
            raise ReviewStoreNotFoundError(f"Review store not found: {db_path}")
        review_dir = db_path.parent
        return cls(
            output_dir=output_dir,
            db_path=db_path,
            schema_path=review_dir / "schema.json",
            saved_views_path=review_dir / "saved_views.json",
        )

    def tables(self) -> list[str]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT name
                FROM sqlite_schema
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()
        return [str(row["name"]) for row in rows]

    def schema(self) -> dict[str, Any]:
        if self.schema_path.is_file():
            try:
                schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                schema = {}
        else:
            schema = {}
        out = dict(schema)
        out["columns"] = self.table_columns()
        return out

    def table_columns(self) -> dict[str, list[dict[str, Any]]]:
        columns: dict[str, list[dict[str, Any]]] = {}
        with closing(self._connect()) as conn:
            for table in self.tables():
                rows = conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
                columns[table] = [
                    {
                        "name": str(row["name"]),
                        "type": str(row["type"] or ""),
                        "notnull": bool(row["notnull"]),
                        "primary_key": bool(row["pk"]),
                    }
                    for row in rows
                ]
        return columns

    def saved_views(self) -> list[dict[str, Any]]:
        if not self.saved_views_path.is_file():
            return []
        data = json.loads(self.saved_views_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(item) for item in data if isinstance(item, dict)]
        if isinstance(data, dict) and isinstance(data.get("views"), list):
            return [dict(item) for item in data["views"] if isinstance(item, dict)]
        return []

    def query(
        self,
        sql: str,
        parameters: Any = None,
        *,
        max_rows: int = 1000,
        max_vm_steps: int = 250_000,
    ) -> ReviewQueryResult:
        statement = _validate_select_sql(sql)
        limit = int(max(max_rows, 1))
        step_budget = int(max(max_vm_steps, 1))
        params = () if parameters is None else parameters
        try:
            with closing(self._connect(authorize=True)) as conn:
                steps = 0

                def _abort_long_query() -> int:
                    nonlocal steps
                    steps += 1000
                    return 1 if steps > step_budget else 0

                conn.set_progress_handler(_abort_long_query, 1000)
                cursor = conn.execute(statement, params)
                rows = cursor.fetchmany(limit + 1)
                conn.set_progress_handler(None, 0)
        except sqlite3.Error as exc:
            message = str(exc)
            if "interrupted" in message.lower():
                message = "Review query exceeded the execution step budget."
            raise ReviewQueryError(message) from exc
        columns = [str(item[0]) for item in (cursor.description or [])]
        truncated = len(rows) > limit
        visible_rows = rows[:limit]
        return ReviewQueryResult(
            columns=columns,
            rows=[{column: row[column] for column in columns} for row in visible_rows],
            row_count=len(visible_rows),
            truncated=truncated,
        )

    def _connect(self, *, authorize: bool = False) -> sqlite3.Connection:
        uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        if authorize:
            conn.set_authorizer(_read_only_authorizer)
        return conn


def _validate_select_sql(sql: str) -> str:
    statement = str(sql or "").strip()
    if not statement:
        raise ReviewQueryError("Review query cannot be empty.")
    if "\x00" in statement:
        raise ReviewQueryError("Review query cannot contain null bytes.")
    statement = _strip_leading_comments(statement).strip()
    statement = _strip_single_trailing_semicolon(statement)
    first_token = _first_sql_token(statement)
    if first_token not in {"select", "with"}:
        raise ReviewQueryError("Review queries are read-only and must start with SELECT or WITH.")
    return statement


def _strip_single_trailing_semicolon(statement: str) -> str:
    semicolon_count = 0
    in_single = False
    in_double = False
    for char in statement:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == ";" and not in_single and not in_double:
            semicolon_count += 1
    if semicolon_count == 0:
        return statement
    stripped = statement.rstrip()
    if semicolon_count == 1 and stripped.endswith(";"):
        return stripped[:-1].rstrip()
    raise ReviewQueryError("Review queries must contain exactly one read-only statement.")


def _strip_leading_comments(statement: str) -> str:
    out = statement.lstrip()
    while out.startswith("--") or out.startswith("/*"):
        if out.startswith("--"):
            _, _, remainder = out.partition("\n")
            out = remainder.lstrip()
            continue
        end = out.find("*/")
        if end < 0:
            raise ReviewQueryError("Review query has an unterminated block comment.")
        out = out[end + 2 :].lstrip()
    return out


def _first_sql_token(statement: str) -> str:
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", statement)
    return match.group(1).lower() if match else ""


def _read_only_authorizer(action_code: int, param1: str | None, param2: str | None, *_args: Any) -> int:
    dangerous = {
        getattr(sqlite3, "SQLITE_ALTER_TABLE", -1),
        getattr(sqlite3, "SQLITE_ATTACH", -1),
        getattr(sqlite3, "SQLITE_CREATE_INDEX", -1),
        getattr(sqlite3, "SQLITE_CREATE_TABLE", -1),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_INDEX", -1),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_TABLE", -1),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_TRIGGER", -1),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_VIEW", -1),
        getattr(sqlite3, "SQLITE_CREATE_TRIGGER", -1),
        getattr(sqlite3, "SQLITE_CREATE_VIEW", -1),
        getattr(sqlite3, "SQLITE_DELETE", -1),
        getattr(sqlite3, "SQLITE_DETACH", -1),
        getattr(sqlite3, "SQLITE_DROP_INDEX", -1),
        getattr(sqlite3, "SQLITE_DROP_TABLE", -1),
        getattr(sqlite3, "SQLITE_DROP_TEMP_INDEX", -1),
        getattr(sqlite3, "SQLITE_DROP_TEMP_TABLE", -1),
        getattr(sqlite3, "SQLITE_DROP_TEMP_TRIGGER", -1),
        getattr(sqlite3, "SQLITE_DROP_TEMP_VIEW", -1),
        getattr(sqlite3, "SQLITE_DROP_TRIGGER", -1),
        getattr(sqlite3, "SQLITE_DROP_VIEW", -1),
        getattr(sqlite3, "SQLITE_INSERT", -1),
        getattr(sqlite3, "SQLITE_PRAGMA", -1),
        getattr(sqlite3, "SQLITE_REINDEX", -1),
        getattr(sqlite3, "SQLITE_SAVEPOINT", -1),
        getattr(sqlite3, "SQLITE_TRANSACTION", -1),
        getattr(sqlite3, "SQLITE_UPDATE", -1),
    }
    if action_code in dangerous:
        return sqlite3.SQLITE_DENY
    if action_code == getattr(sqlite3, "SQLITE_FUNCTION", -2):
        function_name = str(param1 or param2 or "").lower()
        if function_name == "load_extension":
            return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'
