from __future__ import annotations

import sqlite3

from sim.review.query_compatibility import compatible_review_query


def test_legacy_diagnostic_scalar_query_routes_to_normalized_fields() -> None:
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE fsw_diagnostic_fields (object_id TEXT, generated_time_ns INTEGER, "
            "field_name TEXT, value_real REAL, value_integer INTEGER, value_text TEXT)"
        )
        connection.execute(
            "INSERT INTO fsw_diagnostic_fields VALUES ('sat', 10, 'attitude_error_rad', 0.01, NULL, NULL)"
        )
        legacy = (
            "SELECT CAST(json_extract(field.value, '$.value') AS REAL) "
            "FROM fsw_diagnostics AS d, json_each(json_extract(d.detail_json, '$.fields')) AS field "
            "WHERE json_extract(field.value, '$.name') = 'attitude_error_rad' "
            "ORDER BY d.generated_time_ns DESC LIMIT 1"
        )
        query, mode = compatible_review_query(connection, legacy)

        assert mode == "normalized_diagnostic"
        assert connection.execute(query).fetchone() == (0.01,)


def test_legacy_diagnostic_count_preserves_object_time_and_text_predicates() -> None:
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE fsw_diagnostic_fields (object_id TEXT, generated_time_ns INTEGER, "
            "field_name TEXT, value_real REAL, value_integer INTEGER, value_text TEXT)"
        )
        connection.executemany(
            "INSERT INTO fsw_diagnostic_fields VALUES (?, ?, ?, NULL, NULL, ?)",
            [("inspector", 4, "executive_phase", "recovery"), ("inspector", 8, "executive_phase", "primary")],
        )
        legacy = (
            "SELECT COUNT(*) FROM fsw_diagnostics AS d, json_each(json_extract(d.detail_json, '$.fields')) AS field "
            "WHERE d.object_id = 'inspector' AND d.generated_time_ns > 5 "
            "AND json_extract(field.value, '$.name') = 'executive_phase' "
            "AND json_extract(field.value, '$.value') = 'primary'"
        )
        query, mode = compatible_review_query(connection, legacy)

        assert mode == "normalized_diagnostic"
        assert connection.execute(query).fetchone() == (1,)


def test_receipt_query_uses_json_for_legacy_store_and_column_for_normalized_store() -> None:
    legacy = (
        "SELECT COUNT(*) FROM actuator_command_receipts "
        "WHERE json_extract(detail_json, '$.disposition') = 'accepted'"
    )
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE actuator_command_receipts (disposition TEXT, detail_json TEXT)")
        connection.execute("INSERT INTO actuator_command_receipts VALUES (NULL, '{\"disposition\":\"accepted\"}')")
        query, mode = compatible_review_query(connection, legacy)
        assert mode == "legacy"
        assert connection.execute(query).fetchone() == (1,)

        connection.execute("CREATE TABLE fsw_diagnostic_fields (field_name TEXT)")
        query, mode = compatible_review_query(connection, legacy)
        assert mode == "normalized_receipt"
        assert connection.execute(query).fetchone() == (0,)
