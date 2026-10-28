from __future__ import annotations

import logging
import os
from typing import Any

import psycopg
from psycopg.conninfo import conninfo_to_dict

logger = logging.getLogger(__name__)
ENVIRONMENT_VARIABLE = "CARE_PREDICT_DATABASE_URL"
RETENTION_DAYS = 90


def get_connection_url() -> str | None:
    return os.environ.get(ENVIRONMENT_VARIABLE)


def connect() -> psycopg.Connection[Any]:
    connection_url = get_connection_url()
    if not connection_url:
        raise RuntimeError(f"Set {ENVIRONMENT_VARIABLE} to enable PostgreSQL history.")
    parameters = conninfo_to_dict(connection_url)
    host = parameters.get("host", "localhost").split(",")[0].strip("[]").casefold()
    sslmode = parameters.get("sslmode", "prefer").casefold()
    if host not in {"localhost", "127.0.0.1", "::1"} and sslmode != "verify-full":
        raise ValueError(
            "Remote PostgreSQL connections must set sslmode=verify-full. "
            "Use sslrootcert to point at the trusted CA certificate."
        )
    return psycopg.connect(connection_url, connect_timeout=5)


def record_event(
    *,
    model_version: str,
    prediction_source: str,
    total_count: int,
    low_count: int,
    medium_count: int,
    high_count: int,
) -> None:
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO prediction_audit (
                model_version, prediction_source, total_count,
                low_count, medium_count, high_count
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                model_version,
                prediction_source,
                total_count,
                low_count,
                medium_count,
                high_count,
            ),
        )
        connection.execute(
            "DELETE FROM prediction_audit "
            "WHERE created_at < NOW() - (%s * INTERVAL '1 day')",
            (RETENTION_DAYS,),
        )


def recent_events(limit: int = 500) -> list[dict[str, Any]]:
    if not 1 <= limit <= 5000:
        raise ValueError("History limit must be between 1 and 5,000.")
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT created_at, model_version, prediction_source, total_count,
                   low_count, medium_count, high_count
            FROM prediction_audit
            WHERE created_at >= NOW() - (%s * INTERVAL '1 day')
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (RETENTION_DAYS, limit),
        ).fetchall()
    return [
        {
            "Prediction Date (UTC)": row[0].isoformat(),
            "Model Version": row[1],
            "Source": row[2],
            "Total": row[3],
            "Low": row[4],
            "Medium": row[5],
            "High": row[6],
        }
        for row in rows
    ]
