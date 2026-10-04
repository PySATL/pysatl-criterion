from collections.abc import Mapping
from typing import Any

from sqlalchemy import Table, and_, insert, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session


def matched_rows(result: CursorResult[Any]) -> int:
    """Require reliable single-statement row counts rather than guess whether a write won."""
    if not result.supports_sane_rowcount() or result.rowcount < 0:
        raise RuntimeError("The database driver must report matched rows for UPDATE statements")
    return result.rowcount


def save_if_newer(
    session: Session,
    table: Table,
    values: Mapping[str, Any],
    version_column: str = "monte_carlo_count",
) -> bool:
    """Conditionally update or insert through portable SQLAlchemy Core operations.

    A savepoint contains a concurrent insert conflict without rolling back the caller's work.
    Other integrity errors propagate. Serialization failures/deadlocks must be retried by the
    caller as a whole transaction, not by repeating a statement inside a failed transaction.
    """
    key_names = {column.name for column in table.primary_key.columns}
    identity = and_(*(table.c[name] == values[name] for name in key_names))
    version = table.c[version_column]
    incoming_version = values[version_column]
    statement = (
        update(table)
        .where(identity, version < incoming_version)
        .values(**{name: value for name, value in values.items() if name not in key_names})
    )
    connection = session.connection()
    if matched_rows(connection.execute(statement)):
        return True
    current_version = connection.scalar(select(version).where(identity))
    if current_version is not None:
        if current_version < incoming_version:
            # A smaller version may have been inserted between UPDATE and SELECT.
            return matched_rows(connection.execute(statement)) == 1
        return False

    try:
        with session.begin_nested():
            # Acquire through Session so its lazy nested transaction actually emits SAVEPOINT.
            session.execute(insert(table).values(**values))
        return True
    except IntegrityError:
        # A competing writer may have inserted this same identity after our SELECT.
        if matched_rows(connection.execute(statement)):
            return True
        current_version = connection.scalar(select(version).where(identity))
        if current_version is not None and current_version >= incoming_version:
            return False
        # Do not hide FK/CHECK/NOT NULL errors or conflicts not visible in this snapshot.
        raise
