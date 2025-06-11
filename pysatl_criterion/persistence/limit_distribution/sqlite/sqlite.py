import json
import sqlite3
from pathlib import Path
from sqlite3 import Connection

from pysatl_criterion.persistence.model.limit_distribution.limit_distribution import (
    CriticalValueQuery,
    ILimitDistributionStorage,
    LimitDistributionModel,
    LimitDistributionQuery,
)


class SQLiteLimitDistributionStorage(ILimitDistributionStorage):
    """
    SQLite implementation for limit distribution storage.
    """

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.conn: None | Connection = None

    def init(self) -> None:
        """Initialize the database and create tables."""
        db_path = Path(self.connection_string)
        db_dir = db_path.parent
        if not db_dir.exists():
            db_dir.mkdir(parents=True)
        self.conn = sqlite3.connect(self.connection_string)
        self._create_tables()

    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        with self.conn:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS limit_distributions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                criterion_code TEXT NOT NULL,
                criterion_parameters TEXT NOT NULL,
                sample_size INTEGER NOT NULL,
                monte_carlo_count INTEGER NOT NULL,
                results_statistics TEXT NOT NULL,
                UNIQUE(
                experiment_id, criterion_code, criterion_parameters, sample_size, monte_carlo_count
                )
            )
            """)

            # Index for critical value queries
            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cv_queries
            ON limit_distributions (criterion_code, sample_size)
            """)

    def _get_connection(self):
        """Get database connection, ensuring it's initialized."""
        if self.conn is None:
            raise RuntimeError("Storage not initialized. Call init() first.")
        return self.conn

    def _model_to_row(self, model: LimitDistributionModel) -> dict:
        """Convert LimitDistributionModel to database row format."""
        return {
            "experiment_id": model.experiment_id,
            "criterion_code": model.criterion_code,
            "criterion_parameters": json.dumps(model.criterion_parameters),
            "sample_size": model.sample_size,
            "monte_carlo_count": model.monte_carlo_count,
            "results_statistics": json.dumps(model.results_statistics),
        }

    def _row_to_model(self, row: dict) -> LimitDistributionModel:
        """Convert database row to LimitDistributionModel."""
        return LimitDistributionModel(
            experiment_id=row["experiment_id"],
            criterion_code=row["criterion_code"],
            criterion_parameters=json.loads(row["criterion_parameters"]),
            sample_size=row["sample_size"],
            monte_carlo_count=row["monte_carlo_count"],
            results_statistics=json.loads(row["results_statistics"]),
        )

    def insert_data(self, data: LimitDistributionModel) -> None:
        """Insert or update limit distribution data."""
        conn = self._get_connection()
        row_data = self._model_to_row(data)

        columns = ", ".join(row_data.keys())
        placeholders = ", ".join("?" * len(row_data))
        values = list(row_data.values())

        with conn:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO limit_distributions ({columns})
                VALUES ({placeholders})
            """,
                values,
            )

    def get_data(self, query: LimitDistributionQuery) -> LimitDistributionModel | None:
        """Get specific limit distribution data."""
        conn = self._get_connection()
        params_json = json.dumps(query.criterion_parameters)

        cursor = conn.execute(
            """
            SELECT * FROM limit_distributions
            WHERE criterion_code = ?
            AND criterion_parameters = ?
            AND sample_size = ?
            AND monte_carlo_count = ?
        """,
            (query.criterion_code, params_json, query.sample_size, query.monte_carlo_count),
        )

        row = cursor.fetchone()
        if not row:
            return None

        columns = [col[0] for col in cursor.description]
        return self._row_to_model(dict(zip(columns, row)))

    def delete_data(self, query: LimitDistributionQuery) -> None:
        """Delete specific limit distribution data."""
        conn = self._get_connection()
        params_json = json.dumps(query.criterion_parameters)

        with conn:
            conn.execute(
                """
                DELETE FROM limit_distributions
                WHERE criterion_code = ?
                AND criterion_parameters = ?
                AND sample_size = ?
                AND monte_carlo_count = ?
            """,
                (query.criterion_code, params_json, query.sample_size, query.monte_carlo_count),
            )

    def get_data_for_cv(self, query: CriticalValueQuery) -> LimitDistributionModel | None:
        """
        Get limit distribution data for critical value calculation.

        Returns the most complete dataset (highest monte_carlo_count)
        for the criterion and sample size.
        """
        conn = self._get_connection()

        cursor = conn.execute(
            """
            SELECT * FROM limit_distributions
            WHERE criterion_code = ?
            AND sample_size = ?
            ORDER BY monte_carlo_count DESC
            LIMIT 1
        """,
            (query.criterion_code, query.sample_size),
        )

        row = cursor.fetchone()
        if not row:
            return None

        columns = [col[0] for col in cursor.description]
        return self._row_to_model(dict(zip(columns, row)))
