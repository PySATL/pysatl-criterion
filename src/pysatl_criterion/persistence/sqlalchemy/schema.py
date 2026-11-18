"""Validate the calibration storage schema before opening existing tables."""

from typing import cast

from sqlalchemy import Table, inspect
from sqlalchemy.engine import Engine

from pysatl_criterion.persistence.sqlalchemy.models.critical_value import CriticalValueORM
from pysatl_criterion.persistence.sqlalchemy.models.limit_distribution import LimitDistributionORM


def require_configuration_schema(engine: Engine) -> None:
    """Fail before using legacy tables; create_all cannot update primary keys."""
    inspector = inspect(engine)
    for model in (LimitDistributionORM, CriticalValueORM):
        table = cast(Table, model.__table__)
        name = table.name
        if inspector.has_table(name):
            columns = {column["name"] for column in inspector.get_columns(name)}
            keys = inspector.get_pk_constraint(name)["constrained_columns"]
            if not {column.name for column in table.columns} <= columns:
                raise ValueError(
                    "Incompatible calibration schema: "
                    "use a new database with configuration-aware keys."
                )
            if keys != [column.name for column in table.primary_key]:
                raise ValueError(f"Incompatible calibration primary key for {name}")
    if inspector.has_table("critical_values"):
        expected = [
            "criterion_code_hash",
            "criterion_parameters_hash",
            "criterion_configuration_hash",
            "sample_size",
        ]
        if not any(
            foreign_key["referred_table"] == "limit_distributions"
            and foreign_key["constrained_columns"] == expected
            and foreign_key["referred_columns"] == expected
            and (foreign_key.get("options", {}).get("ondelete") or "").upper() == "CASCADE"
            for foreign_key in inspector.get_foreign_keys("critical_values")
        ):
            raise ValueError(
                "Calibration schema requires a configuration-aware distribution foreign key"
            )
