from collections.abc import Sequence
from decimal import Decimal
from typing import cast

from sqlalchemy import Table, and_, delete, or_, select, update
from sqlalchemy.sql.elements import ColumnElement

from pysatl_criterion.persistence.models.critical_value import CriticalValueKey, CriticalValueModel
from pysatl_criterion.persistence.models.distribution_key import DistributionKey
from pysatl_criterion.persistence.queries.critical_value import (
    CriticalValueFilter,
    CriticalValueLookupQuery,
)
from pysatl_criterion.persistence.sqlalchemy.models.critical_value import CriticalValueORM
from pysatl_criterion.persistence.sqlalchemy.models.limit_distribution import (
    LimitDistributionORM,
    criterion_code_hash,
    criterion_parameters_hash,
)
from pysatl_criterion.persistence.sqlalchemy.stores._filters import distribution_filter_conditions
from pysatl_criterion.persistence.sqlalchemy.stores.base import AlchemyStorage
from pysatl_criterion.persistence.sqlalchemy.upsert import matched_rows, save_if_newer
from pysatl_criterion.persistence.stores.critical_value import ICriticalValueStorage


class AlchemyCriticalValueStorage(AlchemyStorage, ICriticalValueStorage):
    def get(self, query: CriticalValueLookupQuery) -> CriticalValueModel | None:
        key = query.distribution_key
        session = self._session
        statement = (
            select(CriticalValueORM)
            .join(LimitDistributionORM)
            .where(
                CriticalValueORM.criterion_code_hash == criterion_code_hash(key.criterion_code),
                CriticalValueORM.criterion_parameters_hash
                == criterion_parameters_hash(key.criterion_parameters),
                CriticalValueORM.sample_size == key.sample_size,
                CriticalValueORM.significance_level == query.significance_level,
                CriticalValueORM.alternative_type == query.alternative_type,
                CriticalValueORM.monte_carlo_count == query.monte_carlo_count,
                LimitDistributionORM.monte_carlo_count == query.monte_carlo_count,
            )
            .execution_options(populate_existing=True)
        )
        result = session.execute(statement).scalar_one_or_none()
        return result.to_model() if result is not None else None

    def save_if_current(self, data: CriticalValueModel) -> bool:
        # An unchanged cache entry is still valid for a concurrent calculator.
        return self._save_if_current(data) is not None

    def insert(self, model: CriticalValueModel) -> bool:
        return self._save_if_current(model) is True

    def bulk_insert(self, models: Sequence[CriticalValueModel]) -> list[CriticalValueModel]:
        best: dict[tuple[str, str, int, Decimal, str], CriticalValueModel] = {}
        for model in models:
            key = model.distribution_key
            identity = (
                criterion_code_hash(key.criterion_code),
                criterion_parameters_hash(key.criterion_parameters),
                key.sample_size,
                model.significance_level,
                model.alternative_type.value,
            )
            if identity not in best or model.monte_carlo_count > best[identity].monte_carlo_count:
                best[identity] = model
        return [best[key] for key in sorted(best) if self.insert(best[key])]

    def get_batch(
        self,
        query: CriticalValueFilter,
        *,
        batch_size: int = 100,
        after: CriticalValueKey | None = None,
    ) -> list[CriticalValueModel]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        statement = (
            select(CriticalValueORM)
            .join(LimitDistributionORM)
            .where(
                CriticalValueORM.monte_carlo_count == LimitDistributionORM.monte_carlo_count,
                *distribution_filter_conditions(CriticalValueORM, query),
            )
        )
        if query.significance_levels is not None:
            statement = statement.where(
                CriticalValueORM.significance_level.in_(query.significance_levels)
            )
        if query.alternative_types is not None:
            statement = statement.where(
                CriticalValueORM.alternative_type.in_(query.alternative_types)
            )
        if after is not None:
            statement = statement.where(self._after_key(after))
        results = self._session.execute(
            statement.order_by(
                CriticalValueORM.criterion_code_hash,
                CriticalValueORM.criterion_parameters_hash,
                CriticalValueORM.sample_size,
                CriticalValueORM.significance_level,
                CriticalValueORM.alternative_type,
            )
            .limit(batch_size)
            .execution_options(populate_existing=True)
        ).scalars()
        return [result.to_model() for result in results]

    @staticmethod
    def _after_key(after: CriticalValueKey) -> ColumnElement[bool]:
        key = after.distribution_key
        cursor = [
            (CriticalValueORM.criterion_code_hash, criterion_code_hash(key.criterion_code)),
            (
                CriticalValueORM.criterion_parameters_hash,
                criterion_parameters_hash(key.criterion_parameters),
            ),
            (CriticalValueORM.sample_size, key.sample_size),
            (CriticalValueORM.significance_level, after.significance_level),
            (CriticalValueORM.alternative_type, after.alternative_type),
        ]
        return or_(
            *(
                and_(
                    *(column == value for column, value in cursor[:index]),
                    column > value,
                )
                for index, (column, value) in enumerate(cursor)
            )
        )

    def _save_if_current(self, data: CriticalValueModel) -> bool | None:
        """Return whether bounds changed, or None when the local source does not match."""
        key = data.distribution_key
        session = self._session
        source = cast(Table, LimitDistributionORM.__table__)
        # A no-op conditional UPDATE locks the source until commit without relying
        # on SELECT FOR UPDATE, which is not implemented by every dialect.
        locked = session.connection().execute(
            update(source)
            .where(
                source.c.criterion_code_hash == criterion_code_hash(key.criterion_code),
                source.c.criterion_parameters_hash
                == criterion_parameters_hash(key.criterion_parameters),
                source.c.sample_size == key.sample_size,
                source.c.monte_carlo_count == data.monte_carlo_count,
            )
            .values(monte_carlo_count=source.c.monte_carlo_count)
        )
        if matched_rows(locked) != 1:
            return None

        orm = CriticalValueORM.from_model(data)
        table = cast(Table, CriticalValueORM.__table__)
        return save_if_newer(
            session,
            table,
            {column.name: getattr(orm, column.name) for column in table.columns},
        )

    def delete_for_distribution(self, key: DistributionKey) -> None:
        session = self._session
        session.execute(
            delete(CriticalValueORM).where(
                CriticalValueORM.criterion_code_hash == criterion_code_hash(key.criterion_code),
                CriticalValueORM.criterion_parameters_hash
                == criterion_parameters_hash(key.criterion_parameters),
                CriticalValueORM.sample_size == key.sample_size,
            )
        )
