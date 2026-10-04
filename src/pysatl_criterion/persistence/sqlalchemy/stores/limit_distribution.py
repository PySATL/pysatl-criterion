from collections.abc import Sequence
from typing import cast

from sqlalchemy import Table, and_, or_, select

from pysatl_criterion.persistence.models.distribution_key import DistributionKey
from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.persistence.queries.limit_distribution import LimitDistributionFilter
from pysatl_criterion.persistence.sqlalchemy.models.limit_distribution import (
    LimitDistributionORM,
    criterion_code_hash,
    criterion_parameters_hash,
)
from pysatl_criterion.persistence.sqlalchemy.stores._filters import distribution_filter_conditions
from pysatl_criterion.persistence.sqlalchemy.stores.base import AlchemyStorage
from pysatl_criterion.persistence.sqlalchemy.upsert import save_if_newer
from pysatl_criterion.persistence.stores.limit_distribution import ILimitDistributionStorage


class AlchemyLimitDistributionStorage(AlchemyStorage, ILimitDistributionStorage):
    def insert(self, model: LimitDistributionModel) -> bool:
        self._validate(model)
        orm = LimitDistributionORM.from_model(model)
        table = cast(Table, LimitDistributionORM.__table__)
        return save_if_newer(
            self._session,
            table,
            {column.name: getattr(orm, column.name) for column in table.columns},
        )

    def bulk_insert(self, models: Sequence[LimitDistributionModel]) -> list[LimitDistributionModel]:
        best: dict[tuple[str, str, int], LimitDistributionModel] = {}
        for model in models:
            self._validate(model)
            identity = (
                criterion_code_hash(model.criterion_code),
                criterion_parameters_hash(model.criterion_parameters),
                model.sample_size,
            )
            if identity not in best or model.monte_carlo_count > best[identity].monte_carlo_count:
                best[identity] = model
        # Acquire row locks in a consistent order across batch writers.
        return [best[key] for key in sorted(best) if self.insert(best[key])]

    def get(self, query: DistributionKey) -> LimitDistributionModel | None:
        result = self._session.execute(
            select(LimitDistributionORM)
            .where(
                LimitDistributionORM.criterion_code_hash
                == criterion_code_hash(query.criterion_code),
                LimitDistributionORM.criterion_parameters_hash
                == criterion_parameters_hash(query.criterion_parameters),
                LimitDistributionORM.sample_size == query.sample_size,
            )
            .execution_options(populate_existing=True)
        ).scalar_one_or_none()
        return result.to_model() if result is not None else None

    def get_batch(
        self,
        query: LimitDistributionFilter,
        *,
        batch_size: int = 100,
        after: DistributionKey | None = None,
    ) -> list[LimitDistributionModel]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        statement = select(LimitDistributionORM).where(
            *distribution_filter_conditions(LimitDistributionORM, query)
        )
        if after is not None:
            code_hash = criterion_code_hash(after.criterion_code)
            parameters_hash = criterion_parameters_hash(after.criterion_parameters)
            # Expand the composite comparison for dialects without row-value comparisons.
            statement = statement.where(
                or_(
                    LimitDistributionORM.criterion_code_hash > code_hash,
                    and_(
                        LimitDistributionORM.criterion_code_hash == code_hash,
                        LimitDistributionORM.criterion_parameters_hash > parameters_hash,
                    ),
                    and_(
                        LimitDistributionORM.criterion_code_hash == code_hash,
                        LimitDistributionORM.criterion_parameters_hash == parameters_hash,
                        LimitDistributionORM.sample_size > after.sample_size,
                    ),
                )
            )
        with self._session.execute(
            statement.order_by(
                LimitDistributionORM.criterion_code_hash,
                LimitDistributionORM.criterion_parameters_hash,
                LimitDistributionORM.sample_size,
            )
            .limit(batch_size)
            .execution_options(populate_existing=True)
        ) as result:
            return [row.to_model() for row in result.scalars()]

    @staticmethod
    def _validate(model: LimitDistributionModel) -> None:
        if model.monte_carlo_count <= 0:
            raise ValueError("monte_carlo_count must be positive")
        if not model.results_statistics:
            raise ValueError("results_statistics must not be empty")
