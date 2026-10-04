from sqlalchemy import and_, false, or_
from sqlalchemy.sql.elements import ColumnElement

from pysatl_criterion.persistence.queries.limit_distribution import LimitDistributionFilter
from pysatl_criterion.persistence.sqlalchemy.models.critical_value import CriticalValueORM
from pysatl_criterion.persistence.sqlalchemy.models.limit_distribution import (
    LimitDistributionORM,
    criterion_code_hash,
    criterion_parameters_hash,
)


def distribution_filter_conditions(
    model: type[LimitDistributionORM] | type[CriticalValueORM], query: LimitDistributionFilter
) -> list[ColumnElement[bool]]:
    """Apply the same criterion pairs and bounds to distributions and cached values."""
    conditions: list[ColumnElement[bool]] = []
    if query.criteria is not None:
        criteria = []
        for criterion in query.criteria:
            condition = model.criterion_code_hash == criterion_code_hash(criterion.criterion_code)
            if criterion.criterion_parameters is not None:
                condition = and_(
                    condition,
                    model.criterion_parameters_hash
                    == criterion_parameters_hash(criterion.criterion_parameters),
                )
            criteria.append(condition)
        conditions.append(or_(false(), *criteria))
    if query.min_sample_size is not None:
        conditions.append(model.sample_size >= query.min_sample_size)
    if query.max_sample_size is not None:
        conditions.append(model.sample_size <= query.max_sample_size)
    if query.min_monte_carlo_count is not None:
        conditions.append(model.monte_carlo_count >= query.min_monte_carlo_count)
    return conditions
