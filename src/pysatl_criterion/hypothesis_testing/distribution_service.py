from collections.abc import Callable, Sequence
from decimal import Decimal

from pysatl_criterion.hypothesis_testing.critical_values.calculator import (
    critical_value_calculator as calculators,
)
from pysatl_criterion.persistence.models.critical_value import (
    CriticalValueModel,
    normalize_significance_level,
)
from pysatl_criterion.persistence.models.distribution_key import DistributionKey
from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.persistence.queries.critical_value import CriticalValueLookupQuery
from pysatl_criterion.persistence.unit_of_work import IUnitOfWork
from pysatl_criterion.statistics.alternative import AlternativeType


class DistributionNotFoundError(LookupError):
    """No distribution exists for the requested identity."""


class DistributionChangedError(RuntimeError):
    """The source kept changing while critical values were being calculated."""


class DistributionService:
    def __init__(self, uow_factory: Callable[[], IUnitOfWork], max_attempts: int = 3):
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._uow_factory = uow_factory
        self._max_attempts = max_attempts

    def save_distribution(self, data: LimitDistributionModel) -> bool:
        """Save an improvement and invalidate all derived values atomically."""
        with self._uow_factory() as uow:
            changed = uow.limit_distributions.insert(data)
            if changed:
                uow.critical_values.delete_for_distribution(data.key)
            uow.commit()
            return changed

    def save_distributions(
        self, models: Sequence[LimitDistributionModel]
    ) -> list[LimitDistributionModel]:
        """Save a batch and invalidate only changed identities in one transaction."""
        with self._uow_factory() as uow:
            saved = uow.limit_distributions.bulk_insert(models)
            for model in saved:
                uow.critical_values.delete_for_distribution(model.key)
            uow.commit()
            return saved

    def get_critical_value(
        self,
        key: DistributionKey,
        significance_level: Decimal,
        alternative_type: AlternativeType,
    ) -> CriticalValueModel:
        """Read or compute bounds, retrying if their source changed during calculation."""
        level = normalize_significance_level(significance_level)
        if not isinstance(alternative_type, AlternativeType):
            raise ValueError("alternative_type must be an AlternativeType")
        for _ in range(self._max_attempts):
            with self._uow_factory() as uow:
                distribution = uow.limit_distributions.get(key)
                if distribution is None:
                    raise DistributionNotFoundError(f"Distribution not found: {key}")
                query = CriticalValueLookupQuery(
                    key, distribution.monte_carlo_count, level, alternative_type
                )
                cached = uow.critical_values.get(query)
                if cached is not None:
                    return cached

            # No session or database lock is held during the calculation.
            calculated = self._calculate(distribution, level, alternative_type)

            with self._uow_factory() as uow:
                if uow.critical_values.save_if_current(calculated):
                    # A concurrent calculator may have populated the same cache entry.
                    saved = uow.critical_values.get(query)
                    if saved is not None:
                        uow.commit()
                        return saved
        raise DistributionChangedError(
            f"Distribution changed during {self._max_attempts} calculation attempts: {key}"
        )

    @staticmethod
    def _calculate(
        distribution: LimitDistributionModel,
        significance_level: Decimal,
        alternative_type: AlternativeType,
    ) -> CriticalValueModel:
        lower: float | None = None
        upper: float | None = None
        level = float(significance_level)
        if alternative_type == AlternativeType.LEFT:
            lower = calculators.LeftCriticalValueCalculator().calculate(
                distribution.results_statistics, level
            )
        elif alternative_type == AlternativeType.RIGHT:
            upper = calculators.RightCriticalValueCalculator().calculate(
                distribution.results_statistics, level
            )
        else:
            lower, upper = calculators.TwoSidedCriticalValueCalculator().calculate(
                distribution.results_statistics, level
            )
        return CriticalValueModel(
            distribution.key,
            distribution.monte_carlo_count,
            significance_level,
            alternative_type,
            lower,
            upper,
        )
