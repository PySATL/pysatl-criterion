from pysatl_criterion.loader.abstract_loader import AbstractLoader
from pysatl_criterion.persistence.models.critical_value import CriticalValueModel
from pysatl_criterion.persistence.queries.critical_value import (
    CriticalValueFilter,
    CriticalValueLookupQuery,
)


class CriticalValueLoader(
    AbstractLoader[CriticalValueModel, CriticalValueLookupQuery, CriticalValueFilter]
):
    """Transfer cached bounds whose versions match the destination distributions."""

    @staticmethod
    def _criterion_code(model: CriticalValueModel) -> str:
        return model.distribution_key.criterion_code
