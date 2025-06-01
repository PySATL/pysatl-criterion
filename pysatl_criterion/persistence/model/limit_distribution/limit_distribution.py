from dataclasses import dataclass
from typing import Protocol

from pysatl_criterion.persistence.model.common.data_storage.data_storage import (
    DataModel,
    DataQuery,
    IDataStorage,
)


@dataclass
class LimitDistributionModel(DataModel):
    experiment_id: int
    criterion_code: str
    criterion_parameters: list[float]
    sample_size: int
    monte_carlo_count: int
    results_statistics: list[float]


@dataclass
class LimitDistributionQuery(DataQuery):
    criterion_code: str
    criterion_parameters: list[float]
    sample_size: int
    monte_carlo_count: int


class ILimitDistributionStorage(
    IDataStorage[LimitDistributionModel, LimitDistributionQuery], Protocol
):
    """
    Limit distribution storage interface.
    """

    pass
