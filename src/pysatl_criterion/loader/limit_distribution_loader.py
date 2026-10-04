from pysatl_criterion.loader.abstract_loader import AbstractLoader
from pysatl_criterion.persistence.models.distribution_key import DistributionKey
from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.persistence.queries.limit_distribution import LimitDistributionFilter


class LimitDistributionLoader(
    AbstractLoader[LimitDistributionModel, DistributionKey, LimitDistributionFilter]
):
    """Transfer limit distributions through a reader and a transactional writer."""

    @staticmethod
    def _criterion_code(model: LimitDistributionModel) -> str:
        return model.criterion_code
