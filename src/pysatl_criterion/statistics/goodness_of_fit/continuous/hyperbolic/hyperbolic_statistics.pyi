# Type declarations maintained alongside the implementation.
from abc import ABC, ABCMeta

import numpy as np

from pysatl_criterion.distribution import HyperbolicDistributionDescriptor
from pysatl_criterion.distribution.parameters import HypothesisSupport, ParameterValues
from pysatl_criterion.statistics.goodness_of_fit import edf, frequency
from pysatl_criterion.statistics.goodness_of_fit.binning import BinningStrategy
from pysatl_criterion.statistics.goodness_of_fit.continuous._edf_factory import EDFSpec
from pysatl_criterion.statistics.goodness_of_fit.continuous._frequency_factory import FrequencySpec
from pysatl_criterion.statistics.goodness_of_fit.goodness_of_fit_statistic import (
    AbstractGoodnessOfFitStatistic,
)

_EDF_STATISTICS: dict[str, EDFSpec]
_FREQUENCY_STATISTICS: dict[str, FrequencySpec]

class AbstractHyperbolicGofStatistic(AbstractGoodnessOfFitStatistic, ABC, metaclass=ABCMeta):
    @property
    def alpha(self) -> float: ...
    @property
    def beta(self) -> float: ...
    @property
    def delta(self) -> float: ...
    @property
    def mu(self) -> float: ...
    def __init__(self, parameters: ParameterValues) -> None: ...
    def __getstate__(self): ...
    def __setstate__(self, state): ...
    @staticmethod
    def distribution() -> type[HyperbolicDistributionDescriptor]: ...
    @classmethod
    def code(cls) -> str: ...
    def _cdf(self, sorted_rvs: np.ndarray) -> np.ndarray: ...
    def _log_probabilities(self, sorted_rvs: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

class KolmogorovSmirnovHyperbolicGofStatistic(
    AbstractHyperbolicGofStatistic, edf.AbstractKSStatistic
):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs, **kwargs) -> float: ...

class CramerVonMisesHyperbolicGofStatistic(
    AbstractHyperbolicGofStatistic, edf.AbstractCramerVonMisesStatistic
):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs, **kwargs) -> float: ...

class AndersonDarlingHyperbolicGofStatistic(
    AbstractHyperbolicGofStatistic, edf.AbstractAndersonDarlingStatistic
):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs, **kwargs) -> float: ...

class KuiperHyperbolicGofStatistic(AbstractHyperbolicGofStatistic, edf.AbstractKuiperStatistic):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs, **kwargs) -> float: ...

class LiaoShimokawaHyperbolicGofStatistic(
    AbstractHyperbolicGofStatistic, edf.AbstractLiaoShimokawaStatistic
):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs, **kwargs) -> float: ...

class WatsonHyperbolicGofStatistic(AbstractHyperbolicGofStatistic, edf.AbstractWatsonStatistic):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs, **kwargs) -> float: ...

class Chi2PearsonHyperbolicGofStatistic(
    AbstractHyperbolicGofStatistic, frequency.AbstractPearsonChi2Statistic
):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs, **kwargs) -> float: ...
    binning: BinningStrategy
    def __init__(
        self, parameters: ParameterValues, *, binning: BinningStrategy | None = None
    ) -> None: ...
    def configuration(self) -> dict: ...

class FreemanTukeyHyperbolicGofStatistic(
    AbstractHyperbolicGofStatistic, frequency.AbstractFreemanTukeyStatistic
):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs, **kwargs) -> float: ...
    binning: BinningStrategy
    def __init__(
        self, parameters: ParameterValues, *, binning: BinningStrategy | None = None
    ) -> None: ...
    def configuration(self) -> dict: ...

class ModifiedLogLikelihoodHyperbolicGofStatistic(
    AbstractHyperbolicGofStatistic, frequency.AbstractModifiedLogLikelihoodStatistic
):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs, **kwargs) -> float: ...
    binning: BinningStrategy
    def __init__(
        self, parameters: ParameterValues, *, binning: BinningStrategy | None = None
    ) -> None: ...
    def configuration(self) -> dict: ...

class NeymanChi2HyperbolicGofStatistic(
    AbstractHyperbolicGofStatistic, frequency.AbstractNeymanChi2Statistic
):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs, **kwargs) -> float: ...
    binning: BinningStrategy
    def __init__(
        self, parameters: ParameterValues, *, binning: BinningStrategy | None = None
    ) -> None: ...
    def configuration(self) -> dict: ...

class CressieReadHyperbolicGofStatistic(
    AbstractHyperbolicGofStatistic, frequency.AbstractCressieReadStatistic
):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs, **kwargs) -> float: ...
    binning: BinningStrategy
    def __init__(
        self, parameters: ParameterValues, *, binning: BinningStrategy | None = None
    ) -> None: ...
    def configuration(self) -> dict: ...

__all__ = [
    "AbstractHyperbolicGofStatistic",
    "AndersonDarlingHyperbolicGofStatistic",
    "Chi2PearsonHyperbolicGofStatistic",
    "CramerVonMisesHyperbolicGofStatistic",
    "CressieReadHyperbolicGofStatistic",
    "FreemanTukeyHyperbolicGofStatistic",
    "KolmogorovSmirnovHyperbolicGofStatistic",
    "KuiperHyperbolicGofStatistic",
    "LiaoShimokawaHyperbolicGofStatistic",
    "ModifiedLogLikelihoodHyperbolicGofStatistic",
    "NeymanChi2HyperbolicGofStatistic",
    "WatsonHyperbolicGofStatistic",
]
