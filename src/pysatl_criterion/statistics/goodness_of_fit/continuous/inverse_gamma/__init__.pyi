"""Static API for inverse-gamma criteria, including factory-generated classes."""

from abc import ABC
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pysatl_criterion.distribution import InverseGammaDistributionDescriptor
from pysatl_criterion.distribution.parameters import HypothesisSupport, ParameterValues
from pysatl_criterion.statistics.alternative import Alternative, AlternativeType
from pysatl_criterion.statistics.goodness_of_fit.binning import BinningStrategy
from pysatl_criterion.statistics.goodness_of_fit.continuous import edf, frequency
from pysatl_criterion.statistics.goodness_of_fit.goodness_of_fit_statistic import (
    AbstractGoodnessOfFitStatistic,
)

class AbstractInverseGammaGofStatistic(AbstractGoodnessOfFitStatistic, ABC):
    @property
    def alpha(self) -> float: ...
    @property
    def beta(self) -> float: ...
    @staticmethod
    def _prepare_sample(rvs: ArrayLike, minimum: int = 1) -> NDArray[np.float64]: ...
    _validate_input = _prepare_sample
    def _cdf(self, sample: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def _log_probabilities(
        self, sample: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    @staticmethod
    def distribution() -> type[InverseGammaDistributionDescriptor]: ...
    @classmethod
    def code(cls) -> str: ...

class GreenwoodInverseGammaGofStatistic(AbstractInverseGammaGofStatistic):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def alternative(self) -> Alternative: ...
    @staticmethod
    def short_code() -> str: ...
    def execute_statistic(self, rvs: ArrayLike, **kwargs: Any) -> float: ...
    @staticmethod
    def do_execute_statistic(cdf_vals: NDArray[np.float64]) -> float: ...

class KolmogorovSmirnovInverseGammaGofStatistic(
    AbstractInverseGammaGofStatistic, edf.AbstractKSStatistic
):
    alternative_type: AlternativeType
    mode: str
    def __init__(
        self,
        parameters: ParameterValues,
        *,
        alternative_type: AlternativeType = AlternativeType.TWO_TAILED,
        mode: str = "auto",
    ) -> None: ...
    def _validate_storage_calibration(self) -> None: ...
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs: ArrayLike, **kwargs: Any) -> float: ...

class KolmogorovSmirnovDPlusInverseGammaGofStatistic(
    AbstractInverseGammaGofStatistic, edf.AbstractKSDPlusStatistic
):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs: ArrayLike, **kwargs: Any) -> float: ...

class KolmogorovSmirnovDMinusInverseGammaGofStatistic(
    AbstractInverseGammaGofStatistic, edf.AbstractKSDMinusStatistic
):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs: ArrayLike, **kwargs: Any) -> float: ...

class AndersonDarlingInverseGammaGofStatistic(
    AbstractInverseGammaGofStatistic, edf.AbstractAndersonDarlingStatistic
):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs: ArrayLike, **kwargs: Any) -> float: ...

class CramerVonMisesInverseGammaGofStatistic(
    AbstractInverseGammaGofStatistic, edf.AbstractCramerVonMisesStatistic
):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs: ArrayLike, **kwargs: Any) -> float: ...

class WatsonInverseGammaGofStatistic(AbstractInverseGammaGofStatistic, edf.AbstractWatsonStatistic):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs: ArrayLike, **kwargs: Any) -> float: ...

class KuiperInverseGammaGofStatistic(AbstractInverseGammaGofStatistic, edf.AbstractKuiperStatistic):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs: ArrayLike, **kwargs: Any) -> float: ...

class KuiperScaledInverseGammaGofStatistic(
    AbstractInverseGammaGofStatistic, edf.AbstractKuiperScaledStatistic
):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs: ArrayLike, **kwargs: Any) -> float: ...

class KuiperCorrectedInverseGammaGofStatistic(
    AbstractInverseGammaGofStatistic, edf.AbstractKuiperCorrectedStatistic
):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs: ArrayLike, **kwargs: Any) -> float: ...

class LiaoShimokawaInverseGammaGofStatistic(
    AbstractInverseGammaGofStatistic, edf.AbstractLiaoShimokawaStatistic
):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs: ArrayLike, **kwargs: Any) -> float: ...

class ZhangAInverseGammaGofStatistic(AbstractInverseGammaGofStatistic, edf.AbstractZhangAStatistic):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs: ArrayLike, **kwargs: Any) -> float: ...

class ZhangCInverseGammaGofStatistic(AbstractInverseGammaGofStatistic, edf.AbstractZhangCStatistic):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs: ArrayLike, **kwargs: Any) -> float: ...

class ZhangKInverseGammaGofStatistic(AbstractInverseGammaGofStatistic, edf.AbstractZhangKStatistic):
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs: ArrayLike, **kwargs: Any) -> float: ...

class Chi2PearsonInverseGammaGofStatistic(
    AbstractInverseGammaGofStatistic, frequency.AbstractPearsonChi2Statistic
):
    binning: BinningStrategy
    def __init__(
        self, parameters: ParameterValues, *, binning: BinningStrategy | None = None
    ) -> None: ...
    def configuration(self) -> dict: ...
    def _validate_storage_calibration(self) -> None: ...
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs: ArrayLike, **kwargs: Any) -> float: ...

class FreemanTukeyInverseGammaGofStatistic(
    AbstractInverseGammaGofStatistic, frequency.AbstractFreemanTukeyStatistic
):
    binning: BinningStrategy
    def __init__(
        self, parameters: ParameterValues, *, binning: BinningStrategy | None = None
    ) -> None: ...
    def configuration(self) -> dict: ...
    def _validate_storage_calibration(self) -> None: ...
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs: ArrayLike, **kwargs: Any) -> float: ...

class ModifiedLogLikelihoodInverseGammaGofStatistic(
    AbstractInverseGammaGofStatistic, frequency.AbstractModifiedLogLikelihoodStatistic
):
    binning: BinningStrategy
    def __init__(
        self, parameters: ParameterValues, *, binning: BinningStrategy | None = None
    ) -> None: ...
    def configuration(self) -> dict: ...
    def _validate_storage_calibration(self) -> None: ...
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs: ArrayLike, **kwargs: Any) -> float: ...

class NeymanChi2InverseGammaGofStatistic(
    AbstractInverseGammaGofStatistic, frequency.AbstractNeymanChi2Statistic
):
    binning: BinningStrategy
    def __init__(
        self, parameters: ParameterValues, *, binning: BinningStrategy | None = None
    ) -> None: ...
    def configuration(self) -> dict: ...
    def _validate_storage_calibration(self) -> None: ...
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs: ArrayLike, **kwargs: Any) -> float: ...

class CressieReadInverseGammaGofStatistic(
    AbstractInverseGammaGofStatistic, frequency.AbstractCressieReadStatistic
):
    binning: BinningStrategy
    def __init__(
        self, parameters: ParameterValues, *, binning: BinningStrategy | None = None
    ) -> None: ...
    def configuration(self) -> dict: ...
    def _validate_storage_calibration(self) -> None: ...
    @classmethod
    def supported_hypotheses(cls) -> tuple[HypothesisSupport, ...]: ...
    def execute_statistic(self, rvs: ArrayLike, **kwargs: Any) -> float: ...

__all__ = [
    "AbstractInverseGammaGofStatistic",
    "AndersonDarlingInverseGammaGofStatistic",
    "Chi2PearsonInverseGammaGofStatistic",
    "CramerVonMisesInverseGammaGofStatistic",
    "CressieReadInverseGammaGofStatistic",
    "FreemanTukeyInverseGammaGofStatistic",
    "GreenwoodInverseGammaGofStatistic",
    "KolmogorovSmirnovDMinusInverseGammaGofStatistic",
    "KolmogorovSmirnovDPlusInverseGammaGofStatistic",
    "KolmogorovSmirnovInverseGammaGofStatistic",
    "KuiperCorrectedInverseGammaGofStatistic",
    "KuiperInverseGammaGofStatistic",
    "KuiperScaledInverseGammaGofStatistic",
    "LiaoShimokawaInverseGammaGofStatistic",
    "ModifiedLogLikelihoodInverseGammaGofStatistic",
    "NeymanChi2InverseGammaGofStatistic",
    "WatsonInverseGammaGofStatistic",
    "ZhangAInverseGammaGofStatistic",
    "ZhangCInverseGammaGofStatistic",
    "ZhangKInverseGammaGofStatistic",
]
