from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from pysatl_criterion.distribution.validator import Validator


@dataclass
class DistributionParameterDescriptor:
    """
    Describe a single configurable parameter for a probability distribution.

    :param display_name: human-readable parameter label used in API output or UI.
    :param name: parameter system label.
    :param description: optional explanation of the parameter's statistical meaning.
    :param default: optional default numeric value for the parameter.
    :param validator: optional validator for the parameter.
    """

    display_name: str
    name: str
    description: str | None = None
    default: float | None = None
    validator: Callable[[float], bool] | Validator | None = None


class DistributionType(Enum):
    """
    Enumerate supported probability distribution identifiers.

    The enum value is the stable string identifier used by public APIs,
    descriptors, and goodness-of-fit statistics.
    """

    NORMAL = "normal"
    EXPONENTIAL = "exponential"
    WEIBULL = "weibull"
    UNIFORM = "uniform"
    STUDENT = "student"
    GAMMA = "gamma"
    BETA = "beta"
    LOG_NORMAL = "log_normal"
    CAUCHY = "cauchy"
    CHI_2 = "chi_2"
    GOMPERTZ = "gompertz"
    GUMBEL = "gumbel"
    INV_GAUSS = "inv_gauss"
    LAPLACE = "laplace"
    LO_CON_NORMAL = "lo_con_normal"
    LOGISTIC = "logistic"
    MIX_CON_NORMAL = "mix_con_normal"
    RICE = "rice"
    SCALE_CON_NORMAL = "scale_con_normal"
    TRUNC_NORMAL = "trunc_normal"
    TUKEY = "tukey"

    def __new__(cls, value: str):
        """
        Create an enum member whose canonical value is the distribution identifier.

        :param value: stable string identifier for the distribution.
        :return: initialized enum member.
        """
        obj = object.__new__(cls)
        # The first item in the tuple becomes the canonical .value
        obj._value_ = value
        return obj

    @classmethod
    def list(cls):
        """
        Return all supported distribution identifiers.

        :return: list of string values for all members in the enum.
        """
        return [member.value for member in cls]
