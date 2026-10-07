from collections.abc import Callable
from dataclasses import dataclass

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
