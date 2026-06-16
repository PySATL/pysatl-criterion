"""
Validators for distribution parameter values.

The classes in this module provide callable predicates used by
``DistributionParameterDescriptor`` to describe numeric parameter constraints.
Each validator returns ``True`` when a value satisfies the constraint and
``False`` otherwise.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

from typing_extensions import override


class Validator(ABC, Callable[[float], bool]):
    """
    Abstract callable validator for a single numeric distribution parameter.

    Implementations define :meth:`validate`, and instances can be used wherever a
    ``Callable[[float], bool]`` is expected.
    """

    @abstractmethod
    def validate(self, value: float) -> bool:
        """
        Check whether the value satisfies the validator constraint.

        :param value: numeric parameter value to validate.
        :return: ``True`` if the value is valid, otherwise ``False``.
        """
        pass

    def __call__(self, value: float) -> bool:
        """
        Validate the value when the instance is used as a callable.

        :param value: numeric parameter value to validate.
        :return: ``True`` if the value is valid, otherwise ``False``.
        """
        return self.validate(value)


class PositiveNumberValidator(Validator):
    """
    Validate that a numeric parameter is strictly positive.
    """

    @override
    def validate(self, value: float) -> bool:
        """
        Check whether the value is greater than zero.

        :param value: numeric parameter value to validate.
        :return: ``True`` if ``value > 0``, otherwise ``False``.
        """
        return value > 0


class NonNegativeNumberValidator(Validator):
    """
    Validate that a numeric parameter is non-negative.
    """

    @override
    def validate(self, value: float) -> bool:
        """
        Check whether the value is greater than or equal to zero.

        :param value: numeric parameter value to validate.
        :return: ``True`` if ``value >= 0``, otherwise ``False``.
        """
        return value >= 0


class ProbabilityValidator(Validator):
    """
    Validate that a numeric parameter is a probability.
    """

    @override
    def validate(self, value: float) -> bool:
        """
        Check whether the value is in the closed probability interval.

        :param value: numeric parameter value to validate.
        :return: ``True`` if ``0 <= value <= 1``, otherwise ``False``.
        """
        return 0 <= value <= 1
