"""Utilities for resolving random value sample generators."""

import importlib
import inspect
from typing import TYPE_CHECKING, Any

from pysatl_criterion import DistributionType
from pysatl_criterion.generator.model import AbstractRVSGenerator


if TYPE_CHECKING:
    from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic


def get_hypothesis_generator(
    statistic: "AbstractGoodnessOfFitStatistic",
) -> AbstractRVSGenerator:
    """Build a sampler using the statistic's hypothesis parameterization.

    Hypothesis metadata retains its public names and units. Sampling uses
    Beta's a/b, Gamma's alfa/rate, and log-normal's logarithmic location mu.
    Direct calls to get_available_generator continue to use generator parameters.
    """
    hypothesis = statistic.hypothesis()
    if hypothesis.parameter_values is None or not statistic.supports_hypothesis(hypothesis):
        raise ValueError("Sampling requires a supported ParameterValues hypothesis")
    if statistic.distribution().type() in (
        DistributionType.WEIBULL,
        DistributionType.EXPONENTIATED_WEIBULL,
    ):
        return _weibull_hypothesis_generator(statistic, hypothesis.parameter_values)
    return _schema_hypothesis_generator(statistic, hypothesis)


def _schema_hypothesis_generator(statistic, hypothesis):
    """Choose representatives only for explicitly invariant composite hypotheses."""
    from pysatl_criterion.distribution.distributions import LogNormalDistributionDescriptor
    from pysatl_criterion.generator.generators import TRVSGenerator

    descriptor = statistic.distribution()
    distribution = descriptor.type()
    values = hypothesis.parameter_values
    if distribution == DistributionType.LOG_NORMAL:
        values = descriptor.convert_parameters(
            values, LogNormalDistributionDescriptor.LOG_LOCATION_SCALE
        )
        return get_available_generator(
            distribution, {"mu": values[descriptor.LOG_LOCATION], "s": values[descriptor.LOG_SCALE]}
        )
    if values.parameterization != descriptor.default_parameterization():
        raise ValueError("Unsupported sampling parameterization")
    external_calibration = {
        DistributionType.PARETO: (
            "Pareto requires external calibration; no built-in generator exists. "
            "For fitted KL specify the simulation shape and window m; "
            "refit unknown parameters with execute_statistic on every replicate"
        ),
        DistributionType.INVERSE_GAMMA: (
            "Inverse Gamma requires external calibration; no built-in generator exists. "
            "For fitted KS specify a shape greater than two and refit every replicate"
        ),
        DistributionType.LOG_LOGISTIC: (
            "Log-logistic requires external calibration; no built-in generator exists. "
            "Preserve interval settings, refit unknown parameters, and for Nikulin "
            "reproduce the censoring plan"
        ),
        DistributionType.HYPERBOLIC: (
            "Hyperbolic requires external calibration; no built-in generator exists"
        ),
    }
    if distribution in external_calibration:
        raise ValueError(external_calibration[distribution])
    complete = frozenset(values) == frozenset(descriptor.default_parameterization().parameters)
    if distribution == DistributionType.BETA and not complete:
        raise ValueError(
            "A composite Beta null has no parameter-free sampler; "
            "calibration must specify simulation shapes externally and call "
            "execute_statistic on each simulated sample"
        )
    if distribution == DistributionType.GAMMA and not complete:
        raise ValueError(
            "A composite Gamma null requires external shape-specific calibration; "
            "refit with execute_statistic on every simulated sample"
        )
    if distribution == DistributionType.STUDENT:
        # The fitted Student procedure removes location and scale, retaining df.
        if not complete:
            values = descriptor.DEFAULT.parse(
                {
                    descriptor.DF: values[descriptor.DF],
                    descriptor.LOCATION: 0,
                    descriptor.SCALE: 1,
                }
            )
        return TRVSGenerator.from_parameters(values)
    if distribution == DistributionType.NORMAL:
        parameters = {
            "mean": values.get(descriptor.MEAN, 0),
            "var": values.get(descriptor.VARIANCE, 1),
        }
    elif distribution == DistributionType.EXPONENTIAL:
        parameters = {"lam": values.get(descriptor.RATE, 1)}
    elif distribution == DistributionType.UNIFORM and not complete:
        parameters = {"a": 0, "b": 1}
    else:
        fixed_coordinates = {
            DistributionType.BETA: (("a", "ALPHA"), ("b", "BETA")),
            DistributionType.GAMMA: (("alfa", "SHAPE"), ("beta", "RATE")),
            DistributionType.LAPLACE: (("t", "LOCATION"), ("s", "SCALE")),
            DistributionType.UNIFORM: (("a", "LOWER"), ("b", "UPPER")),
        }
        if distribution not in fixed_coordinates:
            raise ValueError("No schema-aware sampler for this distribution")
        parameters = {
            name: values[getattr(descriptor, attribute)]
            for name, attribute in fixed_coordinates[distribution]
        }
    return get_available_generator(distribution, parameters)


def _weibull_hypothesis_generator(statistic, values):
    """Resolve distinct Weibull families from explicit schemas and calibration contracts."""
    descriptor = statistic.distribution()
    if values is None or values.parameterization != descriptor.default_parameterization():
        raise ValueError("Unsupported Weibull hypothesis: external calibration is required")
    if not statistic.supports_hypothesis(statistic.hypothesis()):
        raise ValueError("Unsupported Weibull hypothesis: external calibration is required")
    validate = getattr(statistic, "_validate_monte_carlo_calibration", None)
    if validate is not None:
        validate()
    if not values:
        calibration = getattr(statistic, "_standard_calibration_parameters", None)
        if calibration is None:
            raise ValueError("This composite hypothesis requires external calibration")
        values = calibration()
    if values.parameterization != descriptor.default_parameterization() or frozenset(
        values
    ) != frozenset(descriptor.default_parameterization().parameters):
        raise ValueError("Incomplete Weibull calibration parameters")
    parameters = {"shape": values[descriptor.SHAPE], "scale": values[descriptor.SCALE]}
    if descriptor.type() == DistributionType.EXPONENTIATED_WEIBULL:
        parameters["exponent"] = values[descriptor.EXPONENT]
    return get_available_generator(descriptor.type(), parameters)


def get_available_generator(
    distribution: DistributionType, params: dict[str, float] | None
) -> AbstractRVSGenerator:
    """
    Return a concrete random value generator for a distribution.

    The function walks all ``AbstractRVSGenerator`` subclasses, finds the
    non-abstract generator whose ``distribution_type`` matches the requested
    distribution, and initializes it with the provided parameters.

    :param distribution: distribution type to generate random values from.
    :param params: keyword parameters passed to the matching generator constructor.
    :return: initialized random value generator for the requested distribution.
    """
    if distribution == DistributionType.STUDENT:
        from pysatl_criterion.distribution.distributions import StudentDistributionDescriptor
        from pysatl_criterion.generator.generators import TRVSGenerator

        values = StudentDistributionDescriptor.DEFAULT.parse(params or {}, fill_defaults=True)
        return TRVSGenerator.from_parameters(values)
    _load_generators()
    return next(
        cls(**(params or {}))
        for cls in __get_all_subclasses(AbstractRVSGenerator)
        if not inspect.isabstract(cls) and cls.distribution_type() == distribution
    )


def _load_generators() -> None:
    """Import concrete generator classes so subclass discovery can find them."""
    importlib.import_module("pysatl_criterion.generator.generators")


def __get_all_subclasses(cls: type[Any]) -> set[type[AbstractRVSGenerator]]:
    """
    Return all direct and indirect random value generator subclasses.

    The search walks the full subclass tree recursively and returns each
    discovered ``AbstractRVSGenerator`` subclass once.

    :param cls: root class whose subclass hierarchy should be inspected.
    :return: set containing all generator subclasses below the root class.
    """
    subclasses: set[type[AbstractRVSGenerator]] = set()

    for subclass in cls.__subclasses__():
        subclasses.update(__get_all_subclasses(subclass))

        if issubclass(subclass, AbstractRVSGenerator):
            subclasses.add(subclass)

    return subclasses
