import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from pysatl_criterion.distribution.distribution_kind import DistributionKind
from pysatl_criterion.distribution.distribution_parameter_descriptor import (
    DistributionParameterDescriptor,
)
from pysatl_criterion.distribution.distribution_type import DistributionType
from pysatl_criterion.distribution.parameters import (
    ParameterizationDescriptor,
    ParameterSpec,
    ParameterValues,
)
from pysatl_criterion.distribution.validator import (
    NonNegativeNumberValidator,
    PositiveNumberValidator,
    PositiveRealNumberValidator,
    ProbabilityValidator,
)


"""
Distribution descriptors for supported probability distributions.

Each descriptor maps a :class:`DistributionType` value to metadata about the
parameters required to configure that distribution.
"""


class DistributionDescriptor(ABC):
    """
    Base class for descriptors that expose distribution metadata.

    Concrete descriptors identify a supported distribution and list the parameters
    needed to configure it.
    """

    DEFAULT: ClassVar[ParameterizationDescriptor]

    @classmethod
    def default_parameterization(cls) -> ParameterizationDescriptor:
        """Return the preferred coordinates, without filling parameter values."""
        return cls.DEFAULT

    @classmethod
    def convert_parameters(
        cls, values: ParameterValues, target: ParameterizationDescriptor
    ) -> ParameterValues:
        """Accept identity conversion; other conversions require an implementation."""
        if not isinstance(values, ParameterValues):
            raise TypeError("values must be ParameterValues")
        if (
            values.parameterization not in cls.parameterizations()
            or target not in cls.parameterizations()
        ):
            raise ValueError("Unsupported distribution parameterization")
        if values.parameterization != target:
            raise ValueError("Conversion between these parameterizations is not implemented")
        return values

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        """Return the explicit parameterizations available for the distribution."""
        return ()

    @staticmethod
    @abstractmethod
    def kind() -> DistributionKind:
        """Return the kind of probability distribution represented by the descriptor."""

    @staticmethod
    @abstractmethod
    def type() -> DistributionType:
        """
        Return the distribution type represented by the descriptor.

        :return: distribution enum member.
        """

    @staticmethod
    @abstractmethod
    def parameters() -> Sequence[DistributionParameterDescriptor | ParameterSpec]:
        """
        Return metadata for parameters accepted by the distribution.

        :return: list of distribution parameter descriptors.
        """


class ContinuousDistributionDescriptor(DistributionDescriptor, ABC):
    """Base class for continuous distribution descriptors."""

    @staticmethod
    def kind() -> DistributionKind:
        """Return the continuous distribution kind."""
        return DistributionKind.CONTINUOUS


class DiscreteDistributionDescriptor(DistributionDescriptor, ABC):
    """Base class for discrete distribution descriptors."""

    @staticmethod
    def kind() -> DistributionKind:
        """Return the discrete distribution kind."""
        return DistributionKind.DISCRETE


class NormalDistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the normal distribution.
    """

    MEAN = ParameterSpec("normal.mean", "mean", "μ", "Mean", 0)
    VARIANCE = ParameterSpec(
        "normal.variance", "var", "σ²", "Variance. σ² > 0", 1, PositiveNumberValidator()
    )
    DEFAULT = ParameterizationDescriptor(
        "normal.mean_var", DistributionType.NORMAL, (MEAN, VARIANCE)
    )

    @staticmethod
    def type() -> DistributionType:
        """
        Return the normal distribution type.

        :return: normal distribution enum member.
        """
        return DistributionType.NORMAL

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(NormalDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class ExponentialDistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the exponential distribution.
    """

    RATE = ParameterSpec(
        "exponential.rate",
        "lam",
        "λ",
        "Rate, or inverse scale. λ > 0",
        1,
        PositiveNumberValidator(),
    )
    DEFAULT = ParameterizationDescriptor("exponential.rate", DistributionType.EXPONENTIAL, (RATE,))

    @staticmethod
    def type() -> DistributionType:
        """
        Return the exponential distribution type.

        :return: exponential distribution enum member.
        """
        return DistributionType.EXPONENTIAL

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(ExponentialDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class WeibullDistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the Weibull distribution.
    """

    SCALE = ParameterSpec(
        "weibull.scale", "scale", "λ", "Scale. λ > 0", 1, PositiveRealNumberValidator()
    )
    SHAPE = ParameterSpec(
        "weibull.shape", "shape", "k", "Shape. k > 0", 5, PositiveRealNumberValidator()
    )
    DEFAULT = ParameterizationDescriptor(
        "weibull.scale_shape", DistributionType.WEIBULL, (SCALE, SHAPE)
    )

    @staticmethod
    def type() -> DistributionType:
        """
        Return the Weibull distribution type.

        :return: Weibull distribution enum member.
        """
        return DistributionType.WEIBULL

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(WeibullDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class ExponentiatedWeibullDistributionDescriptor(ContinuousDistributionDescriptor):
    """Exponentiated Weibull with positive exponent, shape and scale; location zero."""

    EXPONENT = ParameterSpec(
        "exponentiated_weibull.exponent",
        "exponent",
        "a",
        "Exponent. a > 0",
        1,
        PositiveRealNumberValidator(),
    )
    SHAPE = ParameterSpec(
        "exponentiated_weibull.shape",
        "shape",
        "k",
        "Shape. k > 0",
        5,
        PositiveRealNumberValidator(),
    )
    SCALE = ParameterSpec(
        "exponentiated_weibull.scale",
        "scale",
        "λ",
        "Scale. λ > 0",
        1,
        PositiveRealNumberValidator(),
    )
    DEFAULT = ParameterizationDescriptor(
        "exponentiated_weibull.exponent_shape_scale",
        DistributionType.EXPONENTIATED_WEIBULL,
        (EXPONENT, SHAPE, SCALE),
    )

    @staticmethod
    def type() -> DistributionType:
        return DistributionType.EXPONENTIATED_WEIBULL

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(ExponentiatedWeibullDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class UniformDistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the continuous uniform distribution.
    """

    LOWER = ParameterSpec("uniform.lower", "a", "a", "Interval start", 0)
    UPPER = ParameterSpec("uniform.upper", "b", "b", "Interval end", 1)
    DEFAULT = ParameterizationDescriptor(
        "uniform.lower_upper", DistributionType.UNIFORM, (LOWER, UPPER)
    )

    @staticmethod
    def type() -> DistributionType:
        """
        Return the uniform distribution type.

        :return: uniform distribution enum member.
        """
        return DistributionType.UNIFORM

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(UniformDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class StudentDistributionDescriptor(ContinuousDistributionDescriptor):
    """Student distribution in degrees-of-freedom/location/scale coordinates."""

    DF = ParameterSpec(
        "student.df",
        "df",
        "ν",
        "Degrees of freedom. ν > 0",
        1,
        PositiveNumberValidator(),
    )
    LOCATION = ParameterSpec("student.location", "loc", "μ", "Location", 0)
    SCALE = ParameterSpec(
        "student.scale",
        "scale",
        "s",
        "Scale. s > 0",
        1,
        PositiveNumberValidator(),
    )
    DEFAULT = ParameterizationDescriptor(
        "student.df_loc_scale", DistributionType.STUDENT, (DF, LOCATION, SCALE)
    )

    @staticmethod
    def type() -> DistributionType:
        return DistributionType.STUDENT

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(StudentDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class GammaDistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the gamma distribution.
    """

    SHAPE = ParameterSpec("gamma.shape", "alfa", "α", "Shape. α > 0", 1, PositiveNumberValidator())
    RATE = ParameterSpec("gamma.rate", "beta", "β", "Rate. β > 0", 1, PositiveNumberValidator())
    DEFAULT = ParameterizationDescriptor("gamma.shape_rate", DistributionType.GAMMA, (SHAPE, RATE))

    @staticmethod
    def type() -> DistributionType:
        """
        Return the gamma distribution type.

        :return: gamma distribution enum member.
        """
        return DistributionType.GAMMA

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(GammaDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class BetaDistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the beta distribution.
    """

    ALPHA = ParameterSpec(
        "beta.alpha", "a", "α", "First shape parameter. α > 0", 1, PositiveNumberValidator()
    )
    BETA = ParameterSpec(
        "beta.beta", "b", "β", "Second shape parameter. β > 0", 1, PositiveNumberValidator()
    )
    DEFAULT = ParameterizationDescriptor("beta.alpha_beta", DistributionType.BETA, (ALPHA, BETA))

    @staticmethod
    def type() -> DistributionType:
        """
        Return the beta distribution type.

        :return: beta distribution enum member.
        """
        return DistributionType.BETA

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(BetaDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class LogNormalDistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the log normal distribution.
    """

    LOG_LOCATION = ParameterSpec("log_normal.log_location", "mu", "μ", "Logarithmic location", 0)
    LOG_SCALE = ParameterSpec(
        "log_normal.log_scale",
        "s",
        "σ",
        "Logarithmic standard deviation. σ > 0",
        1,
        PositiveNumberValidator(),
    )
    LOG_LOCATION_SCALE = ParameterizationDescriptor(
        "log_normal.log_location_log_scale", DistributionType.LOG_NORMAL, (LOG_LOCATION, LOG_SCALE)
    )
    DEFAULT = LOG_LOCATION_SCALE

    MEDIAN = ParameterSpec(
        "log_normal.median", "scale", "m", "Median. m > 0", 1, PositiveNumberValidator()
    )
    SHAPE_SCALE = ParameterizationDescriptor(
        "log_normal.shape_scale", DistributionType.LOG_NORMAL, (LOG_SCALE, MEDIAN)
    )

    @staticmethod
    def type() -> DistributionType:
        """
        Return the log normal distribution type.

        :return: log normal distribution enum member.
        """
        return DistributionType.LOG_NORMAL

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(LogNormalDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT, cls.SHAPE_SCALE)

    @classmethod
    def convert_parameters(
        cls, values: ParameterValues, target: ParameterizationDescriptor
    ) -> ParameterValues:
        """Explicitly convert log location and median, preserving unknown parameters.

        The shared log standard deviation is unchanged. Conversion never fills
        defaults; unrepresentable medians (overflow or underflow to zero) are
        rejected. Other distributions and parameterizations are not accepted.
        """
        if not isinstance(values, ParameterValues):
            raise TypeError("values must be ParameterValues")
        if (
            values.parameterization not in cls.parameterizations()
            or target not in cls.parameterizations()
        ):
            raise ValueError("Unsupported LogNormal parameterization")
        if values.parameterization == target:
            return target.parse(dict(values))
        converted = {}
        if cls.LOG_SCALE in values:
            converted[cls.LOG_SCALE] = values[cls.LOG_SCALE]
        if target == cls.SHAPE_SCALE and cls.LOG_LOCATION in values:
            try:
                median = math.exp(values[cls.LOG_LOCATION])
            except OverflowError as error:
                raise ValueError(
                    "LogNormal median is not representable as a finite positive float"
                ) from error
            if not math.isfinite(median) or median <= 0:
                raise ValueError("LogNormal median is not representable as a finite positive float")
            converted[cls.MEDIAN] = median
        elif target == cls.LOG_LOCATION_SCALE and cls.MEDIAN in values:
            converted[cls.LOG_LOCATION] = math.log(values[cls.MEDIAN])
        return target.parse(converted)


class CauchyDistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the Cauchy distribution.
    """

    LOCATION = ParameterSpec("cauchy.location", "t", "x0", "Location", 0.5)
    SCALE = ParameterSpec("cauchy.scale", "s", "v", "Scale. v > 0", 0.5, PositiveNumberValidator())
    DEFAULT = ParameterizationDescriptor(
        "cauchy.loc_scale", DistributionType.CAUCHY, (LOCATION, SCALE)
    )

    @staticmethod
    def type() -> DistributionType:
        """
        Return the Cauchy distribution type.

        :return: Cauchy distribution enum member.
        """
        return DistributionType.CAUCHY

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(CauchyDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class Chi2DistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the chi-squared distribution.
    """

    DF = ParameterSpec(
        "chi_2.df", "df", "v", "Degrees of freedom. ν > 0", 2, PositiveNumberValidator()
    )
    DEFAULT = ParameterizationDescriptor("chi_2.df", DistributionType.CHI_2, (DF,))

    @staticmethod
    def type() -> DistributionType:
        """
        Return the chi-squared distribution type.

        :return: chi-squared distribution enum member.
        """
        return DistributionType.CHI_2

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(Chi2DistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class GompertzDistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the Gompertz distribution.
    """

    SHAPE = ParameterSpec(
        "gompertz.shape", "eta", "η", "Shape. η > 0", 1, PositiveNumberValidator()
    )
    SCALE = ParameterSpec("gompertz.scale", "b", "b", "Scale. b > 0", 1, PositiveNumberValidator())
    DEFAULT = ParameterizationDescriptor(
        "gompertz.shape_scale", DistributionType.GOMPERTZ, (SHAPE, SCALE)
    )

    @staticmethod
    def type() -> DistributionType:
        """
        Return the Gompertz distribution type.

        :return: Gompertz distribution enum member.
        """
        return DistributionType.GOMPERTZ

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(GompertzDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class GumbelDistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the Gumbel distribution.
    """

    LOCATION = ParameterSpec("gumbel.location", "mu", "μ", "Location", 0)
    SCALE = ParameterSpec("gumbel.scale", "beta", "β", "Scale. β > 0", 1, PositiveNumberValidator())
    DEFAULT = ParameterizationDescriptor(
        "gumbel.loc_scale", DistributionType.GUMBEL, (LOCATION, SCALE)
    )

    @staticmethod
    def type() -> DistributionType:
        """
        Return the Gumbel distribution type.

        :return: Gumbel distribution enum member.
        """
        return DistributionType.GUMBEL

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(GumbelDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class InvGaussDistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the inverse Gaussian distribution.
    """

    MEAN = ParameterSpec("inv_gauss.mean", "mu", "μ", "Mean. μ > 0", 1, PositiveNumberValidator())
    SHAPE = ParameterSpec(
        "inv_gauss.shape", "lam", "λ", "Shape. λ > 0", 1, PositiveNumberValidator()
    )
    DEFAULT = ParameterizationDescriptor(
        "inv_gauss.mean_shape", DistributionType.INV_GAUSS, (MEAN, SHAPE)
    )

    @staticmethod
    def type() -> DistributionType:
        """
        Return the inverse Gaussian distribution type.

        :return: inverse Gaussian distribution enum member.
        """
        return DistributionType.INV_GAUSS

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(InvGaussDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class LaplaceDistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the Laplace distribution.
    """

    LOCATION = ParameterSpec("laplace.location", "t", "μ", "Location", 0)
    SCALE = ParameterSpec("laplace.scale", "s", "b", "Scale. b > 0", 1, PositiveNumberValidator())
    DEFAULT = ParameterizationDescriptor(
        "laplace.loc_scale", DistributionType.LAPLACE, (LOCATION, SCALE)
    )

    @staticmethod
    def type() -> DistributionType:
        """
        Return the Laplace distribution type.

        :return: Laplace distribution enum member.
        """
        return DistributionType.LAPLACE

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(LaplaceDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class HyperbolicDistributionDescriptor(ContinuousDistributionDescriptor):
    """Descriptor for the hyperbolic distribution."""

    SHAPE = ParameterSpec(
        "hyperbolic.shape", "alpha", "α", "Shape. α > 0 and |β| < α", 1, PositiveNumberValidator()
    )
    SKEWNESS = ParameterSpec("hyperbolic.skewness", "beta", "β", "Skewness. |β| < α", 0)
    SCALE = ParameterSpec(
        "hyperbolic.scale", "delta", "δ", "Scale. δ > 0", 1, PositiveNumberValidator()
    )
    LOCATION = ParameterSpec("hyperbolic.location", "mu", "μ", "Location", 0)
    DEFAULT = ParameterizationDescriptor(
        "hyperbolic.shape_skewness_scale_loc",
        DistributionType.HYPERBOLIC,
        (SHAPE, SKEWNESS, SCALE, LOCATION),
    )

    @staticmethod
    def type() -> DistributionType:
        """Return the hyperbolic distribution type.

        :return: hyperbolic distribution enum member.
        """
        return DistributionType.HYPERBOLIC

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(HyperbolicDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class LoConNormDistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the location-contaminated normal distribution.
    """

    PROBABILITY = ParameterSpec(
        "lo_con_normal.probability",
        "p",
        "p",
        "Probability of sampling from the shifted normal distribution N(a, 1). 0 <= p <= 1",
        0.5,
        ProbabilityValidator(),
    )
    LOCATION = ParameterSpec(
        "lo_con_normal.location", "a", "a", "Mean (location shift) of the contaminated component", 0
    )
    DEFAULT = ParameterizationDescriptor(
        "lo_con_normal.probability_location",
        DistributionType.LO_CON_NORMAL,
        (PROBABILITY, LOCATION),
    )

    @staticmethod
    def type() -> DistributionType:
        """
        Return the location-contaminated normal distribution type.

        :return: location-contaminated normal distribution enum member.
        """
        return DistributionType.LO_CON_NORMAL

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(LoConNormDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class MixConNormDistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the mixed contaminated normal distribution.
    """

    PROBABILITY = ParameterSpec(
        "mix_con_normal.probability",
        "p",
        "p",
        "Probability of sampling from the contaminated normal distribution N(a, b^2). 0 <= p <= 1",
        0.5,
        ProbabilityValidator(),
    )
    MEAN = ParameterSpec(
        "mix_con_normal.mean", "a", "a", "Mean of the contaminated normal component", 0
    )
    SCALE = ParameterSpec(
        "mix_con_normal.scale",
        "b",
        "b",
        "Standard deviation of the contaminated normal component. b > 0",
        1,
        PositiveNumberValidator(),
    )
    DEFAULT = ParameterizationDescriptor(
        "mix_con_normal.probability_mean_scale",
        DistributionType.MIX_CON_NORMAL,
        (PROBABILITY, MEAN, SCALE),
    )

    @staticmethod
    def type() -> DistributionType:
        """
        Return the mixed contaminated normal distribution type.

        :return: mixed contaminated normal distribution enum member.
        """
        return DistributionType.MIX_CON_NORMAL

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(MixConNormDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class ScaleConNormDistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the scale-contaminated normal distribution.
    """

    PROBABILITY = ParameterSpec(
        "scale_con_normal.probability",
        "p",
        "p",
        "Probability of sampling from the contaminated normal distribution N(a, b^2). 0 <= p <= 1",
        0.5,
        ProbabilityValidator(),
    )
    SCALE = ParameterSpec(
        "scale_con_normal.scale",
        "b",
        "b",
        "Standard deviation of the contaminated normal component. b > 0",
        1,
        PositiveNumberValidator(),
    )
    DEFAULT = ParameterizationDescriptor(
        "scale_con_normal.probability_scale",
        DistributionType.SCALE_CON_NORMAL,
        (PROBABILITY, SCALE),
    )

    @staticmethod
    def type() -> DistributionType:
        """
        Return the scale-contaminated normal distribution type.

        :return: scale-contaminated normal distribution enum member.
        """
        return DistributionType.SCALE_CON_NORMAL

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(ScaleConNormDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class TruncNormDistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the truncated normal distribution.
    """

    MEAN = ParameterSpec("trunc_normal.mean", "mean", "μ", "Mean", 0)
    VARIANCE = ParameterSpec(
        "trunc_normal.variance", "var", "σ²", "Variance. σ² > 0", 1, PositiveNumberValidator()
    )
    LOWER = ParameterSpec("trunc_normal.lower", "a", "a", "Lower truncation bound", -10)
    UPPER = ParameterSpec("trunc_normal.upper", "b", "b", "Upper truncation bound", 10)
    DEFAULT = ParameterizationDescriptor(
        "trunc_normal.mean_var_lower_upper",
        DistributionType.TRUNC_NORMAL,
        (MEAN, VARIANCE, LOWER, UPPER),
    )

    @staticmethod
    def type() -> DistributionType:
        """
        Return the truncated normal distribution type.

        :return: truncated normal distribution enum member.
        """
        return DistributionType.TRUNC_NORMAL

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(TruncNormDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class LogisticDistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the logistic distribution.
    """

    LOCATION = ParameterSpec("logistic.location", "t", "μ", "Location", 0)
    SCALE = ParameterSpec("logistic.scale", "s", "s", "Scale. s > 0", 1, PositiveNumberValidator())
    DEFAULT = ParameterizationDescriptor(
        "logistic.loc_scale", DistributionType.LOGISTIC, (LOCATION, SCALE)
    )

    @staticmethod
    def type() -> DistributionType:
        """
        Return the logistic distribution type.

        :return: logistic distribution enum member.
        """
        return DistributionType.LOGISTIC

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(LogisticDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class RiceDistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the Rice distribution.
    """

    DISTANCE = ParameterSpec(
        "rice.distance",
        "nu",
        "v",
        "Distance between the reference point and the center of the bivariate "
        "distribution. v >= 0,",
        0,
    )
    SCALE = ParameterSpec(
        "rice.scale", "sigma", "σ", "Scale. σ >= 0", 1, NonNegativeNumberValidator()
    )
    DEFAULT = ParameterizationDescriptor(
        "rice.distance_scale", DistributionType.RICE, (DISTANCE, SCALE)
    )

    @staticmethod
    def type() -> DistributionType:
        """
        Return the Rice distribution type.

        :return: Rice distribution enum member.
        """
        return DistributionType.RICE

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(RiceDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class TukeyDistributionDescriptor(ContinuousDistributionDescriptor):
    """
    Descriptor for the Tukey lambda distribution.
    """

    SHAPE = ParameterSpec("tukey.shape", "lam", "λ", "Shape", 2)
    DEFAULT = ParameterizationDescriptor("tukey.shape", DistributionType.TUKEY, (SHAPE,))

    @staticmethod
    def type() -> DistributionType:
        """
        Return the Tukey lambda distribution type.

        :return: Tukey lambda distribution enum member.
        """
        return DistributionType.TUKEY

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(TukeyDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class ParetoDistributionDescriptor(ContinuousDistributionDescriptor):
    """Descriptor for the pareto distribution with zero location."""

    SHAPE = ParameterSpec(
        "pareto.shape", "shape", "α", "Shape. α > 0", 1, PositiveNumberValidator()
    )

    SCALE = ParameterSpec(
        "pareto.scale",
        "scale",
        "xₘ",
        "Scale (lower endpoint). xₘ > 0",
        1,
        PositiveNumberValidator(),
    )
    DEFAULT = ParameterizationDescriptor(
        "pareto.shape_scale", DistributionType.PARETO, (SHAPE, SCALE)
    )

    @staticmethod
    def type() -> DistributionType:
        return DistributionType.PARETO

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(ParetoDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class InverseGammaDistributionDescriptor(ContinuousDistributionDescriptor):
    """Descriptor for the inverse gamma distribution with zero location."""

    SHAPE = ParameterSpec(
        "inverse_gamma.shape", "alpha", "α", "Shape. α > 0", 1, PositiveNumberValidator()
    )

    SCALE = ParameterSpec(
        "inverse_gamma.scale", "beta", "β", "Scale. β > 0", 1, PositiveNumberValidator()
    )
    DEFAULT = ParameterizationDescriptor(
        "inverse_gamma.shape_scale", DistributionType.INVERSE_GAMMA, (SHAPE, SCALE)
    )

    @staticmethod
    def type() -> DistributionType:
        return DistributionType.INVERSE_GAMMA

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(InverseGammaDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)


class LogLogisticDistributionDescriptor(ContinuousDistributionDescriptor):
    """Descriptor for the log logistic distribution with zero location."""

    SCALE = ParameterSpec(
        "log_logistic.scale", "alpha", "α", "Scale. α > 0", 1, PositiveNumberValidator()
    )

    SHAPE = ParameterSpec(
        "log_logistic.shape", "beta", "β", "Shape. β > 0", 1, PositiveNumberValidator()
    )
    DEFAULT = ParameterizationDescriptor(
        "log_logistic.scale_shape", DistributionType.LOG_LOGISTIC, (SCALE, SHAPE)
    )

    @staticmethod
    def type() -> DistributionType:
        return DistributionType.LOG_LOGISTIC

    @staticmethod
    def parameters() -> list[ParameterSpec]:
        return list(LogLogisticDistributionDescriptor.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)
