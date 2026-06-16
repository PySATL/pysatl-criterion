from abc import ABC, abstractmethod

from pysatl_criterion import DistributionParameterDescriptor, DistributionType
from pysatl_criterion.distribution.validator import (
    NonNegativeNumberValidator,
    PositiveNumberValidator,
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

    @staticmethod
    @abstractmethod
    def type() -> DistributionType:
        """
        Return the distribution type represented by the descriptor.

        :return: distribution enum member.
        """
        pass

    @staticmethod
    @abstractmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return metadata for parameters accepted by the distribution.

        :return: list of distribution parameter descriptors.
        """
        pass


class NormalDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the normal distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the normal distribution type.

        :return: normal distribution enum member.
        """
        return DistributionType.NORMAL

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the normal distribution.

        :return: descriptors for mean and variance.
        """
        return [
            DistributionParameterDescriptor("μ", "mean", "Mean", 0),
            DistributionParameterDescriptor(
                "σ²", "var", "Variance. σ² > 0", 1, PositiveNumberValidator()
            ),
        ]


class ExponentialDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the exponential distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the exponential distribution type.

        :return: exponential distribution enum member.
        """
        return DistributionType.EXPONENTIAL

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the exponential distribution.

        :return: descriptor for the rate parameter.
        """
        return [
            DistributionParameterDescriptor(
                "λ", "lam", "Rate, or inverse scale. λ > 0", 1, PositiveNumberValidator()
            ),
        ]


class WeibullDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the Weibull distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the Weibull distribution type.

        :return: Weibull distribution enum member.
        """
        return DistributionType.WEIBULL

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the Weibull distribution.

        :return: descriptors for scale and shape parameters.
        """
        return [
            DistributionParameterDescriptor("λ", "a", "Scale. λ > 0", 1, PositiveNumberValidator()),
            DistributionParameterDescriptor("k", "k", "Shape. k > 0", 5, PositiveNumberValidator()),
        ]


class UniformDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the continuous uniform distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the uniform distribution type.

        :return: uniform distribution enum member.
        """
        return DistributionType.UNIFORM

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the uniform distribution.

        :return: descriptors for interval start and interval end.
        """
        return [
            DistributionParameterDescriptor("a", "a", "Interval start", 0),
            DistributionParameterDescriptor("b", "b", "Interval end", 1),
        ]


class StudentDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for Student's t-distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the Student's t-distribution type.

        :return: Student's t-distribution enum member.
        """
        return DistributionType.STUDENT

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for Student's t-distribution.

        :return: descriptor for degrees of freedom.
        """
        return [
            DistributionParameterDescriptor(
                "ν", "df", "Degrees of freedom. ν > 0", 2, PositiveNumberValidator()
            ),
        ]


class GammaDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the gamma distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the gamma distribution type.

        :return: gamma distribution enum member.
        """
        return DistributionType.GAMMA

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the gamma distribution.

        :return: descriptors for shape and rate parameters.
        """
        return [
            DistributionParameterDescriptor(
                "α", "alfa", "Shape. α > 0", 1, PositiveNumberValidator()
            ),
            DistributionParameterDescriptor(
                "β", "beta", "Rate. β > 0", 1, PositiveNumberValidator()
            ),
        ]


class BetaDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the beta distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the beta distribution type.

        :return: beta distribution enum member.
        """
        return DistributionType.BETA

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the beta distribution.

        :return: descriptors for the two shape parameters.
        """
        return [
            DistributionParameterDescriptor(
                "α", "a", "First shape parameter. α > 0", 1, PositiveNumberValidator()
            ),
            DistributionParameterDescriptor(
                "β", "b", "Second shape parameter. β > 0", 1, PositiveNumberValidator()
            ),
        ]


class LogNormalDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the log normal distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the log normal distribution type.

        :return: log normal distribution enum member.
        """
        return DistributionType.LOG_NORMAL

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the log normal distribution.

        :return: descriptors for logarithmic mean and variance.
        """
        return [
            DistributionParameterDescriptor("μ", "mean", "Logarithm of mean", 0),
            DistributionParameterDescriptor(
                "σ²", "var", "Logarithm of variance. σ² > 0", 1, PositiveNumberValidator()
            ),
        ]


class CauchyDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the Cauchy distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the Cauchy distribution type.

        :return: Cauchy distribution enum member.
        """
        return DistributionType.CAUCHY

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the Cauchy distribution.

        :return: descriptors for location and scale parameters.
        """
        return [
            DistributionParameterDescriptor("x0", "t", "Location", 0.5),
            DistributionParameterDescriptor(
                "v", "s", "Scale. v > 0", 0.5, PositiveNumberValidator()
            ),
        ]


class Chi2DistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the chi-squared distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the chi-squared distribution type.

        :return: chi-squared distribution enum member.
        """
        return DistributionType.CHI_2

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the chi-squared distribution.

        :return: descriptor for degrees of freedom.
        """
        return [
            DistributionParameterDescriptor(
                "v", "df", "Degrees of freedom. ν > 0", 2, PositiveNumberValidator()
            ),
        ]


class GompertzDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the Gompertz distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the Gompertz distribution type.

        :return: Gompertz distribution enum member.
        """
        return DistributionType.GOMPERTZ

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the Gompertz distribution.

        :return: descriptors for shape and scale parameters.
        """
        return [
            DistributionParameterDescriptor(
                "η", "eta", "Shape. η > 0", 1, PositiveNumberValidator()
            ),
            DistributionParameterDescriptor("b", "b", "Scale. b > 0", 1, PositiveNumberValidator()),
        ]


class GumbelDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the Gumbel distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the Gumbel distribution type.

        :return: Gumbel distribution enum member.
        """
        return DistributionType.GUMBEL

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the Gumbel distribution.

        :return: descriptors for location and scale parameters.
        """
        return [
            DistributionParameterDescriptor("μ", "mu", "Location", 0),
            DistributionParameterDescriptor(
                "β", "beta", "Scale. β > 0", 1, PositiveNumberValidator()
            ),
        ]


class InvGaussDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the inverse Gaussian distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the inverse Gaussian distribution type.

        :return: inverse Gaussian distribution enum member.
        """
        return DistributionType.INV_GAUSS

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the inverse Gaussian distribution.

        :return: descriptors for mean and shape parameters.
        """
        return [
            DistributionParameterDescriptor("μ", "mu", "Mean. μ > 0", 0, PositiveNumberValidator()),
            DistributionParameterDescriptor(
                "λ", "lam", "Shape. λ > 0", 1, PositiveNumberValidator()
            ),
        ]


class LaplaceDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the Laplace distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the Laplace distribution type.

        :return: Laplace distribution enum member.
        """
        return DistributionType.LAPLACE

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the Laplace distribution.

        :return: descriptors for location and scale parameters.
        """
        return [
            DistributionParameterDescriptor("μ", "t", "Location", 0),
            DistributionParameterDescriptor("b", "s", "Scale. b > 0", 1, PositiveNumberValidator()),
        ]


class LoConNormDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the location-contaminated normal distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the location-contaminated normal distribution type.

        :return: location-contaminated normal distribution enum member.
        """
        return DistributionType.LO_CON_NORMAL

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the location-contaminated normal distribution.

        :return: descriptors for contamination probability and location shift.
        """
        return [
            DistributionParameterDescriptor(
                "p",
                "p",
                "Probability of sampling from the shifted normal distribution N(a, 1). 0 <= p <= 1",
                0.5,
                ProbabilityValidator(),
            ),
            DistributionParameterDescriptor(
                "a", "a", "Mean (location shift) of the contaminated component", 0
            ),
        ]


class MixConNormDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the mixed contaminated normal distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the mixed contaminated normal distribution type.

        :return: mixed contaminated normal distribution enum member.
        """
        return DistributionType.MIX_CON_NORMAL

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the mixed contaminated normal distribution.

        :return: descriptors for contamination probability, mean, and scale.
        """
        return [
            DistributionParameterDescriptor(
                "p",
                "p",
                "Probability of sampling from the contaminated normal distribution N(a, b^2)."
                " 0 <= p <= 1",
                0.5,
                ProbabilityValidator(),
            ),
            DistributionParameterDescriptor(
                "a", "a", "Mean of the contaminated normal component", 0
            ),
            DistributionParameterDescriptor(
                "b",
                "b",
                "Standard deviation of the contaminated normal component. b > 0",
                1,
                PositiveNumberValidator(),
            ),
        ]


class ScaleConNormDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the scale-contaminated normal distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the scale-contaminated normal distribution type.

        :return: scale-contaminated normal distribution enum member.
        """
        return DistributionType.SCALE_CON_NORMAL

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the scale-contaminated normal distribution.

        :return: descriptors for contamination probability and scale.
        """
        return [
            DistributionParameterDescriptor(
                "p",
                "p",
                "Probability of sampling from the contaminated normal distribution N(a, b^2)."
                " 0 <= p <= 1",
                0.5,
                ProbabilityValidator(),
            ),
            DistributionParameterDescriptor(
                "b",
                "b",
                "Standard deviation of the contaminated normal component. b > 0",
                1,
                PositiveNumberValidator(),
            ),
        ]


class TruncNormDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the truncated normal distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the truncated normal distribution type.

        :return: truncated normal distribution enum member.
        """
        return DistributionType.TRUNC_NORMAL

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the truncated normal distribution.

        :return: descriptors for mean, variance, and truncation bounds.
        """
        return [
            DistributionParameterDescriptor("μ", "mean", "Mean", 0),
            DistributionParameterDescriptor(
                "σ²", "var", "Variance. σ² > 0", 1, PositiveNumberValidator()
            ),
            DistributionParameterDescriptor("a", "a", "Lower truncation bound", -10),
            DistributionParameterDescriptor("b", "b", "Upper truncation bound", 10),
        ]


class LogisticDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the logistic distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the logistic distribution type.

        :return: logistic distribution enum member.
        """
        return DistributionType.LOGISTIC

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the logistic distribution.

        :return: descriptors for location and scale parameters.
        """
        return [
            DistributionParameterDescriptor("μ", "t", "Location", 0),
            DistributionParameterDescriptor("s", "s", "Scale. s > 0", 1, PositiveNumberValidator()),
        ]


class RiceDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the Rice distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the Rice distribution type.

        :return: Rice distribution enum member.
        """
        return DistributionType.RICE

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the Rice distribution.

        :return: descriptors for distance and scale parameters.
        """
        return [
            DistributionParameterDescriptor(
                "v",
                "nu",
                "Distance between the reference point and the center of the bivariate "
                "distribution. v >= 0,",
                0,
            ),
            DistributionParameterDescriptor(
                "σ", "sigma", "Scale. σ >= 0", 1, NonNegativeNumberValidator()
            ),
        ]


class TukeyDistributionDescriptor(DistributionDescriptor):
    """
    Descriptor for the Tukey lambda distribution.
    """

    @staticmethod
    def type() -> DistributionType:
        """
        Return the Tukey lambda distribution type.

        :return: Tukey lambda distribution enum member.
        """
        return DistributionType.TUKEY

    @staticmethod
    def parameters() -> list[DistributionParameterDescriptor]:
        """
        Return parameters for the Tukey lambda distribution.

        :return: descriptor for the shape parameter.
        """
        return [
            DistributionParameterDescriptor("λ", "lam", "Shape", 2),
        ]
