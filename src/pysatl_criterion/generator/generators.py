"""Random value generators for statistical experiment pipelines.

This module contains implementations of random value sample generators
used in goodness-of-fit experiments. Each generator wraps a distribution
sampling function from ``pysatl_criterion`` and provides:

- unique experiment code generation,
- unified sample generation API,
- parameterized distribution configuration.

All generators inherit from ``AbstractRVSGenerator``.

Classes
-------
BetaRVSGenerator
    Beta distribution generator.

CauchyRVSGenerator
    Cauchy distribution generator.

LaplaceRVSGenerator
    Laplace distribution generator.

LogisticRVSGenerator
    Logistic distribution generator.

TRVSGenerator
    Student's t-distribution generator.

TukeyRVSGenerator
    Tukey distribution generator.

LognormGenerator
    Log-normal distribution generator.

GammaGenerator
    Gamma distribution generator.

TruncnormGenerator
    Truncated normal distribution generator.

Chi2Generator
    Chi-squared distribution generator.

GumbelGenerator
    Gumbel distribution generator.

WeibullGenerator
    Weibull distribution generator.

LoConNormGenerator
    Location-contaminated normal distribution generator.

ScConNormGenerator
    Scale-contaminated normal distribution generator.

MixConNormGenerator
    Mixed contaminated normal distribution generator.

ExponentialGenerator
    Exponential distribution generator.

InvGaussGenerator
    Inverse Gaussian distribution generator.

RiceGenerator
    Rice distribution generator.

GompertzGenerator
    Gompertz distribution generator.

NormalGenerator
    Normal distribution generator.
"""

from typing_extensions import override

from pysatl_criterion import DistributionType
from pysatl_criterion.core.distributions.beta import generate_beta
from pysatl_criterion.core.distributions.cauchy import generate_cauchy
from pysatl_criterion.core.distributions.chi2 import generate_chi2
from pysatl_criterion.core.distributions.expon import generate_expon
from pysatl_criterion.core.distributions.gamma import generate_gamma
from pysatl_criterion.core.distributions.gompertz import generate_gompertz
from pysatl_criterion.core.distributions.gumbel import generate_gumbel
from pysatl_criterion.core.distributions.invgauss import generate_invgauss
from pysatl_criterion.core.distributions.laplace import generate_laplace
from pysatl_criterion.core.distributions.lo_con_norm import generate_lo_con_norm
from pysatl_criterion.core.distributions.logistic import generate_logistic
from pysatl_criterion.core.distributions.lognormal import generate_lognorm
from pysatl_criterion.core.distributions.mix_con_norm import generate_mix_con_norm
from pysatl_criterion.core.distributions.norm import generate_norm
from pysatl_criterion.core.distributions.rice import generate_rice
from pysatl_criterion.core.distributions.scale_con_norm import generate_scale_con_norm
from pysatl_criterion.core.distributions.student import generate_t
from pysatl_criterion.core.distributions.truncnormal import generate_truncnorm
from pysatl_criterion.core.distributions.tukey import generate_tukey
from pysatl_criterion.core.distributions.uniform import generate_uniform
from pysatl_criterion.core.distributions.weibull import generate_weibull
from pysatl_criterion.generator.model import AbstractRVSGenerator


class BetaRVSGenerator(AbstractRVSGenerator):
    """Beta distribution random value generator.

    Parameters
    ----------
    a : float
        First shape parameter.
    b : float
        Second shape parameter.
    """

    def __init__(self, a=1, b=1, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    @override
    def parameters(self) -> dict[str, float]:
        """Return beta distribution parameters."""
        return {"a": self.a, "b": self.b}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return beta distribution type."""
        return DistributionType.BETA

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.a, self.b])

    @override
    def generate(self, size):
        """Generate beta-distributed random sample."""
        return generate_beta(size=size, a=self.a, b=self.b)


class CauchyRVSGenerator(AbstractRVSGenerator):
    """Cauchy distribution random value generator.

    Parameters
    ----------
    t : float
        Location parameter.
    s : float
        Scale parameter.
    """

    def __init__(self, t, s, **kwargs):
        super().__init__(**kwargs)
        self.t = t
        self.s = s

    @override
    def parameters(self) -> dict[str, float]:
        """Return Cauchy distribution parameters."""
        return {"t": self.t, "s": self.s}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return Cauchy distribution type."""
        return DistributionType.CAUCHY

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.t, self.s])

    @override
    def generate(self, size):
        """Generate Cauchy-distributed random sample."""
        return generate_cauchy(size=size, t=self.t, s=self.s)


class LaplaceRVSGenerator(AbstractRVSGenerator):
    """Laplace distribution random value generator.

    Parameters
    ----------
    t : float
        Location parameter.
    s : float
        Scale parameter.
    """

    def __init__(self, t, s, **kwargs):
        super().__init__(**kwargs)
        self.t = t
        self.s = s

    @override
    def parameters(self) -> dict[str, float]:
        """Return Laplace distribution parameters."""
        return {"t": self.t, "s": self.s}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return Laplace distribution type."""
        return DistributionType.LAPLACE

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.t, self.s])

    @override
    def generate(self, size):
        """Generate Laplace-distributed random sample."""
        return generate_laplace(size=size, t=self.t, s=self.s)


class LogisticRVSGenerator(AbstractRVSGenerator):
    """Logistic distribution random value generator.

    Parameters
    ----------
    t : float
        Location parameter.
    s : float
        Scale parameter.
    """

    def __init__(self, t, s, **kwargs):
        super().__init__(**kwargs)
        self.t = t
        self.s = s

    @override
    def parameters(self) -> dict[str, float]:
        """Return logistic distribution parameters."""
        return {"t": self.t, "s": self.s}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return logistic distribution type."""
        return DistributionType.LOGISTIC

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.t, self.s])

    @override
    def generate(self, size):
        """Generate logistic-distributed random sample."""
        return generate_logistic(size=size, t=self.t, s=self.s)


class TRVSGenerator(AbstractRVSGenerator):
    """Student's t-distribution random value generator.

    Parameters
    ----------
    df : float
        Degrees of freedom.
    """

    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.df = df

    @override
    def parameters(self) -> dict[str, float]:
        """Return Student's t-distribution parameters."""
        return {"df": self.df}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return Student's t-distribution type."""
        return DistributionType.STUDENT

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.df])

    @override
    def generate(self, size):
        """Generate Student's t-distributed random sample."""
        return generate_t(size=size, df=self.df)


class TukeyRVSGenerator(AbstractRVSGenerator):
    """Tukey distribution random value generator.

    Parameters
    ----------
    lam : float
        Shape parameter.
    """

    def __init__(self, lam, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam

    @override
    def parameters(self) -> dict[str, float]:
        """Return Tukey distribution parameters."""
        return {"lam": self.lam}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return Tukey distribution type."""
        return DistributionType.TUKEY

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.lam])

    @override
    def generate(self, size):
        """Generate Tukey-distributed random sample."""
        return generate_tukey(size=size, lam=self.lam)


class LognormGenerator(AbstractRVSGenerator):
    """Log-normal distribution random value generator.

    Parameters
    ----------
    s : float, default=1
        Shape parameter.
    mu : float, default=0
        Mean parameter.
    """

    def __init__(self, s=1, mu=0, **kwargs):
        super().__init__(**kwargs)
        self.s = s
        self.mu = mu

    @override
    def parameters(self) -> dict[str, float]:
        """Return log-normal distribution parameters."""
        return {"s": self.s, "mu": self.mu}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return log-normal distribution type."""
        return DistributionType.LOG_NORMAL

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.s, self.mu])

    @override
    def generate(self, size):
        """Generate log-normal distributed random sample."""
        return generate_lognorm(size=size, s=self.s, mu=self.mu)


class GammaGenerator(AbstractRVSGenerator):
    """Gamma distribution random value generator.

    Parameters
    ----------
    alfa : float, default=1
        Shape parameter.
    beta : float, default=0
        Scale parameter.
    """

    def __init__(self, alfa=1, beta=1, **kwargs):
        super().__init__(**kwargs)
        self.alfa = alfa
        self.beta = beta

    @override
    def parameters(self) -> dict[str, float]:
        """Return gamma distribution parameters."""
        return {"alfa": self.alfa, "beta": self.beta}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return gamma distribution type."""
        return DistributionType.GAMMA

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.alfa, self.beta])

    @override
    def generate(self, size):
        """Generate gamma-distributed random sample."""
        return generate_gamma(size=size, alfa=self.alfa, beta=self.beta)


class TruncnormGenerator(AbstractRVSGenerator):
    """Truncated normal distribution random value generator.

    Parameters
    ----------
    mean : float, default=0
        Mean value.
    var : float, default=1
        Variance value.
    a : float, default=-10
        Lower truncation bound.
    b : float, default=10
        Upper truncation bound.
    """

    def __init__(self, mean=0, var=1, a=-10, b=10, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.var = var
        self.a = a
        self.b = b

    @override
    def parameters(self) -> dict[str, float]:
        """Return truncated normal distribution parameters."""
        return {"mean": self.mean, "var": self.var, "a": self.a, "b": self.b}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return truncated normal distribution type."""
        return DistributionType.TRUNC_NORMAL

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code(
            [self.distribution_type(), self.mean, self.var, self.a, self.b]
        )

    @override
    def generate(self, size):
        """Generate truncated normal random sample."""
        return generate_truncnorm(size=size, mean=self.mean, var=self.var, a=self.a, b=self.b)


class Chi2Generator(AbstractRVSGenerator):
    """Chi-squared distribution random value generator.

    Parameters
    ----------
    df : float, default=2
        Degrees of freedom.
    """

    def __init__(self, df=2, **kwargs):
        super().__init__(**kwargs)
        self.df = df

    @override
    def parameters(self) -> dict[str, float]:
        """Return chi-squared distribution parameters."""
        return {"df": self.df}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return chi-squared distribution type."""
        return DistributionType.CHI_2

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.df])

    @override
    def generate(self, size):
        """Generate chi-squared random sample."""
        return generate_chi2(size=size, df=self.df)


class GumbelGenerator(AbstractRVSGenerator):
    """Gumbel distribution random value generator.

    Parameters
    ----------
    mu : float, default=0
        Location parameter.
    beta : float, default=1
        Scale parameter.
    """

    def __init__(self, mu=0, beta=1, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.beta = beta

    @override
    def parameters(self) -> dict[str, float]:
        """Return Gumbel distribution parameters."""
        return {"mu": self.mu, "beta": self.beta}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return Gumbel distribution type."""
        return DistributionType.GUMBEL

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.mu, self.beta])

    @override
    def generate(self, size):
        """Generate Gumbel-distributed random sample."""
        return generate_gumbel(size=size, mu=self.mu, beta=self.beta)


class WeibullGenerator(AbstractRVSGenerator):
    """Weibull distribution random value generator.

    Parameters
    ----------
    a : float, default=1
        Scale parameter.
    k : float, default=5
        Shape parameter.
    """

    def __init__(self, a=1, k=5, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.k = k

    @override
    def parameters(self) -> dict[str, float]:
        """Return Weibull distribution parameters."""
        return {"a": self.a, "k": self.k}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return Weibull distribution type."""
        return DistributionType.WEIBULL

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.a, self.k])

    @override
    def generate(self, size):
        """Generate Weibull-distributed random sample."""
        return generate_weibull(size=size, a=self.a, k=self.k)


class LoConNormGenerator(AbstractRVSGenerator):
    """Location-contaminated normal distribution generator.

    Parameters
    ----------
    p : float, default=0.5
        Contamination probability.
    a : float, default=0
        Location shift parameter.
    """

    def __init__(self, p=0.5, a=0, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.a = a

    @override
    def parameters(self) -> dict[str, float]:
        """Return location-contaminated normal distribution parameters."""
        return {"p": self.p, "a": self.a}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return location-contaminated normal distribution type."""
        return DistributionType.LO_CON_NORMAL

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.p, self.a])

    @override
    def generate(self, size):
        """Generate contaminated normal random sample."""
        return generate_lo_con_norm(size=size, p=self.p, a=self.a)


class ScConNormGenerator(AbstractRVSGenerator):
    """Scale-contaminated normal distribution generator.

    Parameters
    ----------
    p : float, default=0.5
        Contamination probability.
    b : float, default=1
        Scale parameter.
    """

    def __init__(self, p=0.5, b=1, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.b = b

    @override
    def parameters(self) -> dict[str, float]:
        """Return scale-contaminated normal distribution parameters."""
        return {"p": self.p, "b": self.b}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return scale-contaminated normal distribution type."""
        return DistributionType.SCALE_CON_NORMAL

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.p, self.b])

    @override
    def generate(self, size):
        """Generate scale-contaminated normal random sample."""
        return generate_scale_con_norm(size=size, p=self.p, b=self.b)


class MixConNormGenerator(AbstractRVSGenerator):
    """Mixed contaminated normal distribution generator.

    Parameters
    ----------
    p : float, default=0.5
        Contamination probability.
    a : float, default=0
        Mean shift parameter.
    b : float, default=1
        Scale parameter.
    """

    def __init__(self, p=0.5, a=0, b=1, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.a = a
        self.b = b

    @override
    def parameters(self) -> dict[str, float]:
        """Return mixed contaminated normal distribution parameters."""
        return {"p": self.p, "a": self.a, "b": self.b}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return mixed contaminated normal distribution type."""
        return DistributionType.MIX_CON_NORMAL

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.p, self.a, self.b])

    @override
    def generate(self, size):
        """Generate mixed contaminated normal random sample."""
        return generate_mix_con_norm(size=size, p=self.p, a=self.a, b=self.b)


class ExponentialGenerator(AbstractRVSGenerator):
    """Exponential distribution random value generator.

    Parameters
    ----------
    lam : float, default=0.5
        Rate parameter.
    """

    def __init__(self, lam=0.5, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam

    @override
    def parameters(self) -> dict[str, float]:
        """Return exponential distribution parameters."""
        return {"lam": self.lam}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return exponential distribution type."""
        return DistributionType.EXPONENTIAL

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.lam])

    @override
    def generate(self, size):
        """Generate exponentially distributed random sample."""
        return generate_expon(size=size, lam=self.lam)


class InvGaussGenerator(AbstractRVSGenerator):
    """Inverse Gaussian distribution random value generator.

    Parameters
    ----------
    mu : float, default=0
        Mean parameter.
    lam : float, default=1
        Shape parameter.
    """

    def __init__(self, mu=0, lam=1, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.lam = lam

    @override
    def parameters(self) -> dict[str, float]:
        """Return inverse Gaussian distribution parameters."""
        return {"mu": self.mu, "lam": self.lam}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return inverse Gaussian distribution type."""
        return DistributionType.INV_GAUSS

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.mu, self.lam])

    @override
    def generate(self, size):
        """Generate inverse Gaussian random sample."""
        return generate_invgauss(size=size, mu=self.mu, lam=self.lam)


class RiceGenerator(AbstractRVSGenerator):
    """Rice distribution random value generator.

    Parameters
    ----------
    nu : float, default=0
        Noncentrality parameter.
    sigma : float, default=1
        Scale parameter.
    """

    def __init__(self, nu=0, sigma=1, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu
        self.sigma = sigma

    @override
    def parameters(self) -> dict[str, float]:
        """Return Rice distribution parameters."""
        return {"nu": self.nu, "sigma": self.sigma}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return Rice distribution type."""
        return DistributionType.RICE

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.nu, self.sigma])

    @override
    def generate(self, size):
        """Generate Rice-distributed random sample."""
        return generate_rice(size=size, nu=self.nu, sigma=self.sigma)


class GompertzGenerator(AbstractRVSGenerator):
    """Gompertz distribution random value generator.

    Parameters
    ----------
    eta : float, default=0
        Shape parameter.
    b : float, default=1
        Scale parameter.
    """

    def __init__(self, eta=0, b=1, **kwargs):
        super().__init__(**kwargs)
        self.eta = eta
        self.b = b

    @override
    def parameters(self) -> dict[str, float]:
        """Return Gompertz distribution parameters."""
        return {"eta": self.eta, "b": self.b}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return Gompertz distribution type."""
        return DistributionType.GOMPERTZ

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.eta, self.b])

    @override
    def generate(self, size):
        """Generate Gompertz-distributed random sample."""
        return generate_gompertz(size=size, eta=self.eta, b=self.b)


class NormalGenerator(AbstractRVSGenerator):
    """Normal distribution random value generator.

    Parameters
    ----------
    mean : float, default=0
        Mean value.
    var : float, default=1
        Variance value.
    """

    def __init__(self, mean=0, var=1, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.var = var

    @override
    def parameters(self) -> dict[str, float]:
        """Return normal distribution parameters."""
        return {"mean": self.mean, "var": self.var}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return normal distribution type."""
        return DistributionType.NORMAL

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.mean, self.var])

    @override
    def generate(self, size):
        """Generate normally distributed random sample."""
        return generate_norm(size=size, mean=self.mean, var=self.var)


class UniformGenerator(AbstractRVSGenerator):
    """Uniform distribution random value generator.

    Parameters
    ----------
    a : float, default=0
        Left value.
    b : float, default=1
        Right value.
    """

    def __init__(self, a=0, b=1, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    @override
    def parameters(self) -> dict[str, float]:
        """Return uniform distribution parameters."""
        return {"a": self.a, "b": self.b}

    @staticmethod
    @override
    def distribution_type() -> DistributionType:
        """Return uniform distribution type."""
        return DistributionType.UNIFORM

    @override
    def code(self):
        """Return unique generator code."""
        return super()._convert_to_code([self.distribution_type(), self.a, self.b])

    @override
    def generate(self, size):
        """Generate uniform distributed random sample."""
        return generate_uniform(size=size, a=self.a, b=self.b)
