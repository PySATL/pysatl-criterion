# Universal goodness-of-fit statistics

Universal statistics implement an algorithm once and bind it to distributions at
runtime. The first built-in template is `UniversalKolmogorovSmirnov`. Adding a
continuous distribution descriptor with a vectorized CDF makes KS available
through the existing `utils` functions without writing a KS class.

For frequency-based statistics, see [Binning](binning/index.md): the shared
counting pipeline, all available grouping methods, and calibration requirements.

## Using an automatically generated KS

```python
from pysatl_criterion import DistributionType
from pysatl_criterion.distribution import WeibullDistributionDescriptor
from pysatl_criterion.utils.statistic import (
    get_available_criteria,
    get_available_criteria_codes,
    get_criterion_by_code,
)

assert "KS" in get_available_criteria_codes(DistributionType.WEIBULL)
KS = get_criterion_by_code("KS_UNIVERSAL_WEIBULL_GOODNESS_OF_FIT")
assert KS in get_available_criteria(DistributionType.WEIBULL)
parameters = WeibullDistributionDescriptor.DEFAULT.parse({"shape": 2, "scale": 3})
statistic = KS(parameters)
distance = statistic.execute_statistic([0, 0.5, 1, 1, 3, 5])
```

The returned object is a concrete subclass of `AbstractGoodnessOfFitStatistic`
and, when available, the distribution's abstract GoF family class. Repeated
lookups return the same class. Weibull KS accepts **fixed shape and scale**;
existing fitted Weibull criteria continue to accept unknown parameters.

## Adding a distribution

A descriptor remains the source of parameter metadata. It can now also expose
mathematical functions as class or static methods with the signature
`function(x, parameters) -> array`. `x` is a NumPy array, and `parameters` is an
explicit `ParameterValues` object. A function returns one value per input and
must not fill missing parameters implicitly.

Here is a complete extension for the existing `RAYLEIGH` enum member, which has
no built-in descriptor. Importing the module containing this class is sufficient:

```python
import numpy as np
from numpy.typing import NDArray
from scipy import stats

from pysatl_criterion import DistributionType
from pysatl_criterion.distribution import ContinuousDistributionDescriptor
from pysatl_criterion.distribution.parameters import (
    ParameterizationDescriptor,
    ParameterSpec,
    ParameterValues,
)
from pysatl_criterion.distribution.validator import PositiveNumberValidator


class RayleighDistributionDescriptor(ContinuousDistributionDescriptor):
    SCALE = ParameterSpec("rayleigh.scale", "scale", "σ", validator=PositiveNumberValidator())
    DEFAULT = ParameterizationDescriptor("rayleigh.scale", DistributionType.RAYLEIGH, (SCALE,))

    @staticmethod
    def type() -> DistributionType:
        return DistributionType.RAYLEIGH

    @classmethod
    def parameters(cls) -> list[ParameterSpec]:
        return list(cls.DEFAULT.parameters)

    @classmethod
    def parameterizations(cls) -> tuple[ParameterizationDescriptor, ...]:
        return (cls.DEFAULT,)

    @classmethod
    def cdf(cls, x: NDArray[np.float64], parameters: ParameterValues) -> NDArray[np.float64]:
        return stats.rayleigh.cdf(x, scale=parameters[cls.SCALE])
```

```python
from pysatl_criterion.utils.statistic import get_criterion_by_code

KS = get_criterion_by_code("KS_UNIVERSAL_RAYLEIGH_GOODNESS_OF_FIT")
parameters = RayleighDistributionDescriptor.DEFAULT.parse({"scale": 2})
assert KS.distribution() is RayleighDistributionDescriptor
assert 0 <= KS(parameters).execute_statistic([0, 1, 2, 4]) <= 1
```

An entirely new distribution identifier still requires an entry in
`DistributionType`, as before. There must be only one concrete descriptor for an
identifier. A separate family-specific abstract statistic is optional. If an
extension provides one, it can explicitly bind it **before the first lookup**:

```python
from pysatl_criterion.statistics.goodness_of_fit.continuous.registry import criterion_registry

# AbstractRayleighGofStatistic is an optional existing extension class.
criterion_registry.register_distribution(
    RayleighDistributionDescriptor,
    family_base=AbstractRayleighGofStatistic,
)
```

Direct abstract GoF family subclasses from the existing library are discovered
automatically. Parameter validation belongs to the descriptor's parameter schema
and mathematical functions. Generated algorithms initialize the shared
`ParameterValues` contract; they do not run arbitrary family-specific constructors.

## Mathematical capabilities

`DistributionDescriptor.mathematical_functions()` discovers callable `cdf`,
`pdf`, `ppf`, `sf`, `logcdf`, and `logsf` methods. Abstract or missing methods are
not capabilities. Override this mapping to expose additional named functions.
All entries use the same `(x, parameters)` contract. Function names alone do not
supply numerical implementations.

A universal algorithm declares its requirements:

```python
required_capabilities = frozenset({"cdf"})
```

Compatibility also requires a concrete continuous descriptor with a declared
default parameterization. The initial generic hypothesis contract fixes every
parameter in the calculation parameterization. A discrete CDF is deliberately
insufficient for this continuous-KS implementation.

CDF methods were added to Normal, Beta, Uniform, LogNormal, Weibull and
Exponentiated Weibull descriptors, using existing core functions or SciPy.
Other existing KS classes retain family-specific CDF hooks, including stable
transformations and numerical checks. Additional descriptors can opt in by
implementing the same CDF contract; no registry table of distribution names needs
to be edited.

## Registering another universal algorithm

The base `UniversalGoodnessOfFitStatistic` supplies the fixed-parameter
hypothesis contract. For example, the following Cramér–von Mises template reuses
the existing numerical kernel. It is an extension example, not a second
algorithm registered by the library by default.

```python
from abc import ABC

import numpy as np

from pysatl_criterion.statistics.goodness_of_fit.edf import AbstractCrammerVonMisesStatistic
from pysatl_criterion.statistics.goodness_of_fit.continuous.registry import criterion_registry
from pysatl_criterion.statistics.goodness_of_fit.continuous.universal import (
    UniversalGoodnessOfFitStatistic,
)


class UniversalCramerVonMises(
    UniversalGoodnessOfFitStatistic, AbstractCrammerVonMisesStatistic, ABC
):
    required_capabilities = frozenset({"cdf"})

    @staticmethod
    def short_code() -> str:
        return "CVM"

    def execute_statistic(self, rvs, **kwargs) -> float:
        raw = np.asarray(rvs)
        if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iuf":
            raise ValueError("Expected a nonempty one-dimensional real sample")
        if np.ma.isMaskedArray(rvs) and np.any(np.ma.getmaskarray(rvs)):
            raise ValueError("Masked observations are not supported")
        x = np.sort(np.asarray(raw, dtype=float))
        if not np.all(np.isfinite(x)):
            raise ValueError("Observations must be finite")
        cdf = self.distribution().mathematical_functions()["cdf"]
        u = np.asarray(cdf(x, self._parameters), dtype=float)
        if (
            u.shape != x.shape
            or not np.all(np.isfinite(u))
            or np.any((u < 0) | (u > 1))
            or np.any(np.diff(u) < 0)
        ):
            raise FloatingPointError("Invalid CDF values")
        return float(self.do_execute_statistic(x, u))


criterion_registry.register_algorithm(UniversalCramerVonMises)
```

The template must leave `distribution()` abstract. Registering the same template
again is harmless; registering a different template with the same short code
raises an error. Special-purpose algorithms may override `is_compatible()` and
`supported_hypotheses()` to impose narrower requirements.

## Initialization, factory and discovery

All three utility functions use the same registry. At lookup time it:

1. Imports the built-in GoF modules.
2. Recursively discovers concrete distribution descriptors and statistic classes.
3. Finds specialized implementations for each universal algorithm.
4. Generates missing compatible classes using `type()`, with the universal
   template before the optional family base in the MRO.
5. Walks the statistic inheritance tree and returns the selected concrete classes.

The registry owns strong references, serializes initialization with a reentrant
lock, and caches classes by `(distribution_type, algorithm.short_code())`.
Generated class names combine the template name without `Universal`, the
descriptor name without `DistributionDescriptor`, and `GofStatistic`.
Generated codes insert `UNIVERSAL` before the existing family prefix, or before
the uppercase distribution value when there is no family base. For example:
`KS_UNIVERSAL_RAYLEIGH_GOODNESS_OF_FIT`. This prevents reusing retired identifiers
and their calibrations: `KS_WEIBULL_GOODNESS_OF_FIT` remains unavailable, while
the new fixed-parameter Weibull KS is `KS_UNIVERSAL_WEIBULL_GOODNESS_OF_FIT`.
All existing public class names and full codes are retained.

Results are sorted by full code, module and qualified name. Built-in import order
and whether code lookup or family lookup happens first do not change discovery.
External modules must first be imported or registered; the registry does not
search the filesystem for unloaded extensions. Fetch generated classes through
`utils`, rather than assuming a new symbol is exported from a family module.

## Specialized implementations and migration

A specialized class has priority over an automatically generated class with the
same distribution and universal algorithm short code. A specialization imported
after generation takes priority on the next lookup as well. Existing instances
of the generated class remain valid; its presence in `__subclasses__()` cannot be
removed, so the public catalogue filters it out. Two competing specializations
raise a clear error instead of choosing whichever was imported first. Unrelated
legacy algorithms retain their existing discovery behavior.

All 13 existing public KS classes inherit the shared `execute_statistic()`.
Their constructors, hypotheses, full codes and public imports are retained.
Family-specific sample validation, stable CDF evaluation, supported execution
settings and calibration checks remain in place. Beta and Hyperbolic retain
their accelerated numerical kernels. Other classes use `AbstractKSStatistic`'s existing
kernel. The universal execution method checks CDF shape, finite probabilities,
range and monotonicity before evaluating the statistic.

## Calibration limitations

KS generation supplies a statistic only. It does not automatically provide a
sampler, a p-value, stored calibration, or critical values for a new distribution.
Existing resolvers remain responsible for those operations. For example, the
existing Weibull generator can calibrate a generated fixed-parameter Weibull KS,
whereas a new Rayleigh descriptor also needs a compatible sampling/calibration
integration before Monte Carlo hypothesis testing is available.

Unknown parameters are rejected by generated KS rather than fitted implicitly.
A fitted null requires an explicit estimator and suitable calibration. Do not
reuse fixed-null critical values for such a procedure.

`alternative_type` selects D+, D-, or max(D+, D-); every variant rejects for
large distances. Generated KS rejects one-sided storage lookup because the
current storage key does not encode this setting. The legacy `mode` option is
retained but does not calculate a p-value or change the statistic.
