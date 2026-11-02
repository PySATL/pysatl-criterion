# Saving and retrieving calibrations

Binning affects the null distribution. A calibration for four bins cannot be
reused for eight bins. Storage identity now includes the criterion code,
null-hypothesis parameters, **criterion configuration**, and sample size.
Simulation count still determines which calibration is newer for that identity.
The critical-value cache uses the same configuration and checks the source's
simulation count, significance level and alternative.

## Save and retrieve

```python
from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous.uniform import (
    Chi2PearsonUniformGofStatistic,
)
from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    MonteCarloLimitDistributionResolver,
    StorageLimitDistributionResolver,
)
from pysatl_criterion.persistence.models.distribution_key import DistributionKey
from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.persistence.sqlalchemy.database import AlchemyDatabase
from pysatl_criterion.persistence.sqlalchemy.stores.limit_distribution import (
    AlchemyLimitDistributionStorage,
)

criterion = Chi2PearsonUniformGofStatistic(
    Uniform.DEFAULT.parse({"a": 0, "b": 1}),
    binning=EqualProbabilityBins(8, min_expected=5),
)
values = MonteCarloLimitDistributionResolver(1000).resolve(criterion, 100)
record = LimitDistributionModel.from_statistic(criterion, 100, values)

database = AlchemyDatabase("sqlite:///:memory:")
database.init()
with database.session_factory.begin() as session:
    store = AlchemyLimitDistributionStorage(session)
    store.insert(record)
    saved = store.get(DistributionKey.from_statistic(criterion, 100))
    assert saved is not None
    resolved = StorageLimitDistributionResolver(store).resolve(criterion, 100)
    assert resolved is not None
database.dispose()
```

`StorageLimitDistributionResolver` returns `None` if that configuration is absent.
Place it before `MonteCarloLimitDistributionResolver` in a
`CompositeLimitDistributionResolver` to simulate on a miss. The composite does
not automatically save simulations; use `from_statistic` to persist them.

## Configuration and filters

`criterion.code()` remains unchanged. `criterion.configuration()` includes
`schema_version`, `statistic_version` and `binning`. Built-in policies expose
`to_config()` with a stable type name, version and explicit defaults: count rule,
merging threshold, maximum bins and other policy parameters. An optimized policy
records the selected strategy and sample size; selection scores are excluded.

Both tables store canonical JSON in `criterion_configuration` and SHA-256 in
`criterion_configuration_hash`. The hash participates in primary keys and the
critical-value foreign key. Callers do not calculate hashes themselves.
Dictionary order does not affect identity; equivalent numbers such as `4` and
`4.0` match. Infinite explicit edges use `"-inf"` and `"+inf"`; other non-finite
numbers are rejected.

Custom policies and count rules must implement a stable `to_config()` dictionary
for storage. Monte Carlo does not require this method. Include every option that
affects computation, and increment the policy or statistic version when its
meaning changes.

```python
from pysatl_criterion.persistence.queries.limit_distribution import (
    CriterionFilter,
    LimitDistributionFilter,
)

all_variants = LimitDistributionFilter(
    criteria=[CriterionFilter(criterion.code())],
)
exact_variant = LimitDistributionFilter(
    criteria=[
        CriterionFilter(
            criterion.code(),
            criterion.hypothesis().parameters(),
            criterion.configuration(),
        )
    ],
)
legacy_only = LimitDistributionFilter(
    criteria=[CriterionFilter(criterion.code(), legacy_only=True)],
)
# Within a session:
# rows = store.get_batch(exact_variant, batch_size=100)
# next_rows = store.get_batch(exact_variant, batch_size=100, after=rows[-1].key)
```

Configuration `None` in a **filter** means all variants. Configuration `None` in
a **DistributionKey** means unknown legacy settings. `{}` is an explicit empty
configuration and is distinct from `None`.

## Database schema

`database.init()` rejects incompatible existing tables: `create_all` cannot
change their primary and foreign keys. Use a new database for configuration-aware
calibrations and recalibrate to populate it. No schema migration or transfer of
old records is provided.

Unknown settings are represented by JSON `null` and never match a newly configured
criterion, including defaults. Do not infer missing settings from current defaults.
