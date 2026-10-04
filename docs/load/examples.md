# Distribution loader examples

These examples use the public loader API described in the [loading guide](index.md).
The first example is self-contained and needs no remote server. The later examples assume
that `shared.db` already contains PySATL distributions.

## Run a complete transfer locally

This script creates two temporary SQLite databases, stores one demonstration distribution,
caches a critical value, and copies both records. It then repeats the load to show that existing
equal-version records are skipped. The small synthetic statistics are only demonstration data.

```python
from decimal import Decimal
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from pysatl_criterion.hypothesis_testing.distribution_service import DistributionService
from pysatl_criterion.loader import DistributionLoader
from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.persistence.queries.critical_value import CriticalValueLookupQuery
from pysatl_criterion.persistence.queries.limit_distribution import (
    CriterionFilter,
    LimitDistributionFilter,
)
from pysatl_criterion.persistence.sqlalchemy.database import AlchemyDatabase
from pysatl_criterion.persistence.sqlalchemy.unit_of_work import AlchemyUnitOfWork
from pysatl_criterion.statistics.alternative import AlternativeType


with TemporaryDirectory() as directory:
    remote = AlchemyDatabase(f"sqlite:///{Path(directory) / 'source.db'}")
    local = AlchemyDatabase(f"sqlite:///{Path(directory) / 'destination.db'}")
    try:
        remote.init()
        remote_uow = partial(AlchemyUnitOfWork, remote.session_factory)
        local_uow = partial(AlchemyUnitOfWork, local.session_factory)
        source = DistributionService(remote_uow)

        distribution = LimitDistributionModel(
            criterion_code="demo",
            criterion_parameters={"shape": 2.0},
            sample_size=20,
            monte_carlo_count=4,
            results_statistics=[0.25, 0.5, 0.75, 1.0],
        )
        source.save_distribution(distribution)
        cached = source.get_critical_value(
            distribution.key,
            Decimal("0.05"),
            AlternativeType.RIGHT,
        )

        query = LimitDistributionFilter(
            criteria=[CriterionFilter("demo", {"shape": 2.0})],
            min_sample_size=20,
            max_sample_size=20,
        )
        loader = DistributionLoader(remote=remote, local=local)
        first = loader.load(query, batch_size=1)
        assert first.limit_distributions.saved_count == 1
        assert first.critical_values.saved_count == 1

        # The destination tables are initialized by DistributionLoader.load().
        with local_uow() as uow:
            stored = uow.limit_distributions.get(distribution.key)
            value = uow.critical_values.get(
                CriticalValueLookupQuery(
                    distribution.key,
                    distribution.monte_carlo_count,
                    Decimal("0.05"),
                    AlternativeType.RIGHT,
                )
            )
        assert stored == distribution
        assert value == cached

        repeated = loader.load(query, batch_size=1)
        assert repeated.fetched_count == 2
        assert repeated.saved_count == 0
        assert repeated.skipped_count == 2
    finally:
        remote.dispose()
        local.dispose()
```

Expected output:

```text
Loaded 2 records (limit distributions: 1, critical values: 1; fetched: 2, skipped: 0).
Loaded 0 records (limit distributions: 0, critical values: 0; fetched: 2, skipped: 2).
```

The call to `source.get_critical_value()` prepares the source cache for the demonstration.
`DistributionLoader` itself only transfers that cached record.

## Selecting several parameter sets

Construct one `CriterionFilter` per code and parameter combination. The same code may appear
more than once. This example loads gamma criteria for two different shapes while leaving
sample sizes unrestricted:

```python
from pysatl_criterion.loader import DistributionLoader
from pysatl_criterion.persistence.queries.limit_distribution import (
    CriterionFilter,
    LimitDistributionFilter,
)
from pysatl_criterion.statistics.goodness_of_fit import KolmogorovSmirnovGammaGofStatistic


statistics = [
    KolmogorovSmirnovGammaGofStatistic(alpha=2.0, beta=1.0),
    KolmogorovSmirnovGammaGofStatistic(alpha=3.0, beta=1.0),
]
query = LimitDistributionFilter(
    criteria=[
        CriterionFilter(statistic.code(), statistic.hypothesis().parameters())
        for statistic in statistics
    ],
    min_monte_carlo_count=10_000,
)
loader = DistributionLoader(remote="sqlite:///shared.db", local="sqlite:///local.db")
result = loader.load(query)
print(result.limit_distributions.not_found_codes)
```

To select every parameter variant of one statistic instead, use
`CriterionFilter(statistics[0].code())`. To select exactly one sample size, add equal
`min_sample_size` and `max_sample_size` bounds to the query.

## Reusing database objects

Passing `AlchemyDatabase` objects lets your application control connection options and
reuse the connections after loading:

```python
from pysatl_criterion.loader import DistributionLoader
from pysatl_criterion.persistence.sqlalchemy.database import AlchemyDatabase


remote = AlchemyDatabase("sqlite:///shared.db")
local = AlchemyDatabase("sqlite:///local.db")
try:
    loader = DistributionLoader(remote=remote, local=local)
    result = loader.load()
    # Both objects remain available for subsequent application operations here.
finally:
    remote.dispose()
    local.dispose()
```

The caller owns database objects passed to the loader. When you pass URL strings instead,
each `load()` call creates and disposes its own database objects, including when loading fails.
You can mix a URL for one side with a database object for the other. In all cases, the loader
initializes the local tables and expects the remote tables to exist already.

For loading only distributions, copying only cached values, or reading from an application
API, see [custom loaders and adapters](custom-loaders.md).
