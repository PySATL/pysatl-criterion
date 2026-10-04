# Loading distributions

Use `DistributionLoader` to copy stored limit distributions and their cached critical values
into a local database. This lets an application reuse simulation results already available
in another PySATL database. Loading transfers existing records; it does not run simulations
or calculate missing critical values.

Start with the guide below, try the [runnable examples](examples.md), or see
[custom loaders and adapters](custom-loaders.md) to connect your own data source.

## Quick start

With `pysatl-criterion` installed, create a loader and choose a local database:

```python
from pysatl_criterion.loader import DistributionLoader


loader = DistributionLoader(local="sqlite:///local.db")
result = loader.load()
print(result.saved_count)
```

This copies all available distributions and their current cached critical values from the
configured remote database. The loader creates missing tables in `local.db`. The source must
already contain the PySATL tables and the records you want to load.

When `remote` is omitted, the loader uses
`pysatl_criterion.utils.constants.REMOTE_PYSATL_URL`. To configure it, set
`PYSATL_REMOTE_DB_URL` before importing the library, for example before starting Python:

```bash
export PYSATL_REMOTE_DB_URL='sqlite:///shared.db'
```

You can also pass the source explicitly:

```python
loader = DistributionLoader(
    remote="sqlite:///shared.db",
    local="sqlite:///local.db",
)
result = loader.load(batch_size=50)
```

Both arguments accept a SQLAlchemy connection URL or an `AlchemyDatabase` object. `local` is
required; `PYSATL_LOCAL_DB_URL` is not used as its default. An explicit `remote` takes precedence
over the environment configuration. A URL using another database engine requires its Python
driver and a reachable database. See [connection ownership](examples.md#reusing-database-objects)
if your application already manages database objects.

## Selecting distributions

Pass a `LimitDistributionFilter` to `load()`. All its fields are optional:

| Field | Default | Meaning |
| --- | --- | --- |
| `criteria` | `None` | Select any criterion, or supply a sequence of `CriterionFilter` objects. |
| `min_sample_size` | `None` | Inclusive lower sample-size bound; no lower bound when omitted. |
| `max_sample_size` | `None` | Inclusive upper sample-size bound; no upper bound when omitted. |
| `min_monte_carlo_count` | `None` | Inclusive minimum number of simulations; no minimum when omitted. |

`sample_size` is the number of observations in each simulated sample. `monte_carlo_count` is
the number of simulations used to produce a stored distribution. `batch_size` controls how
many database records are transferred in one batch; it is separate from both quantities.

### Bind each code to its parameters

Each `CriterionFilter` contains a required `criterion_code` and optional
`criterion_parameters`. Obtain these from the statistic you use so that they agree with the
stored distribution identity:

```python
from pysatl_criterion.persistence.queries.limit_distribution import (
    CriterionFilter,
    LimitDistributionFilter,
)
from pysatl_criterion.statistics.goodness_of_fit import KolmogorovSmirnovGammaGofStatistic


statistic = KolmogorovSmirnovGammaGofStatistic(alpha=2.0, beta=1.0)
query = LimitDistributionFilter(
    criteria=[
        CriterionFilter(
            criterion_code=statistic.code(),
            criterion_parameters=statistic.hypothesis().parameters(),
        ),
    ],
    min_sample_size=20,
    max_sample_size=100,
    min_monte_carlo_count=10_000,
)
result = loader.load(query, batch_size=50)
```

This example reuses `loader` from the quick start. It selects sample sizes from 20 through 100
for the specified code and complete parameter dictionary, with at least 10,000 simulations.
The result may be empty if the source has no matching records.

A record must match at least one entry in `criteria`, as well as all size and simulation-count
bounds. Each entry's parameters apply only to that entry's code. You can repeat a code to
select several parameter sets, as shown in the [examples](examples.md#selecting-several-parameter-sets).

| Selection | Meaning |
| --- | --- |
| `LimitDistributionFilter()` or `criteria=None` | All criteria and parameter variants. |
| `criteria=[]` | No distributions. |
| `CriterionFilter(code)` or `CriterionFilter(code, None)` | All parameter variants of this code. |
| `CriterionFilter(code, {})` | Only the parameterless variant of this code. |
| `CriterionFilter(code, {"alpha": 2.0, "beta": 1.0})` | Exactly this complete parameter set for this code. |

Parameter matching is not a partial dictionary match: specifying only `alpha` does not match
a stored dictionary containing both `alpha` and `beta`. Dictionary order is ignored and numeric
values such as `2` and `2.0` match the same parameter value.

### Use either size bound independently

```python
# Every size at or above 100; no upper bound.
query = LimitDistributionFilter(min_sample_size=100)

# Every size at or below 100; no lower bound.
query = LimitDistributionFilter(max_sample_size=100)

# Exactly sample size 100.
query = LimitDistributionFilter(min_sample_size=100, max_sample_size=100)
```

Pass the desired query to `loader.load(query)`. Omitted bounds and explicit `None` have the same
meaning. Specified size/count bounds must be positive, and the minimum sample size must not
exceed the maximum. Invalid bounds raise `ValueError` when the filter is constructed.

## Selecting cached critical values

A plain `LimitDistributionFilter` copies all current cached levels and alternatives for the
selected distributions. Use `CriticalValueFilter` to restrict those values:

```python
from decimal import Decimal

from pysatl_criterion.persistence.queries.critical_value import CriticalValueFilter
from pysatl_criterion.statistics.alternative import AlternativeType


query = CriticalValueFilter(
    min_sample_size=20,
    max_sample_size=100,
    significance_levels=[Decimal("0.05"), Decimal("0.01")],
    alternative_types=[AlternativeType.RIGHT],
)
result = loader.load(query)
```

`CriticalValueFilter` inherits `criteria` and all size/count bounds. Those fields select the
distributions in both transfer stages. `significance_levels` and `alternative_types` affect
only the critical-value stage: the distributions are still copied even when no cached values
match. For either extra field, `None` means unrestricted and `[]` means no critical values.

Use `Decimal` levels strictly between zero and one, with at most ten decimal places. Alternatives
are `AlternativeType.LEFT`, `AlternativeType.RIGHT`, and `AlternativeType.TWO_TAILED`.
Only existing current cache entries are transferred. Requesting a level that is absent from
the source does not calculate it.

## Understanding the result

`load()` returns `DistributionLoadResult`. Its `limit_distributions` and `critical_values`
attributes are separate `LoadResult` objects:

```python
print(result.limit_distributions.saved_count)
print(result.critical_values.saved_count)
print(result.fetched_count, result.saved_count, result.skipped_count)
print(result.limit_distributions.not_found_codes)
print(result.critical_values.not_found_codes)
```

| Attribute | Meaning |
| --- | --- |
| `fetched_count` | Number of matching records read from the source. |
| `saved_count` | Number of records that changed the destination. |
| `skipped_count` | `fetched_count - saved_count`. |
| `not_found_codes` | Requested codes with no matching source records in this stage; available on each `LoadResult`. |

The three counts on `DistributionLoadResult` are sums across both stages. One distribution
and its one cached critical-value record therefore count as two records.

Missing codes are tracked after applying the complete filter, not by checking whether a code
exists anywhere in the source. They are deduplicated in request order. If several entries
request one code, any matching record for that code removes it from `not_found_codes`; this
field does not report individual missing parameter sets. With `criteria=None`, there are no
explicitly requested codes, so the lists are empty.

The facade prints one summary after a successful load. To disable summaries, use the component
loaders with `print_summary=False`, as described in [custom loaders](custom-loaders.md).

## Repeated loads and failures

A distribution is identified by its code, complete parameter dictionary, and sample size.
For that identity, the destination keeps the record with the larger `monte_carlo_count`.
Equal or smaller counts are skipped. Updating a distribution also invalidates its local cached
critical values in the same transaction.

The loader finishes all selected distributions before starting the critical-value stage.
Cached values are accepted only when their simulation count matches the destination
distribution; missing distributions, mismatched versions, and already stored equal versions
are skipped. A repeat load can still fill missing cached values even when no distribution changes.

Each destination batch is committed separately. If reading or writing raises an exception,
it propagates to the caller and earlier completed batches remain saved. A failure during
distribution loading prevents the critical-value stage from starting. There is no automatic
retry or saved resume cursor; calling `load()` again rereads the selection and applies the
same rules for keeping existing records.

`batch_size` defaults to 100 and must be positive. It limits records per batch, not the number
of statistics inside each distribution. The built-in readers release each source transaction
before yielding a batch. Reads across multiple batches do not form a single database snapshot.
