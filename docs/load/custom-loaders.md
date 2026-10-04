# Custom loaders and storage adapters

Use the built-in [DistributionLoader](index.md) for transfers between supported
databases. To read distributions from a file, an API, or another storage system,
implement a reader and pass it to `LimitDistributionLoader`. To change where
records are saved, implement a writer. The existing loaders already handle
batch iteration and result counts.

See [Usage examples](examples.md) for transfers that use the standard database
adapters.

## Choose an extension point

| Task | Extension point |
| --- | --- |
| Read limit distributions from a custom source | `IBulkStoreReader` with `LimitDistributionLoader` |
| Read cached critical values from a custom source | `IBulkStoreReader` with `CriticalValueLoader` |
| Save either model type to a custom destination | `IBulkStoreWriter` with the corresponding loader |
| Transfer a different model with criterion-based filtering | A subclass of `AbstractLoader` plus suitable adapters |

`DistributionLoader` accepts database URLs or `AlchemyDatabase` objects. It does
not accept custom readers or writers. Use `LimitDistributionLoader` and
`CriticalValueLoader` directly when supplying your own adapters. If you transfer
both kinds of records, finish loading distributions before loading their cached
critical values.

## Reader and writer interfaces

The interfaces are defined in `pysatl_criterion.persistence.stores.base`.

| Interface | Required method | Contract |
| --- | --- | --- |
| `IBulkStoreReader[M, Q, F]` | `get(query: Q) -> M \| None` | Return one model by its identity, or `None` when absent. |
| `IBulkStoreReader[M, Q, F]` | `iter_batches(query: F, *, batch_size: int = 100) -> Iterator[list[M]]` | Apply the filter and yield nonempty batches of at most `batch_size` models. Reject nonpositive batch sizes. |
| `IBulkStoreWriter[M]` | `insert(model: M) -> bool` | Save one model according to the destination policy; return whether it changed stored data. |
| `IBulkStoreWriter[M]` | `bulk_insert(models: Sequence[M]) -> list[M]` | Save a batch and return the accepted model for each identity that changed. |

`get` and `insert` are inherited abstract methods and must be implemented even
though the loaders call only `iter_batches` and `bulk_insert` during a transfer.
Readers need no writing methods, and writers need no reading methods.

The built-in loaders use these types:

| Loader | Model (`M`) | Single-record query (`Q`) | Batch filter (`F`) |
| --- | --- | --- | --- |
| `LimitDistributionLoader` | `LimitDistributionModel` | `DistributionKey` | `LimitDistributionFilter` |
| `CriticalValueLoader` | `CriticalValueModel` | `CriticalValueLookupQuery` | `CriticalValueFilter` |

### Reader responsibilities

Filtering happens in the reader. The loader passes the query through without
filtering the returned models again. Apply every relevant field of the
[filter](index.md), including these distinctions:

- `criteria=None` leaves criterion selection unrestricted; `criteria=[]` matches
  nothing.
- Each `CriterionFilter` binds a code to its own parameters. Match at least one
  complete code/parameter pair, not independent lists of codes and parameters.
- `criterion_parameters=None` matches all parameter variants of that code;
  `{}` matches only the parameterless variant. A supplied dictionary must match
  the entire parameter set.
- Sample-size and Monte Carlo count bounds are inclusive. An omitted bound
  (`None`) imposes no restriction.

For a critical-value reader, also apply the significance-level and alternative
filters and return only values that match the source distribution's current
Monte Carlo count.

Release any source transaction before yielding a batch. A database reader can
use a stable identity cursor to fetch one bounded page at a time; the standard
readers use this approach. The interface does not promise a single snapshot
across batches. Avoid fetching the complete source into memory merely to split
it into batches.

### Writer responsibilities

The writer owns transaction boundaries, conflict handling, and cache consistency.
The batch loader does not open transactions, commit changes, or retry failed
reads and writes. Exceptions propagate, and previously committed batches remain
saved. To retain the standard behavior, commit each batch atomically before
`bulk_insert` returns and roll back that batch on failure.

The returned list controls `LoadResult.saved_count`. Return only changed
identities, not all attempted rows; an unchanged batch returns `[]`.
`fetched_count` counts source rows, while `skipped_count` is the difference
between fetched and saved counts.

For the library's distribution database, prefer the existing
`LimitDistributionWriter(DistributionService(uow_factory))`. It saves new
distributions or improvements with a higher Monte Carlo count and invalidates
their cached critical values in the same transaction.
`CriticalValueWriter(uow_factory)` commits cached values only when their Monte
Carlo count matches the destination distribution. A custom writer should
preserve these rules when maintaining the same data model.

## Example: read distributions from an in-memory collection

This runnable example adapts a collection of already decoded models. It uses
the standard transactional writer with a temporary, in-memory SQLite database.
The sample criterion code and statistics are illustrative.

```python
from collections.abc import Iterator, Sequence
from functools import partial

from pysatl_criterion.hypothesis_testing.distribution_service import DistributionService
from pysatl_criterion.loader.limit_distribution_loader import LimitDistributionLoader
from pysatl_criterion.loader.store_adapters import LimitDistributionWriter
from pysatl_criterion.persistence.models.distribution_key import DistributionKey
from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.persistence.queries.limit_distribution import (
    CriterionFilter,
    LimitDistributionFilter,
)
from pysatl_criterion.persistence.sqlalchemy.database import AlchemyDatabase
from pysatl_criterion.persistence.sqlalchemy.unit_of_work import AlchemyUnitOfWork
from pysatl_criterion.persistence.stores.base import IBulkStoreReader


class SequenceDistributionReader(
    IBulkStoreReader[LimitDistributionModel, DistributionKey, LimitDistributionFilter]
):
    def __init__(self, models: Sequence[LimitDistributionModel]):
        self._models = models

    def get(self, query: DistributionKey) -> LimitDistributionModel | None:
        return next((model for model in self._models if model.key == query), None)

    @staticmethod
    def _matches(model: LimitDistributionModel, query: LimitDistributionFilter) -> bool:
        if query.criteria is not None and not any(
            model.criterion_code == criterion.criterion_code
            and (
                criterion.criterion_parameters is None
                or model.criterion_parameters == criterion.criterion_parameters
            )
            for criterion in query.criteria
        ):
            return False
        if query.min_sample_size is not None and model.sample_size < query.min_sample_size:
            return False
        if query.max_sample_size is not None and model.sample_size > query.max_sample_size:
            return False
        if (
            query.min_monte_carlo_count is not None
            and model.monte_carlo_count < query.min_monte_carlo_count
        ):
            return False
        return True

    def iter_batches(
        self, query: LimitDistributionFilter, *, batch_size: int = 100
    ) -> Iterator[list[LimitDistributionModel]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        batch: list[LimitDistributionModel] = []
        for model in self._models:
            if self._matches(model, query):
                batch.append(model)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch


source = SequenceDistributionReader(
    [
        LimitDistributionModel("example", {"shape": 2.0}, 40, 4, [0.1, 0.2, 0.3, 0.4]),
        LimitDistributionModel("example", {"shape": 3.0}, 40, 4, [0.2, 0.3, 0.4, 0.5]),
        LimitDistributionModel("example", {"shape": 2.0}, 80, 4, [0.3, 0.4, 0.5, 0.6]),
    ]
)
query = LimitDistributionFilter(
    criteria=[CriterionFilter("example", {"shape": 2.0})],
    max_sample_size=40,
    min_monte_carlo_count=4,
)

database = AlchemyDatabase("sqlite:///:memory:")
try:
    database.init()
    uow_factory = partial(AlchemyUnitOfWork, database.session_factory)
    destination = LimitDistributionWriter(DistributionService(uow_factory))
    loader = LimitDistributionLoader(source, destination, print_summary=False)

    result = loader.load(query, batch_size=1)
    assert (result.fetched_count, result.saved_count, result.skipped_count) == (1, 1, 0)

    repeated = loader.load(query, batch_size=1)
    assert (repeated.fetched_count, repeated.saved_count, repeated.skipped_count) == (1, 0, 1)
finally:
    database.dispose()
```

Keep the input collection stable during iteration and supply one current model
per distribution identity. For a file or API reader, replace the collection
scan with streaming reads or pagination while preserving the same filtering
and batch-size contracts.

## Extending `AbstractLoader`

`AbstractLoader[M, Q, F]`, from `pysatl_criterion.loader.abstract_loader`, is the
shared transfer implementation. `F` must be `LimitDistributionFilter` or a
subclass, because the loader reads `query.criteria` to report missing codes.
The only abstract loader method is:

```python
@staticmethod
def _criterion_code(model: M) -> str:
    ...
```

Implement it to return the criterion code associated with your model. The
inherited constructor accepts `source`, `destination`, and the keyword argument
`print_summary=True`; `load(query, *, batch_size=100)` returns `LoadResult`.

For each batch, the base implementation counts fetched rows, records which
requested codes were seen, calls the destination's `bulk_insert`, and counts
the returned models. It writes the current batch before requesting the next
one. `not_found_codes` reports requested codes with no source matches; it does
not report missing individual parameter variants or rejected destination rows.

Use the supplied concrete loaders for `LimitDistributionModel` and
`CriticalValueModel`: their specializations already extract the code from
`model.criterion_code` and `model.distribution_key.criterion_code`, respectively.
