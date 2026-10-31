# Custom loaders and storage adapters

Use the built-in [DistributionLoader](index.md) for transfers between supported
databases. To read distributions from a file, an API, or another storage system,
implement a reader and pass it to `LimitDistributionLoader`. To change where
records are saved, implement a writer. The existing loaders already handle
batch iteration and result counts.

See [Usage examples](examples.md) for transfers that use the standard database
adapters.

## Share a storage across pipeline threads

Use `DistributionStorage` and `CriticalValueStorage` from
`pysatl_criterion.persistence.sqlalchemy.storage` when pipeline stages need to
read, calculate, write, and read again. Create one `AlchemyDatabase` per database
configuration in each process, initialize it before starting workers, and pass
the same storage objects between stages. Each method creates its own short-lived
unit of work. Neither storage keeps an active session between calls.

Both storages implement the reader/writer interfaces below and can be passed
directly to `LimitDistributionLoader` or `CriticalValueLoader` as either source
or destination. They also expose `get_batch` for cursor-based reads;
`CriticalValueStorage` exposes `save_if_current` and `delete_for_distribution`.

This example uses illustrative simulated values to demonstrate ownership and
threading; replace the calculation with your calibration algorithm:

```python
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.persistence.sqlalchemy.config import StorageConfig
from pysatl_criterion.persistence.sqlalchemy.database import AlchemyDatabase
from pysatl_criterion.persistence.sqlalchemy.storage import CriticalValueStorage, DistributionStorage


config = StorageConfig(max_workers=30)
database = AlchemyDatabase("sqlite:///calibrations.db", config=config)
try:
    database.init()
    distributions = DistributionStorage(database)
    critical_values = CriticalValueStorage(database)
    initial = [
        LimitDistributionModel(f"example_{i}", {}, 20, 2, [1.0, 2.0])
        for i in range(30)
    ]
    distributions.bulk_insert(initial)

    def process(key):
        source = distributions.get(key)
        if source is None:
            raise LookupError(key)
        # No session or connection is held during this calculation.
        values = source.results_statistics + [3.0, 4.0]
        updated = replace(source, monte_carlo_count=len(values), results_statistics=values)
        distributions.insert(updated)  # Returns after commit and cache invalidation.
        return distributions.get(key)

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        results = list(executor.map(process, [item.key for item in initial]))
finally:
    database.dispose()  # Wait for accepted public operations and stop the SQLite writer.
```

`get` returns a plain model, independent of the closed session. Each
`iter_batches` page releases its transaction and connection **before yielding**;
the complete iteration does not promise one database snapshot. Do not mutate
input models, filter dictionaries, or sequences concurrently with an operation,
and do not advance the same iterator from multiple threads.

`DistributionStorage.insert` and `bulk_insert` save improvements and invalidate
derived critical values in one transaction through `DistributionService`.
`CriticalValueStorage` accepts writes only for the matching local source
version. `save_if_current` returns true for an unchanged but still-current
entry; `insert` reports actual changes. A read/calculate/write pipeline is not
one transaction: the source may change during calculation, so a rejected stale
write must be recomputed from a fresh source when required.

### Configure connection pools and the SQLite writer

`StorageConfig` separates pipeline concurrency from connection limits:

| Setting | Default | Meaning |
| --- | --- | --- |
| `max_workers` | `30` | Suggested executor size; pass it to your executor explicitly. Storage does not start a compute pool. |
| `pool_size` | `None` | Resolves to 5 for file SQLite, or `max_workers` for server databases. |
| `max_overflow` | `0` | Extra connections above `pool_size`; the pool is bounded by default. |
| `pool_timeout` | `30` seconds | Maximum wait to check out a connection; exhaustion raises SQLAlchemy `TimeoutError`. |
| `pool_pre_ping` | `True` | Validate a pooled connection at checkout; does not recover a failed transaction. |
| `pool_recycle` | `-1` | No age-based recycling; set a nonnegative lifetime when required by the server. |
| `sqlite_timeout` | `30` seconds | SQLite driver lock timeout. |
| `write_queue_size` | `30` | Pending SQLite writes, in addition to one running write. |
| `write_queue_timeout` | `30` seconds | Maximum wait for queue capacity; raises built-in `TimeoutError` before submitting the write. |
| `retry_attempts` | `3` | Total attempts for recognized lock, deadlock, and serialization failures. |
| `retry_backoff` | `0.05` seconds | Initial delay between attempts, doubled on each retry. |

For PostgreSQL/MySQL, writes execute in the calling workers using separate
sessions from the shared pool. For example, pass
`StorageConfig(max_workers=30, pool_size=10, max_overflow=0)` to allow 30 compute
workers while limiting that engine to 10 simultaneous database connections.
Database URLs select the driver; install the corresponding driver separately.
Existing `AlchemyDatabase(..., **engine_options)` arguments remain supported and
override configuration-derived engine options. Explicit pool/poolclass options
retain responsibility for their own concurrency behavior.

For file SQLite, all public storage instances sharing one `AlchemyDatabase`
submit writes to its single writer thread. The bounded queue applies
backpressure, and the caller waits for commit or receives the transaction's
exception. Reads continue in their calling threads with independent sessions.
The queue limit is a task count, not a byte limit; choose batch sizes with the
size of compressed arrays in mind. Other database objects or processes have
their own writers and still contend for SQLite's database-wide write lock.
WAL is not enabled automatically.

Ordinary in-memory SQLite retains single-thread behavior. Public storages reject
calls from another thread rather than silently opening a separate empty
database. Use file SQLite for threaded pipelines. A shared `StaticPool` alone
does not make simultaneous use of one SQLite connection safe.

Connection pools open connections on demand. A fresh `Session` does not imply a
new physical connection: completed operations return connections to the pool.
Inspect `database.engine.pool.status()` and measure checkout waits, transaction
durations, throughput, and lock/retry rates on representative workloads before
increasing limits. A larger pool does not provide additional SQLite writers.

### Transactions, retries, and shutdown

Retries reopen the entire operation in a fresh unit of work after rollback.
Validation/integrity errors, queue/pool exhaustion, and invalidated connections
are not retried automatically. A disconnected commit may have an uncertain
outcome and requires application-level handling. Do not add external side
effects to retryable transaction operations.

Keep the existing `AlchemyLimitDistributionStorage(session)` and
`AlchemyCriticalValueStorage(session)` for advanced operations requiring an
explicit shared transaction. The existing Reader/Writer adapters also remain
available. Direct session/UoW access and those adapters bypass the new public
storage dispatcher: callers own their retries, concurrency limits, and shutdown.

Stop pipeline workers before `database.dispose()`. Disposal rejects new public
operations, waits for accepted ones, stops the writer, and releases the engine's
pool; it is terminal for that database object. Exhausted iterators should be
discarded, and subsequent calls must use a new database object. Storages do not
dispose caller-owned databases. Finish direct sessions/UoWs yourself before
disposing the database.

The default example uses one process. For multiprocessing, create the database,
pool, storages, and worker threads **inside each process**; never pass them
through a process pool or reuse them after fork. Public operations detect use
from a different process and raise an error. Budget the sum of
`pool_size + max_overflow` across all engines and processes against the server's
connection limit. The SQLite writer is process-local.

`DistributionLoader(remote=..., local=..., config=config)` uses the public
storages internally. Its configuration applies to databases created from URLs;
passed `AlchemyDatabase` instances retain their own settings and remain
caller-owned. Transfers still finish distributions before critical values and
commit each batch independently.

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
