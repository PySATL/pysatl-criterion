from dataclasses import dataclass, field


@dataclass(frozen=True)
class LoadResult:
    """Transferred row counts; missing codes refer only to the source query."""

    fetched_count: int
    saved_count: int
    not_found_codes: list[str] = field(default_factory=list)

    @property
    def skipped_count(self) -> int:
        """Fetched rows that did not improve the destination."""
        return self.fetched_count - self.saved_count


@dataclass(frozen=True)
class DistributionLoadResult:
    """Counts for both completed transfer stages, with combined row totals."""

    limit_distributions: LoadResult
    critical_values: LoadResult

    @property
    def fetched_count(self) -> int:
        return self.limit_distributions.fetched_count + self.critical_values.fetched_count

    @property
    def saved_count(self) -> int:
        return self.limit_distributions.saved_count + self.critical_values.saved_count

    @property
    def skipped_count(self) -> int:
        return self.limit_distributions.skipped_count + self.critical_values.skipped_count
