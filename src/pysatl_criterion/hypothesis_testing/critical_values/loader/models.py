from dataclasses import dataclass, field


@dataclass(frozen=True)
class BulkLoadResult:
    """Distribution row counts; missing codes refer only to the source query."""

    fetched_count: int
    saved_count: int
    not_found_codes: list[str] = field(default_factory=list)

    @property
    def skipped_count(self) -> int:
        """Fetched rows that did not improve the destination."""
        return self.fetched_count - self.saved_count
