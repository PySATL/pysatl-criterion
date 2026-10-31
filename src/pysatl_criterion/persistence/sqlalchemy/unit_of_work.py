from types import TracebackType

from sqlalchemy.orm import Session, sessionmaker
from typing_extensions import Self

from pysatl_criterion.persistence.sqlalchemy.stores.critical_value import (
    AlchemyCriticalValueStorage,
)
from pysatl_criterion.persistence.sqlalchemy.stores.limit_distribution import (
    AlchemyLimitDistributionStorage,
)
from pysatl_criterion.persistence.unit_of_work import IUnitOfWork


class AlchemyUnitOfWork(IUnitOfWork):
    def __init__(self, session_factory: sessionmaker[Session]):
        self._session_factory = session_factory
        self._session: Session | None = None

    def __enter__(self) -> Self:
        if self._session is not None:
            raise RuntimeError("Unit of work is already active")
        # Session.close() normally resets a reusable session. A repository retained
        # outside this UoW must never reopen a transaction on that old session.
        self._session = self._session_factory(close_resets_only=False)
        self.limit_distributions = AlchemyLimitDistributionStorage(self._session)
        self.critical_values = AlchemyCriticalValueStorage(self._session)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._session is not None:
            try:
                self._session.rollback()
            finally:
                self._session.close()
                self._session = None

    def commit(self) -> None:
        if self._session is None:
            raise RuntimeError("Unit of work is not active")
        self._session.commit()

    def rollback(self) -> None:
        if self._session is None:
            raise RuntimeError("Unit of work is not active")
        self._session.rollback()
