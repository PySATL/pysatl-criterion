import json
import logging
from typing import Optional

from sqlalchemy import create_engine, desc, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from pysatl_criterion.persistence.model.limit_distribution.limit_distribution import (
    CriticalValueQuery,
    ILimitDistributionStorage,
    LimitDistributionModel,
    LimitDistributionQuery,
)
from pysatl_criterion.persistence.model.orm.orm import Base, LimitDistributionORM


logger = logging.getLogger(__name__)


class AlchemyLimitDistributionStorage(ILimitDistributionStorage):
    """
    SQLAlchemy-based implementation of ILimitDistributionStorage.
    """

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = create_engine(connection_string, echo=True, future=True)
        self.Session = sessionmaker(bind=self.engine, future=True)

    def init(self) -> None:
        """Create all tables."""
        try:
            with self.engine.connect():
                pass
            Base.metadata.create_all(self.engine)
        except (OperationalError, Exception) as e:
            raise ConnectionError(f"Database unavailable: {self.connection_string}") from e

    @staticmethod
    def create_safe(
        connection_string: str, label: str = "Database"
    ) -> Optional["AlchemyLimitDistributionStorage"]:
        """
        Try to create and initialize storage.
        If database is unavailable, return None instead of raising an exception.
        """
        try:
            storage = AlchemyLimitDistributionStorage(connection_string)
            storage.init()
            logger.info(f"{label} connected successfully.")
            return storage
        except Exception as e:
            logger.warning(f"{label} connection failed: {e}")
            return None

    def insert_data(self, data: LimitDistributionModel) -> None:
        """
        Insert or update limit distribution data in the database.

        If the data already exists, it will be updated; otherwise, a new record will be created.

        :param data: LimitDistributionModel instance containing the data to be inserted or updated.
        """

        orm_obj = LimitDistributionORM.from_model(data)
        with self.Session() as session:
            session.merge(orm_obj)
            session.commit()

    def get_data(self, query: LimitDistributionQuery) -> LimitDistributionModel | None:
        """
        Retrieve limit distribution data based on the provided query parameters.

        :param query: LimitDistributionQuery instance containing the search criteria.

        :return: LimitDistributionModel instance if found, otherwise None.
        """
        with self.Session() as session:
            stmt = select(LimitDistributionORM).where(
                LimitDistributionORM.criterion_code == query.criterion_code,
                LimitDistributionORM.criterion_parameters == json.dumps(query.criterion_parameters),
                LimitDistributionORM.sample_size == query.sample_size,
                LimitDistributionORM.monte_carlo_count == query.monte_carlo_count,
            )
            result = session.execute(stmt).scalar_one_or_none()
            return result.to_model() if result else None

    def delete_data(self, query: LimitDistributionQuery) -> None:
        """
        Delete limit distribution data based on the provided query parameters.
        """
        with self.Session() as session:
            stmt = select(LimitDistributionORM).where(
                LimitDistributionORM.criterion_code == query.criterion_code,
                LimitDistributionORM.criterion_parameters == json.dumps(query.criterion_parameters),
                LimitDistributionORM.sample_size == query.sample_size,
                LimitDistributionORM.monte_carlo_count == query.monte_carlo_count,
            )
            obj = session.execute(stmt).scalar_one_or_none()
            if obj:
                session.delete(obj)
                session.commit()

    def get_data_for_cv(self, query: CriticalValueQuery) -> LimitDistributionModel | None:
        """
        Retrieve limit distribution data for critical value calculation \
        based on the provided query parameters.

        :param query: CriticalValueQuery instance containing the search criteria.

        :return: LimitDistributionModel instance with the highest monte_carlo_count \
        for the given criterion_code and sample_size,
        """
        with self.Session() as session:
            stmt = (
                select(LimitDistributionORM)
                .where(
                    LimitDistributionORM.criterion_code == query.criterion_code,
                    LimitDistributionORM.sample_size == query.sample_size,
                )
                .order_by(desc(LimitDistributionORM.monte_carlo_count))
                .limit(1)
            )
            result = session.execute(stmt).scalar_one_or_none()
            return result.to_model() if result else None
