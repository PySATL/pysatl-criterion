from types import SimpleNamespace

import pytest

from pysatl_criterion.hypothesis_testing.distribution_service import DistributionService
from pysatl_criterion.loader.limit_distribution_loader import LimitDistributionLoader
from pysatl_criterion.loader.store_adapters import (
    LimitDistributionReader,
    LimitDistributionWriter,
)
from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.persistence.sqlalchemy.database import AlchemyDatabase
from pysatl_criterion.persistence.sqlalchemy.unit_of_work import AlchemyUnitOfWork


@pytest.fixture
def setup():
    local = AlchemyDatabase("sqlite:///:memory:")
    remote = AlchemyDatabase("sqlite:///:memory:")
    local.init()
    remote.init()
    local_uow = lambda: AlchemyUnitOfWork(local.session_factory)  # noqa: E731
    remote_uow = lambda: AlchemyUnitOfWork(remote.session_factory)  # noqa: E731
    local_service = DistributionService(local_uow)
    remote_service = DistributionService(remote_uow)
    local_reader = LimitDistributionReader(local_uow)
    remote_reader = LimitDistributionReader(remote_uow)
    local_writer = LimitDistributionWriter(local_service)
    yield SimpleNamespace(
        local_reader=local_reader,
        remote_reader=remote_reader,
        local_service=local_service,
        remote_service=remote_service,
        local_uow=local_uow,
        remote_uow=remote_uow,
        local_database=local,
        remote_database=remote,
        local_writer=local_writer,
        loader=LimitDistributionLoader(remote_reader, local_writer),
    )
    local.dispose()
    remote.dispose()


@pytest.fixture
def model():
    return LimitDistributionModel("ks", {}, 100, 1000, [1.0, 2.0])
