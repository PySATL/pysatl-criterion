from dataclasses import replace
from decimal import Decimal

import pytest
from sqlalchemy import Column, MetaData, Table

from pysatl_criterion.persistence.models.critical_value import CriticalValueModel
from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.persistence.queries.critical_value import (
    CriticalValueFilter,
    CriticalValueLookupQuery,
)
from pysatl_criterion.persistence.queries.limit_distribution import (
    CriterionFilter,
    LimitDistributionFilter,
)
from pysatl_criterion.persistence.sqlalchemy.database import AlchemyDatabase
from pysatl_criterion.persistence.sqlalchemy.models.critical_value import CriticalValueORM
from pysatl_criterion.persistence.sqlalchemy.models.limit_distribution import LimitDistributionORM
from pysatl_criterion.persistence.sqlalchemy.stores.critical_value import (
    AlchemyCriticalValueStorage,
)
from pysatl_criterion.persistence.sqlalchemy.stores.limit_distribution import (
    AlchemyLimitDistributionStorage,
)
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.configuration import configuration_hash


def test_canonical_configuration():
    assert configuration_hash({"b": [2.0], "a": 1}) == configuration_hash({"a": 1.0, "b": [2]})
    assert configuration_hash(None) != configuration_hash({})
    for invalid in (float("nan"), float("inf"), object()):
        with pytest.raises((ValueError, TypeError)):
            configuration_hash({"value": invalid})


def test_configurations_isolate_storage_and_critical_cache():
    db = AlchemyDatabase("sqlite:///:memory:")
    db.init()
    with db.session_factory() as session:
        store = AlchemyLimitDistributionStorage(session)
        cache = AlchemyCriticalValueStorage(session)
        models = [
            LimitDistributionModel("CHI2", {}, 20, 2, [1.0, 2.0], config)
            for config in (None, {"bins": 4}, {"bins": 8})
        ]
        assert len(store.bulk_insert(models)) == 3
        criticals = [
            CriticalValueModel(m.key, 2, Decimal(".05"), AlternativeType.RIGHT, upper_value=i + 2)
            for i, m in enumerate(models)
        ]
        assert len(cache.bulk_insert(criticals)) == 3
        for model, critical in zip(models, criticals, strict=True):
            assert store.get(model.key) == model
            assert (
                cache.get(
                    CriticalValueLookupQuery(model.key, 2, Decimal(".05"), AlternativeType.RIGHT)
                )
                == critical
            )
        for storage, query in ((store, LimitDistributionFilter()), (cache, CriticalValueFilter())):
            seen = []
            after = None
            while batch := storage.get_batch(query, batch_size=1, after=after):
                seen.extend(batch)
                after = batch[-1].key
            assert len(seen) == 3
        assert (
            len(store.get_batch(LimitDistributionFilter(criteria=[CriterionFilter("CHI2")]))) == 3
        )
        assert (
            store.get_batch(
                LimitDistributionFilter(criteria=[CriterionFilter("CHI2", legacy_only=True)])
            )
            == models[:1]
        )
        assert (
            store.get_batch(
                LimitDistributionFilter(
                    criteria=[CriterionFilter("CHI2", criterion_configuration={"bins": 4.0})]
                )
            )
            == models[1:2]
        )
        assert store.insert(replace(models[1], monte_carlo_count=3))
        assert len(cache.get_batch(CriticalValueFilter())) == 2
        cache.delete_for_distribution(criticals[2].distribution_key)
        assert cache.get_batch(CriticalValueFilter()) == criticals[:1]
    db.dispose()


def test_fresh_statistic_cannot_request_legacy_identity():
    from pysatl_criterion.persistence.models.distribution_key import DistributionKey

    class InvalidStatistic:
        def configuration(self):
            return None

    with pytest.raises(TypeError, match="reserved for legacy"):
        DistributionKey.from_statistic(InvalidStatistic(), 20)


def test_schema_requires_configuration_foreign_key():
    db = AlchemyDatabase("sqlite:///:memory:")
    metadata = MetaData()
    for model in (LimitDistributionORM, CriticalValueORM):
        Table(
            model.__tablename__,
            metadata,
            *(
                Column(c.name, c.type, primary_key=c.primary_key, nullable=c.nullable)
                for c in model.__table__.columns
            ),
        )
    metadata.create_all(db.engine)
    try:
        with pytest.raises(ValueError, match="foreign key"):
            db.init()
    finally:
        db.dispose()
