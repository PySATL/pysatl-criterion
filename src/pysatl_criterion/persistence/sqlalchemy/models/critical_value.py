import json
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    Enum,
    ForeignKeyConstraint,
    Numeric,
    PrimaryKeyConstraint,
    String,
    UnicodeText,
)
from sqlalchemy.orm import Mapped, mapped_column

from pysatl_criterion.persistence.models.critical_value import CriticalValueModel
from pysatl_criterion.persistence.models.distribution_key import DistributionKey
from pysatl_criterion.persistence.sqlalchemy.base import Base
from pysatl_criterion.persistence.sqlalchemy.models.limit_distribution import (
    criterion_code_hash,
    criterion_parameters_hash,
    normalize_criterion_parameters,
    serialize_criterion_parameters,
)
from pysatl_criterion.statistics.alternative import AlternativeType


class CriticalValueORM(Base):
    __tablename__ = "critical_values"

    criterion_code_hash: Mapped[str] = mapped_column(String(64))
    criterion_parameters_hash: Mapped[str] = mapped_column(String(64))
    criterion_code: Mapped[str] = mapped_column(UnicodeText)
    criterion_parameters: Mapped[str] = mapped_column(UnicodeText)
    sample_size: Mapped[int]
    monte_carlo_count: Mapped[int]
    significance_level: Mapped[Decimal] = mapped_column(Numeric(12, 10))
    alternative_type: Mapped[AlternativeType] = mapped_column(
        Enum(
            AlternativeType,
            values_callable=lambda enum: [v.value for v in enum],
            native_enum=False,
            create_constraint=True,
            name="critical_value_alternative",
        )
    )
    lower_value: Mapped[float | None]
    upper_value: Mapped[float | None]

    __table_args__ = (
        PrimaryKeyConstraint(
            "criterion_code_hash",
            "criterion_parameters_hash",
            "sample_size",
            "significance_level",
            "alternative_type",
            name="pk_critical_values",
        ),
        ForeignKeyConstraint(
            ["criterion_code_hash", "criterion_parameters_hash", "sample_size"],
            [
                "limit_distributions.criterion_code_hash",
                "limit_distributions.criterion_parameters_hash",
                "limit_distributions.sample_size",
            ],
            ondelete="CASCADE",
            name="fk_cv_distribution",
        ),
        CheckConstraint(
            "significance_level > 0 AND significance_level < 1",
            name="ck_cv_significance",
        ),
        CheckConstraint("monte_carlo_count > 0", name="ck_critical_values_count"),
        CheckConstraint(
            "(alternative_type = 'left' AND lower_value IS NOT NULL AND upper_value IS NULL)"
            " OR (alternative_type = 'right' AND lower_value IS NULL AND upper_value IS NOT NULL)"
            " OR (alternative_type = 'two_tailed' AND lower_value IS NOT NULL"
            " AND upper_value IS NOT NULL AND lower_value <= upper_value)",
            name="ck_critical_values_bounds",
        ),
    )

    def to_model(self) -> CriticalValueModel:
        return CriticalValueModel(
            distribution_key=DistributionKey(
                self.criterion_code,
                normalize_criterion_parameters(json.loads(self.criterion_parameters)),
                self.sample_size,
            ),
            monte_carlo_count=self.monte_carlo_count,
            significance_level=self.significance_level,
            alternative_type=self.alternative_type,
            lower_value=self.lower_value,
            upper_value=self.upper_value,
        )

    @staticmethod
    def from_model(model: CriticalValueModel) -> "CriticalValueORM":
        key = model.distribution_key
        return CriticalValueORM(
            criterion_code_hash=criterion_code_hash(key.criterion_code),
            criterion_parameters_hash=criterion_parameters_hash(key.criterion_parameters),
            criterion_code=key.criterion_code,
            criterion_parameters=serialize_criterion_parameters(key.criterion_parameters),
            sample_size=key.sample_size,
            monte_carlo_count=model.monte_carlo_count,
            significance_level=model.significance_level,
            alternative_type=model.alternative_type,
            lower_value=model.lower_value,
            upper_value=model.upper_value,
        )
