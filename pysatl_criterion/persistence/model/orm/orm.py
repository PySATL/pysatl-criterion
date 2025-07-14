import json

from pysatl_criterion.persistence.model.limit_distribution.limit_distribution import LimitDistributionModel

from sqlalchemy.orm import declarative_base, Mapped, mapped_column

from sqlalchemy import (
    Text,
    UniqueConstraint,
)

Base = declarative_base()


class LimitDistributionORM(Base):
    __tablename__ = 'limit_distributions'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_id: Mapped[int]
    criterion_code: Mapped[str] = mapped_column(Text)
    criterion_parameters: Mapped[str] = mapped_column(Text)
    sample_size: Mapped[int]
    monte_carlo_count: Mapped[int]
    results_statistics: Mapped[str] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint(
            "experiment_id",
            "criterion_code",
            "criterion_parameters",
            "sample_size",
            "monte_carlo_count",
            name="uix_limit_distribution",
        ),
    )

    def to_model(self) -> LimitDistributionModel:
        """
        Convert ORM object to LimitDistributionModel.

        :return: LimitDistributionModel instance.
        """
        return LimitDistributionModel(
            experiment_id=self.experiment_id,
            criterion_code=self.criterion_code,
            criterion_parameters=json.loads(self.criterion_parameters),
            sample_size=self.sample_size,
            monte_carlo_count=self.monte_carlo_count,
            results_statistics=json.loads(self.results_statistics),
        )

    @staticmethod
    def from_model(model: LimitDistributionModel) -> "LimitDistributionORM":
        """
        Convert LimitDistributionModel to ORM object.

        :param model: LimitDistributionModel instance to convert.

        :return: LimitDistributionORM instance.
        """
        return LimitDistributionORM(
            experiment_id=model.experiment_id,
            criterion_code=model.criterion_code,
            criterion_parameters=json.dumps(model.criterion_parameters),
            sample_size=model.sample_size,
            monte_carlo_count=model.monte_carlo_count,
            results_statistics=json.dumps(model.results_statistics),
        )
