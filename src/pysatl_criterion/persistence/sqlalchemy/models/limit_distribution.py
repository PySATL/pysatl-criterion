import json
from hashlib import sha256
from typing import Any

from sqlalchemy import Index, PrimaryKeyConstraint, String, UnicodeText
from sqlalchemy.orm import Mapped, mapped_column

from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.persistence.sqlalchemy.base import Base
from pysatl_criterion.persistence.sqlalchemy.types import CompressedFloatArray


def normalize_criterion_parameters(parameters: Any) -> dict[str, float]:
    """Sort parameter names and normalize numeric values to floats."""
    if parameters is None:
        return {}

    if isinstance(parameters, list) and not parameters:
        return {}

    if not isinstance(parameters, dict):
        raise TypeError("criterion_parameters must be a dict")

    return {key: float(value) for key, value in sorted(parameters.items())}


def serialize_criterion_parameters(parameters: dict[str, float]) -> str:
    return json.dumps(
        normalize_criterion_parameters(parameters),
        separators=(",", ":"),
        sort_keys=True,
    )


def criterion_code_hash(code: str) -> str:
    return sha256(code.encode("utf-8")).hexdigest()


def criterion_parameters_hash(parameters: dict[str, float]) -> str:
    return sha256(serialize_criterion_parameters(parameters).encode("utf-8")).hexdigest()


class LimitDistributionORM(Base):
    """
    ORM model for limit distribution storage.
    """

    __tablename__ = "limit_distributions"

    criterion_code_hash: Mapped[str] = mapped_column(String(64))
    criterion_parameters_hash: Mapped[str] = mapped_column(String(64))
    criterion_code: Mapped[str] = mapped_column(UnicodeText)
    criterion_parameters: Mapped[str] = mapped_column(UnicodeText)
    sample_size: Mapped[int]
    monte_carlo_count: Mapped[int]
    results_statistics: Mapped[list[float]] = mapped_column(CompressedFloatArray(use_float32=True))

    __table_args__ = (
        PrimaryKeyConstraint(
            "criterion_code_hash",
            "criterion_parameters_hash",
            "sample_size",
            name="uix_limit_distribution",
        ),
        Index("ix_ld_monte_carlo_count", "monte_carlo_count"),
    )

    def to_model(self) -> LimitDistributionModel:
        """
        Convert ORM object to LimitDistributionModel.

        :return: LimitDistributionModel instance.
        """
        return LimitDistributionModel(
            criterion_code=self.criterion_code,
            criterion_parameters=normalize_criterion_parameters(
                json.loads(self.criterion_parameters)
            ),
            sample_size=self.sample_size,
            monte_carlo_count=self.monte_carlo_count,
            results_statistics=self.results_statistics,
        )

    @staticmethod
    def from_model(model: LimitDistributionModel) -> "LimitDistributionORM":
        """
        Convert LimitDistributionModel to ORM object.

        :param model: LimitDistributionModel instance to convert.

        :return: LimitDistributionORM instance.
        """
        return LimitDistributionORM(
            criterion_code_hash=criterion_code_hash(model.criterion_code),
            criterion_parameters_hash=criterion_parameters_hash(model.criterion_parameters),
            criterion_code=model.criterion_code,
            criterion_parameters=serialize_criterion_parameters(model.criterion_parameters),
            sample_size=model.sample_size,
            monte_carlo_count=model.monte_carlo_count,
            results_statistics=model.results_statistics,
        )
