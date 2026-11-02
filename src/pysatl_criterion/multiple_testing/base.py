"""Generic high-level contract for typed multiple-testing procedures."""

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from .metadata import ProcedureMetadata
from .result import MultipleTestingResult


InputT = TypeVar("InputT")


class AbstractMultipleTesting(ABC, Generic[InputT]):
    metadata: ClassVar[ProcedureMetadata]

    @abstractmethod
    def test(self, data: InputT, alpha: float = 0.05) -> MultipleTestingResult:
        raise NotImplementedError

    @classmethod
    def code(cls) -> str:
        return f"MULTIPLE_TESTING_{cls.short_code().upper()}"

    @classmethod
    def short_code(cls) -> str:
        return cls.metadata.code
