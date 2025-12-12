from abc import ABC, abstractmethod


class CriticalArea(ABC):
    @abstractmethod
    def contains(self, value: float) -> bool:
        """
        Check critical area contains value.
        :param value: true, if critical area contains value, false otherwise
        """
        pass
