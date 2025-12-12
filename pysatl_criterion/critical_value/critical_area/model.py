from typing import Protocol


class CriticalArea(Protocol):
    def contains(self, value: float) -> bool:
        """
        Check critical area contains value.
        :param value: true, if critical area contains value, false otherwise
        """
        pass
