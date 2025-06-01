from typing import Protocol


class IStorage(Protocol):
    """
    Storage interface.
    """

    def init(self) -> None:
        """
        Initialize storage.
        """
        pass
