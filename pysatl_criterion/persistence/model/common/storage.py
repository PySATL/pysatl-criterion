from typing import Protocol

class IStorage(Protocol):
    """
    Storage interface.
    """
    def migrate(self):
        """
        Migrate storage.
        """
        pass

    def init(self):
        """
        Initialize storage.
        """
        pass