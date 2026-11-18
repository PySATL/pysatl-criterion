from sqlalchemy.orm import Session


class AlchemyStorage:
    """Use the caller's session without opening or committing an independent transaction."""

    def __init__(self, session: Session):
        self._session = session
