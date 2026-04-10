import os
from pathlib import Path

from dotenv import load_dotenv


env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

LOCAL_PYSATL_URL = os.getenv("PYSATL_LOCAL_DB_URL", "sqlite:///pysatl.sqlite")

REMOTE_PYSATL_URL = os.getenv(
    "PYSATL_REMOTE_DB_URL", "postgresql://postgres:postgres@localhost:5432/pysatl"
)
