import importlib


def reload_constants():
    import pysatl_criterion.utils.constants as constants

    return importlib.reload(constants)


def test_database_urls_use_defaults_when_environment_is_missing(monkeypatch):
    monkeypatch.delenv("PYSATL_LOCAL_DB_URL", raising=False)
    monkeypatch.delenv("PYSATL_REMOTE_DB_URL", raising=False)

    constants = reload_constants()

    assert constants.LOCAL_PYSATL_URL == "sqlite:///pysatl.sqlite"
    assert constants.REMOTE_PYSATL_URL == "postgresql://postgres:postgres@db.pysatl.com:5432/pysatl"


def test_database_urls_use_environment_values(monkeypatch):
    monkeypatch.setenv("PYSATL_LOCAL_DB_URL", "sqlite:///local-test.sqlite")
    monkeypatch.setenv(
        "PYSATL_REMOTE_DB_URL",
        "postgresql://test:test@example.com:5432/test",
    )

    constants = reload_constants()

    assert constants.LOCAL_PYSATL_URL == "sqlite:///local-test.sqlite"
    assert constants.REMOTE_PYSATL_URL == "postgresql://test:test@example.com:5432/test"
