# Running tests

Install the project's test dependencies with `poetry sync --only main,test`.

```sh
# Default suite, including SQLite integration tests
poetry run pytest --random-order

# Reproduce a particular test order (pytest prints its seed)
poetry run pytest --random-order --random-order-seed=42

# Exclude loader and persistence tests
poetry run pytest -m 'not integration and not statistical'

# Statement and branch coverage; .coveragerc sets the combined minimum to 80%
poetry run pytest --cov=pysatl_criterion --cov-report=term-missing

# Deterministic references for alternative-specific bin selection
poetry run pytest tests/statistics/test_optimized_binning.py

# Opt-in validation of statistical level and power
PYSATL_STATISTICAL=1 PYSATL_MULTIPLE_TESTING_MONTE_CARLO=1 \
  poetry run pytest -m statistical
```

PostgreSQL and MySQL checks in `tests/persistence/test_distribution_service.py`
also run when `CRITERION_TEST_POSTGRES_URL` and `CRITERION_TEST_MYSQL_URL` are set.
The accounts must be able to create and drop isolated test schemas. MySQL requires
an optional SQLAlchemy driver, such as PyMySQL with its RSA extra; the extended CI
workflow installs it separately. Use dedicated test databases.

The `Statistical and database validation` workflow runs weekly and supports manual
dispatch. Both it and the main CI workflow preserve JUnit reports as artifacts.
