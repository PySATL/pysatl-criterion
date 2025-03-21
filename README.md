# PySATL Criterion

[![Pysatl_criterion CI](https://github.com/PySATL/pysatl-criterion/workflows/PySATL%20CI/badge.svg)](https://github.com/PySATL/pysatl-criterion/actions)
[![Coverage Status](https://coveralls.io/repos/github/PySATL/pysatl-criterion/badge.svg?branch=main)](https://coveralls.io/github/PySATL/pysatl-criterion?branch=main)
[![Documentation](https://readthedocs.org/projects/pysatl-criterion/badge)](https://pysatl-criterion.readthedocs.io)

This repository contains a collection of Python scripts and modules for performing various statistical tests. These tests are designed to help you analyze data, make inferences, and draw conclusions based on statistical methods. The repository is organized to be easy to use, modular, and extensible.

## Installation
To use this repository, you need to have Python 3.9 or later installed on your system. You can install the required dependencies by following these steps:

Clone the repository:

```bash
git clone https://github.com/PySATL/pysatl-criterion.git
```

Install dependencies:

```bash
poetry install
```
You're all set! You can now import and use the statistical tests in your Python scripts.

## PySATL Criterion module usage example:

```python
# import needed criterion from pysatl_criterion
from pysatl_criterion.normal import KolmogorovSmirnovNormalityGofStatistic


# make a criterion object
criterion = KolmogorovSmirnovNormalityGofStatistic(mean=0, var=1)

# initialize test data
x = [0.1, 0.7, 0.5, 0.3]

# then run algorithm
statistic = criterion.execute_statistic(x)

# print the results
print(f"Statistic result: {statistic}")
# output:
# Statistic result: 0.539827837277029
```

## Documentation
We invite you to read the bot documentation to ensure you understand how the PySATL Criterion lib is working.

Please find the complete documentation on the [PySATL Criterion website](https://pysatl-criterion.readthedocs.io/en/latest/).

## Support
### [Bugs / Issues](https://github.com/PySATL/pysatl-criterion/issues?q=is%3Aissue)

If you discover a bug in the PySATL criterion lib, please
[search the issue tracker](https://github.com/PySATL/pysatl-criterion/issues?q=is%3Aissue)
first. If it hasn't been reported, please
[create a new issue](https://github.com/PySATL/pysatl-criterion/issues/new/choose) and
ensure you follow the template guide so that the team can assist you as
quickly as possible.

For every [issue](https://github.com/PySATL/pysatl-criterion/issues/new/choose) created, kindly follow up and mark satisfaction or reminder to close issue when equilibrium ground is reached.

--Maintain github's [community policy](https://docs.github.com/en/site-policy/github-terms/github-community-code-of-conduct)--

### [Feature Requests](https://github.com/PySATL/pysatl-criterion/labels/enhancement)

Have you a great idea to improve the bot you want to share? Please,
first search if this feature was not [already discussed](https://github.com/PySATL/pysatl-criterion/labels/enhancement).
If it hasn't been requested, please
[create a new request](https://github.com/PySATL/pysatl-criterion/issues/new/choose)
and ensure you follow the template guide so that it does not get lost
in the bug reports.

### [Pull Requests](https://github.com/PySATL/pysatl-criterion/pulls)

Feel like the PySATL criterion lib is missing a feature? We welcome your pull requests!

Please read the
[Contributing document](https://github.com/PySATL/pysatl-criterion/blob/develop/CONTRIBUTING.md)
to understand the requirements before sending your pull-requests.

Coding is not a necessity to contribute - maybe start with improving the documentation?
Issues labeled [good first issue](https://github.com/PySATL/pysatl-criterion/labels/good%20first%20issue) can be good first contributions, and will help get you familiar with the codebase.

**Note** before starting any major new feature work, *please open an issue describing what you are planning to do*. This will ensure that interested parties can give valuable feedback on the feature, and let others know that you are working on it.

**Important:** Always create your PR against the `develop` branch, not `stable`.


## Software requirements
- [Python >= 3.9](http://docs.python-guide.org/en/latest/starting/installation/)
- [poetry](https://python-poetry.org/docs/)
- [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- [virtualenv](https://virtualenv.pypa.io/en/stable/installation.html) (Recommended)

## License

This project is licensed under the terms of the **MIT** license. See the [LICENSE](LICENSE) for more information.
