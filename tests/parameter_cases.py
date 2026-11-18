"""Explicit representative hypotheses for parametrized criterion tests."""


def parameters_for(statistic_class, **fixed):
    """Build test values for the first declared hypothesis, with explicit overrides.

    Defaults are test data only. Production constructors never fill parameters.
    Tests of missing parameters construct their ParameterValues directly.
    """
    support = statistic_class.supported_hypotheses()[0]
    values = {
        parameter.name: parameter.default
        for parameter in support.parameterization.parameters
        if parameter in support.fixed_parameters
    }
    if (
        statistic_class.__name__
        in {
            "CadirciRenyiStudentGofStatistic",
            "LequesneRenyiStudentGofStatistic",
            "MeintanisStudentGofStatistic",
        }
        and "df" in values
    ):
        values["df"] = 5  # These moment-based criteria require finite variance.
    values.update(fixed)
    return support.parameterization.parse(values)
