from pysatl_criterion.distribution.parameters import ParameterValues


class Hypothesis:
    pass


class GoodnessOfFitHypothesis(Hypothesis):
    """Describe the fixed distribution parameters of a goodness-of-fit null.

    Parameters
    ----------
    parameters : dict of str to float, ParameterValues, or None
        Parameter names and values fixed by the null hypothesis. An empty
        dictionary or ``None`` leaves every parameter free within the family
        returned by the statistic's ``distribution()`` method. Omitted keys
        represent free parameters, not generator defaults.

    Notes
    -----
    Algorithm settings, such as histogram bin counts, do not belong here.
    This object describes the hypothesis; the statistic and its calibration
    procedure determine how unknown distribution parameters are handled.
    """

    def __init__(self, parameters: dict[str, float] | ParameterValues | None):
        self.parameter_values = parameters if isinstance(parameters, ParameterValues) else None
        self.params = (
            parameters.as_dict() if isinstance(parameters, ParameterValues) else parameters
        )

    def parameters(self) -> dict[str, float]:
        """Return a copy of the fixed parameters, ordered by name.

        Returns
        -------
        dict of str to float
            Fixed parameter values, or an empty dictionary for the full family.
        """
        if self.parameter_values is not None:
            return dict(sorted(self.parameter_values.as_dict().items()))
        if self.params is None:
            return {}

        return dict(sorted(self.params.items()))


class IndependenceHypothesis(Hypothesis):
    pass
