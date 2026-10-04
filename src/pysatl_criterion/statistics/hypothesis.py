class Hypothesis:
    pass


class GoodnessOfFitHypothesis(Hypothesis):
    def __init__(self, parameters: dict[str, float] | None):
        self.params = parameters

    def parameters(self) -> dict[str, float]:
        if self.params is None:
            return {}

        return dict(sorted(self.params.items()))


class IndependenceHypothesis(Hypothesis):
    pass
