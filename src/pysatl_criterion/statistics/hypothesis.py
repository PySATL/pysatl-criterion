class Hypothesis:
    pass


class GoodnessOfFitHypothesis(Hypothesis):
    def __init__(self, parameters: dict[str, float] | None):
        self.params = parameters

    def parameters(self) -> list[float]:
        if self.params is None:
            return []

        return list(self.params.values())


class IndependenceHypothesis(Hypothesis):
    pass
