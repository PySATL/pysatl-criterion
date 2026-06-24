from abc import ABC


class Hypothesis(ABC):
    pass


class GoodnessOfFitHypothesis(Hypothesis):
    def __init__(self, parameters: dict[str, float] | None):
        self.parameters = parameters

    def parameters(self) -> list[float]:
        if self.parameters is None:
            return []

        return list(self.parameters.values())


class IndependenceHypothesis(Hypothesis):
    pass
