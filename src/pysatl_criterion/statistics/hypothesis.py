from abc import ABC, abstractmethod


class Hypothesis(ABC):
    pass


class GoodnessOfFitHypothesis(Hypothesis):
    def __init__(self, parameters: dict[str, float] | None):
        self.parameters = parameters


class IndependenceHypothesis(Hypothesis):
    pass

