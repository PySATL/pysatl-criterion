from abc import ABC

import numpy as np
from numpy import float64
from typing_extensions import override

from pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic


class AbstractGraphTestStatistic(AbstractGoodnessOfFitStatistic, ABC):
    @override
    def execute_statistic(self, rvs, **kwargs) -> float | float64:
        dist = self._compute_dist(rvs)

        adjacency_list = self._make_adjacency_list(rvs, dist)
        statistic = self.get_graph_stat(adjacency_list)
        return statistic

    @staticmethod
    def get_graph_stat(graph: list[list[int]]) -> float:
        raise NotImplementedError("Method is not implemented")

    @staticmethod
    def _make_adjacency_list(rvs, dist: float) -> list[list[int]]:
        adjacency_list: list[list[int]] = []

        for i in range(len(rvs)):
            adjacency_list.append([])
            for j in range(i):
                if abs(rvs[i] - rvs[j]) < dist:
                    adjacency_list[i].append(j)
                    adjacency_list[j].append(i)

        return adjacency_list

    @staticmethod
    def _compute_dist(rvs: list[float]) -> float:  # TODO (normalize for different distributions)
        return (max(rvs) - min(rvs)) / 10


class GraphEdgesNumberTestStatistic(AbstractGraphTestStatistic):
    @staticmethod
    @override
    def get_graph_stat(graph: list[list[int]]) -> float:
        return sum(map(len, graph)) // 2

    @staticmethod
    @override
    def short_code() -> str:
        return "EDGESNUMBER"


class GraphMaxDegreeTestStatistic(AbstractGraphTestStatistic):
    @staticmethod
    @override
    def get_graph_stat(graph: list[list[int]]) -> float:
        return max(map(len, graph))

    @staticmethod
    @override
    def short_code() -> str:
        return "MAXDEGREE"


class GraphAverageDegreeTestStatistic(AbstractGraphTestStatistic):
    @staticmethod
    @override
    def get_graph_stat(graph: list[list[int]]) -> float:
        degrees = list(map(len, graph))
        return float(np.mean(degrees)) if degrees != 0 else 0.0

    @staticmethod
    @override
    def short_code() -> str:
        return "AVGDEGREE"


class GraphConnectedComponentsTestStatistic(AbstractGraphTestStatistic):
    @staticmethod
    @override
    def get_graph_stat(graph) -> float:
        visited = set()
        components = 0

        def dfs(node):
            stack = [node]
            while stack:
                v = stack.pop()
                if v not in visited:
                    visited.add(v)
                    stack.extend(neighbor for neighbor in graph[v] if neighbor not in visited)

        for node in range(len(graph)):
            if node not in visited:
                dfs(node)
                components += 1
        return components

    @staticmethod
    @override
    def short_code() -> str:
        return "CONNECTEDCOMPONENTS"


class GraphCliqueNumberTestStatistic(AbstractGraphTestStatistic):
    @override
    def execute_statistic(self, rvs, **kwargs) -> float | float64:
        dist = self._compute_dist(rvs)
        rvs.sort()

        right_border = 0
        clique_number = 0
        for left_border in range(len(rvs)):
            while right_border < len(rvs) and rvs[left_border] + dist > rvs[right_border]:
                right_border += 1
            if right_border == len(rvs):
                clique_number = max(clique_number, right_border - left_border + 1)
                break
            clique_number = max(clique_number, right_border - left_border)
        return clique_number

    @staticmethod
    @override
    def short_code() -> str:
        return "CLIQUENUMBER"


class GraphIndependenceNumberTestStatistic(AbstractGraphTestStatistic):
    @override
    def execute_statistic(self, rvs, **kwargs) -> float | float64:
        if not rvs:
            return 0

        dist = self._compute_dist(rvs)
        rvs.sort()

        stat = 1
        last_chosen_position = rvs[0]

        for i in range(1, len(rvs)):
            current_point = rvs[i]
            if current_point >= last_chosen_position + dist:
                stat += 1
                last_chosen_position = current_point

        return stat

    @staticmethod
    @override
    def short_code() -> str:
        return "INDEPENDENCENUMBER"
