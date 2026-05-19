from abc import ABC

import numpy as np
from numpy import float64
from typing_extensions import override

from pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic


class AbstractGraphTestStatistic(AbstractGoodnessOfFitStatistic, ABC):
    """
    Abstract base class for graph-based goodness-of-fit statistics.
    """

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | float64:
        dist = self._compute_dist(rvs)

        adjacency_list = self._make_adjacency_list(rvs, dist)
        statistic = self.get_graph_stat(adjacency_list)
        return statistic

    @staticmethod
    def get_graph_stat(graph: list[list[int]]) -> float:
        """
        Compute the specific graph statistic from the adjacency list.

        :param graph: adjacency list representation of the proximity graph.
        :return: computed statistic value.
        :raises NotImplementedError: if not implemented by subclass.
        """
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
    """
    Graph test statistic based on the total number of edges.
    """

    @staticmethod
    @override
    def get_graph_stat(graph: list[list[int]]) -> float:
        """
        Calculate the total number of edges in the graph.

        :param graph: adjacency list representation of the proximity graph.
        :return: total number of edges (undirected).
        """

        return sum(map(len, graph)) // 2

    @staticmethod
    @override
    def short_code() -> str:
        """
        Get short code identifier for this test.

        :return: short code string "EDGESNUMBER".
        """
        return "EDGESNUMBER"


class GraphMaxDegreeTestStatistic(AbstractGraphTestStatistic):
    """
    Graph test statistic based on the maximum vertex degree.
    """

    @staticmethod
    @override
    def get_graph_stat(graph: list[list[int]]) -> float:
        """
        Calculate the maximum degree among all vertices in the graph.

        :param graph: adjacency list representation of the proximity graph.
        :return: maximum vertex degree.
        """
        return max(map(len, graph))

    @staticmethod
    @override
    def short_code() -> str:
        """
        Get short code identifier for this test.

        :return: short code string "MAXDEGREE".
        """
        return "MAXDEGREE"


class GraphAverageDegreeTestStatistic(AbstractGraphTestStatistic):
    """
    Graph test statistic based on the average vertex degree.
    """

    @staticmethod
    @override
    def get_graph_stat(graph: list[list[int]]) -> float:
        """
        Calculate the average degree of vertices in the graph.

        :param graph: adjacency list representation of the proximity graph.
        :return: average vertex degree (0.0 if graph is empty).
        """
        degrees = list(map(len, graph))
        return float(np.mean(degrees)) if degrees != 0 else 0.0

    @staticmethod
    @override
    def short_code() -> str:
        """
        Get short code identifier for this test.

        :return: short code string "AVGDEGREE".
        """
        return "AVGDEGREE"


class GraphConnectedComponentsTestStatistic(AbstractGraphTestStatistic):
    """
    Graph test statistic based on the number of connected components.
    """

    @staticmethod
    @override
    def get_graph_stat(graph) -> float:
        """
        Calculate the number of connected components in the graph using DFS.

        :param graph: adjacency list representation of the proximity graph.
        :return: number of connected components.
        """
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
        """
        Get short code identifier for this test.

        :return: short code string "CONNECTEDCOMPONENTS".
        """
        return "CONNECTEDCOMPONENTS"


class GraphCliqueNumberTestStatistic(AbstractGraphTestStatistic):
    """
    Graph test statistic based on the maximum clique size.
    """

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | float64:
        """
        Execute the clique number test statistic for 1D data.

        :param rvs: array of observed data samples.
        :return: size of the maximum clique.
        """
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
        """
        Get short code identifier for this test.

        :return: short code string "CLIQUENUMBER".
        """
        return "CLIQUENUMBER"


class GraphIndependenceNumberTestStatistic(AbstractGraphTestStatistic):
    """
    Graph test statistic based on the independence number.
    """

    @override
    def execute_statistic(self, rvs, **kwargs) -> float | float64:
        """
        Execute the independence number test statistic for 1D data.

        :param rvs: array of observed data samples.
        :return: size of the maximum independent set.
        """
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
        """
        Get short code identifier for this test.

        :return: short code string "INDEPENDENCENUMBER".
        """
        return "INDEPENDENCENUMBER"
