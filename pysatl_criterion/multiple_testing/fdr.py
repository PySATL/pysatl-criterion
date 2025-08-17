from .abstract_multiple_testing import AbstractMultipleTesting


class BenjaminiHochberg(AbstractMultipleTesting):
    """
    Implements the Benjamini-Hochberg procedure for False Discovery Rate (FDR) control.
    Inherits from AbstractMultipleTesting.
    """

    @classmethod
    def adjust(cls, p_values: list[float]) -> list[float]:
        """
        Compute adjusted p-values using the Benjamini-Hochberg procedure to control the FDR.
        The procedure aims to control the expected proportion of Type I errors among
        rejected hypotheses.

        Algorithm steps based on the provided document:
        1. (Implicitly handled by _validate_p_values and sorting)
        2. Compute p-values (input `p_values`).
        3. Order the m p-values so that p_(1) <= p_(2) <= ... <= p_(m).
        4. Calculate adjusted p-values starting from the largest ordered p-value
           and working backward
           to ensure monotonicity (p_adj[k] = min(bh_value, p_adj[k+1])).

        :param p_values: List of raw p-values for hypothesis testing.
        :return: List of adjusted p-values.
        """
        cls._validate_p_values(p_values)

        m = len(p_values)
        if m == 0:
            return []

        # 3. Order the m p-values so that p_(1) <= p_(2) <= ... <= p_(m).
        # We need to keep track of the original indices to reorder later.
        indexed_p_values = sorted([(p, i) for i, p in enumerate(p_values)])
        sorted_p_values = [p for p, _ in indexed_p_values]
        original_indices = [i for _, i in indexed_p_values]

        p_values_adjusted = [0.0] * m

        # Calculate adjusted p-values working backward to ensure monotonicity.
        # The formula for the k-th ordered p-value (1-based index) is p_k * (m / k).
        # When working backward, the adjusted p-value for rank k
        # is min(p_k * (m/k), adjusted_p_value_for_k+1).
        p_values_adjusted[m - 1] = (
            sorted_p_values[m - 1] * m / m
        )  # For the last one, k=m, so p_m * m/m = p_m

        for k_idx in range(m - 2, -1, -1):  # k_idx is 0-based index
            # k_rank is 1-based index
            k_rank = k_idx + 1
            # Calculate the Benjamini-Hochberg value for the current rank
            bh_value = sorted_p_values[k_idx] * m / k_rank
            # Сurrent adjusted p-value must be less than or equal to the next one
            p_values_adjusted[k_idx] = min(bh_value, p_values_adjusted[k_idx + 1])

        # Cap adjusted p-values at 1.0
        p_values_adjusted = [min(p, 1.0) for p in p_values_adjusted]

        # Reorder the adjusted p-values back to the original input order
        final_adjusted_p_values = [0.0] * m
        for i, original_index in enumerate(original_indices):
            final_adjusted_p_values[original_index] = p_values_adjusted[i]

        return final_adjusted_p_values

    @classmethod
    def test(cls, p_values: list[float], q: float = 0.05) -> tuple[list[bool], list[float]]:
        """
        Perform the Benjamini-Hochberg procedure to control the False Discovery Rate (FDR).
        This method overrides the base class's `test` method to specifically use
        the 'q' parameter as the FDR threshold, as described in the algorithm.

        Algorithm steps based on the provided document:
        1. Specify q, the level at which to control the FDR.
        2. Compute p-values (input `p_values`).
        3. Order the m p-values so that p_(1) <= p_(2) <= ... <= p_(m).
        4. Define L = max{j : p_(j) < qj/m}.
        5. Reject all null hypotheses H_0j for which p_j <= p_(L).

        Note: The implementation here simplifies step 5 by comparing the adjusted p-values
        (computed in `adjust`) directly to 'q', which is mathematically equivalent to
        finding 'L' and rejecting based on it for the Benjamini-Hochberg procedure.

        :param p_values: List of raw p-values for hypothesis testing.
        :param q: The desired False Discovery Rate (FDR) level (default is 0.05).
        :return: Tuple containing:
                - Boolean list indicating rejected hypotheses (True where rejected)
                - List of adjusted p-values after Benjamini-Hochberg correction
        """
        cls._validate_p_values(p_values)

        m = len(p_values)
        if m == 0:
            return [], []

        # Compute adjusted p-values using the Benjamini-Hochberg adjustment
        adjusted_p_values = cls.adjust(p_values)

        # Reject all null hypotheses whose adjusted p-value is less than or equal to q.
        rejected = [adj_p <= q for adj_p in adjusted_p_values]

        return rejected, adjusted_p_values
