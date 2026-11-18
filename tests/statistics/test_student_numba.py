"""Independent references and nopython coverage for Student numerical kernels."""

import numpy as np
import pytest
from scipy.special import digamma, gammaln
from scipy.stats import t

from pysatl_criterion.statistics.goodness_of_fit import edf
from pysatl_criterion.statistics.goodness_of_fit.continuous import student


class TestStudentNumba:
    @pytest.mark.parametrize("size", [2, 30, 1000])
    @pytest.mark.parametrize("u", [0.0, 0.1, 20.0])
    def test_ecf_integrand(self, size, u):
        sample = np.random.default_rng(size).standard_t(5, size)[::2]
        theoretical, weight = 0.7, 0.3
        expected = abs(np.mean(np.exp(1j * u * sample)) - theoretical) ** 2 * weight
        kernel = student.MeintanisStudentGofStatistic.do_execute_statistic
        assert kernel(sample, u, theoretical, weight) == pytest.approx(expected, abs=1e-14)
        assert kernel.nopython_signatures

    @pytest.mark.parametrize("delta", [0.5, 0.1, 1e-12])
    def test_entropy_reduction(self, delta):
        distances = np.geomspace(1e-100, 1e100, 31)
        gamma_ratio = digamma(3) if delta < 1e-10 else (gammaln(3 + delta) - gammaln(3)) / delta
        if delta < 1e-10:
            expected = np.log(60) + np.mean(np.log(distances)) - gamma_ratio
        else:
            expected = np.log(60) + np.log(np.mean(distances**delta)) / delta - gamma_ratio
        kernel = student.AbstractStudentGofStatistic._renyi_entropy
        assert kernel(distances, delta, gamma_ratio) == pytest.approx(expected, abs=1e-8)
        assert kernel.nopython_signatures

    @pytest.mark.parametrize("magnitude", [1e-300, 1.0, 1e300])
    def test_shape_sample(self, magnitude):
        sample = magnitude * np.array([-2.0, 0.0, 1.0, 3.0])
        original = sample.copy()
        kernel = student.AbstractStudentGofStatistic._shape_sample
        np.testing.assert_allclose(kernel(sample), [-1.0, -0.2, 0.2, 1.0])
        np.testing.assert_array_equal(sample, original)
        assert kernel.nopython_signatures

    @pytest.mark.parametrize(
        "algorithm",
        [
            edf.AbstractAndersonDarlingStatistic,
            edf.AbstractLiaoShimokawaStatistic,
            edf.AbstractZhangKStatistic,
            edf.AbstractZhangAStatistic,
            edf.AbstractZhangCStatistic,
        ],
    )
    @pytest.mark.parametrize("tail", [4.0, 1e50])
    def test_log_edf_kernels(self, algorithm, tail):
        sample = np.array([-tail, -1.0, 0.5, tail])
        logcdf, logsf = t.logcdf(sample, 5), t.logsf(sample, 5)
        kernel = algorithm.do_execute_log_statistic
        expected = kernel.py_func(logcdf, logsf)
        assert kernel(logcdf, logsf) == pytest.approx(expected)
        assert kernel.nopython_signatures
