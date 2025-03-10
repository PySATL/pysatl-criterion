# Test for exponentiality of Epps and Pulley

## Description
Performs Epps and Pulley test for the composite hypothesis of exponentiality, see e.g. Henze and Meintanis (2005, Sec. 2.8.1).
The Epps and Pulley test is a statistical hypothesis test used to assess the composite hypothesis of exponentiality. This test is designed to determine whether a given sample of data is consistent with an exponential distribution, which is a common assumption in various fields such as reliability engineering, survival analysis, and queuing theory.

Composite Hypothesis of Exponentiality
The composite hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with an unspecified rate parameter 
λ. Unlike a simple hypothesis, where the parameter λ is known, the composite hypothesis allows λ to be any positive value, making the test more flexible and widely applicable.

Test Statistic
The Epps and Pulley test statistic is based on the empirical characteristic function (ECF) of the data. The characteristic function of the exponential distribution has a specific form, and the test compares the ECF of the sample to the theoretical characteristic function of the exponential distribution. The test statistic measures the discrepancy between these two functions.

Calculate the Test Statistic: The test statistic is derived from the integrated squared difference between the ECF and the theoretical characteristic function of the standard exponential distribution.

Limitations
The test may require large sample sizes to achieve high power.

The computation of the test statistic can be more complex compared to simpler tests like the Kolmogorov-Smirnov test.

## Usage

## Arguments

## Details

The Epps and Pulley test is a test for the composite hypothesis of exponentiality. The test statistic is
$$EP_n = (48n)^{1/2} [ \frac{1}{n} \sum_{j=1}^{n} exp(-Y_j) -\frac{1}{2} ]$$

where $Y_j=X_j/\overline{X}$. EP_n is asymptotically standard normal (see, e.g., Henze and Meintanis (2005, Sec. 2.8.1).

## Author(s)
Lev Golofastov

## References
Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. — Metrika, vol. 61, pp. 29–45.

## Examples
