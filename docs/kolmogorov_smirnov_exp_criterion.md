# Test for exponentiality of Kolmogorov and Smirnov

## Description

The Kolmogorov-Smirnov (KS) test for exponentiality is a non-parametric statistical test used to determine whether a given sample of data follows an exponential distribution. It is a goodness-of-fit test that compares the empirical distribution function (EDF) of the sample data with the cumulative distribution function (CDF) of the exponential distribution. The test is named after Andrey Kolmogorov and Nikolai Smirnov, who developed the foundational concepts behind the test.

## Usage

## Arguments

## Details
The Kolmogorov-Smirnov test for exponentiality is based on the following statistic:

$$KS_n =\sup_{x≥q0}|F_n(x)-(1-\exp(-x))|, $$

where $F_n$ is the empirical distribution function of the scaled data $Y_j=X_j/\overline{X}$.

## Author(s)
Lev Golofastov

## References
Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. — Metrika, vol. 61, pp. 29–45.

## Examples
