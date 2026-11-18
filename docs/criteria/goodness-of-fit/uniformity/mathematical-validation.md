# Mathematical verification of uniformity statistics

The 25 criteria were checked against the formulas and sources below. All bounds
are fixed before testing. For EDF statistics, write $u_i=F_{a,b}(x_{(i)})$;
for spacing statistics include $u_0=0$ and $u_{n+1}=1$.

## Formula checks

| Criteria | Checked definition | Scientific source |
|---|---|---|
| Kolmogorov–Smirnov D+, D−, D | Maximum positive and negative EDF discrepancies; D is their maximum, without squaring or scaling | [Zhang (2002), §2.1](https://doi.org/10.1111/1467-9868.00337) |
| Anderson–Darling | Weighted logarithmic sum with factor $1/n$ and subtraction of $n$ | [Anderson and Darling (1952)](https://doi.org/10.1214/aoms/1177729437); [Zhang (2002), §2.2](https://doi.org/10.1111/1467-9868.00337) |
| Cramér–von Mises | Squared deviations from $(i-1/2)/n$, plus $1/(12n)$ | [Zhang (2002), §2.3](https://doi.org/10.1111/1467-9868.00337) |
| Watson | $W^2-n(\bar u-1/2)^2$, evaluated through centered deviations | [Watson (1961)](https://doi.org/10.1093/biomet/48.1-2.109); [Lemeshko et al. (2014), eq. (6)](https://doi.org/10.3103/S875669901401004X) |
| Kuiper, scaled Kuiper, corrected Kuiper | $V=D^++D^-$; $\sqrt n V$; $(\sqrt n+0.155+0.24/\sqrt n)V$ | [Lemeshko et al. (2014), eqs. (1)–(3)](https://doi.org/10.3103/S875669901401004X) |
| Zhang K, A, C | Likelihood-ratio expressions with half-ranks and the quarter-rank log-odds correction | [Zhang (2002), eqs. (3.1)–(3.3)](https://doi.org/10.1111/1467-9868.00337) |
| Liao–Shimokawa | Sum of local maximum EDF deviations divided by $\sqrt{u_i(1-u_i)}$ and $\sqrt n$ | [Liao and Shimokawa (1999)](https://doi.org/10.1080/00949659908811965) |
| Pearson, Freeman–Tukey, modified log-likelihood, Neyman chi-squared, Cressie–Read | Power divergences at $\lambda=1,-1/2,-1,-2,2/3$; zero-count limits checked separately | [Cressie and Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |
| Greenwood | Sum of squares of all $n+1$ spacings | [Does, Helmers and Klaassen (1988), eqs. (1)–(2)](https://ir.cwi.nl/pub/1694/1694D.pdf) |
| Sherman | Half the sum of absolute spacing deviations from $1/(n+1)$ | [Sherman (1950)](https://doi.org/10.1214/aoms/1177729794) |
| Quesenberry–Miller | Squared spacings plus adjacent-spacing products, with no cyclic product | [Quesenberry and Miller (1977)](https://doi.org/10.1080/00949657708810150) |
| Neyman smooth | Sum of squared empirical coefficients in the orthonormal shifted Legendre basis, excluding the constant | [Neyman (1937)](https://doi.org/10.1080/03461238.1937.10404821); [Rayner and Rayner (2001), §2](https://doi.org/10.1155/S117391260100013X) |
| Stein, censored Stein | Pairwise signed U-statistic and its inverse-censoring-probability weighted form; denominator $n(n-1)/2$ | [Sreedevi and Kattumannil (2023)](https://doi.org/10.1007/s42952-023-00205-8), [author preprint, §§2–3](https://arxiv.org/html/2106.06368v1) |
| Bickel–Rosenblatt-type | Gaussian density estimate integrated squared error over $[0,1]$ | Quadratic discrepancy motivated by [Bickel and Rosenblatt (1973)](https://doi.org/10.1214/aos/1176342558); the precise integral is a library adaptation |

## Scope and qualifications

No formula changes were needed for the definitions checked. This verifies
statistic calculations, not exact finite-sample rejection probabilities or
power against every alternative.

Bickel–Rosenblatt returns the raw integral with its specified bandwidth policy,
without asymptotic centering or scaling. Liao–Shimokawa uses the specified uniform
null here; fitted-model critical values from its original paper do not apply.

Stein statistics are signed and use both tails. Censoring flags use 1 for censored
observations, opposite to the paper's event indicator. The censoring estimator
uses left limits and an explicit tie convention. The returned statistic omits
the paper's variance estimation and studentization.

Where original full text was inaccessible, bibliographic verification was
separated from formula verification: Sherman and Quesenberry–Miller formulas
were checked in the authors' comparative study in [§4.1 of the Stein preprint](https://arxiv.org/html/2106.06368v1#S4.SS1),
and Neyman's polynomial form in Rayner and Rayner (2001). The Liao–Shimokawa
expression was additionally compared with its use in a [published goodness-of-fit
study](https://www.m-hikari.com/ams/ams-2012/ams-125-128-2012/alzahraniAMS125-128-2012.pdf).
The Bickel–Rosenblatt integral was verified directly by quadrature, rather than
claimed to reproduce the original asymptotic statistic.

## Numerical verification

Checks include independent EDF integrals, SciPy reference values, high-precision
Zhang examples, published Legendre polynomials, the order-statistic representation
of Stein's pairwise sum, hand-calculated censoring weights, and numerical
integration of the density discrepancy. Boundary values, empty bins, tied
observations and affine changes of fixed uniform bounds are covered.

The focused suite passed 542 tests. The run isolated the pre-existing obsolete
imports in `continuous/__init__.py`; it does not establish that ordinary package
initialization works. Every concrete uniform criterion's docstring now includes
a scientific reference and a URL.
