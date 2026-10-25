# Критерии гиперболического распределения

Пакет `pysatl_criterion.statistics.goodness_of_fit.continuous.hyperbolic`
экспортирует базовый `AbstractHyperbolicGofStatistic` и десять конкретных критериев: пять EDF и пять частотных.
Все проверяют полностью заданную непрерывную iid-модель на всей вещественной оси.
Задайте все четыре параметра явно: `alpha > abs(beta)`, `delta > 0`, конечный `mu`.
Неуказанные параметры считаются неизвестными; составная гипотеза не поддерживается.

| Публичный класс и описание | Полный код |
| --- | --- |
| [AndersonDarlingHyperbolicGofStatistic](anderson_darling_hyperbolic_criterion.md) | `AD_HYPERBOLIC_GOODNESS_OF_FIT` |
| [CramerVonMisesHyperbolicGofStatistic](cramer_von_mises_hyperbolic_criterion.md) | `CVM_HYPERBOLIC_GOODNESS_OF_FIT` |
| [KolmogorovSmirnovHyperbolicGofStatistic](kolmogorov_smirnov_hyperbolic_criterion.md) | `KS_HYPERBOLIC_GOODNESS_OF_FIT` |
| [KuiperHyperbolicGofStatistic](kuiper_hyperbolic_criterion.md) | `KUI_HYPERBOLIC_GOODNESS_OF_FIT` |
| [WatsonHyperbolicGofStatistic](watson_hyperbolic_criterion.md) | `WAT_HYPERBOLIC_GOODNESS_OF_FIT` |
| [Chi2PearsonHyperbolicGofStatistic](chi2_pearson_hyperbolic_criterion.md) | `CHI2_PEARSON_HYPERBOLIC_GOODNESS_OF_FIT` |
| [FreemanTukeyHyperbolicGofStatistic](freeman_tukey_hyperbolic_criterion.md) | `FREEMAN_TUKEY_HYPERBOLIC_GOODNESS_OF_FIT` |
| [ModifiedLogLikelihoodHyperbolicGofStatistic](modified_log_likelihood_hyperbolic_criterion.md) | `MODIFIED_LOG_LIKELIHOOD_HYPERBOLIC_GOODNESS_OF_FIT` |
| [NeymanChi2HyperbolicGofStatistic](neyman_chi2_hyperbolic_criterion.md) | `CHI2_NEYMAN_HYPERBOLIC_GOODNESS_OF_FIT` |
| [CressieReadHyperbolicGofStatistic](cressie_read_hyperbolic_criterion.md) | `CRESSIE_READ_HYPERBOLIC_GOODNESS_OF_FIT` |

Все критерии отклоняют гипотезу при больших значениях статистики.
KS вычисляет двустороннее расстояние D; пять EDF-классов генерируются EDF-фабрикой.
Частотные критерии генерируются общей frequency-фабрикой и принимают `binning`.

[Математическая проверка, совместимость и измерения](mathematical-validation.md).
