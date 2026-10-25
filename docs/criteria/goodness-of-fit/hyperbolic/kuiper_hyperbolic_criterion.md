# Kuiper: гиперболическое распределение

`KuiperHyperbolicGofStatistic` проверяет iid-выборку из полностью заданного
гиперболического распределения на R. Параметры фиксированы:
все четыре значения `alpha`, `beta`, `delta`, `mu` необходимо задать явно;
`alpha > abs(beta)`, `delta > 0`, все параметры — конечные вещественные скаляры.
Подгонки параметров нет.

Для отсортированных наблюдений `u_i=F(x_(i))`, `i=1,...,n`:

`V=D++D−, D+=max(i/n−u_i), D−=max(u_i−(i−1)/n)`.

Дополнительных поправок нет. Критическая область — большие значения.
Минимум одно конечное вещественное наблюдение; повторы и константы допустимы.
Невалидные выборки вызывают `ValueError`, обнаруженные численные ошибки CDF —
`FloatingPointError`. Вызов возвращает `float`, вход не изменяется,
дополнительные kwargs игнорируются.


```python
from pysatl_criterion.distribution import HyperbolicDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.hyperbolic import (
    KuiperHyperbolicGofStatistic,
)

parameters = HyperbolicDistributionDescriptor.DEFAULT.parse(
    {"alpha": 1.5, "beta": 0.2, "delta": 1, "mu": 0}
)
statistic = KuiperHyperbolicGofStatistic(parameters)
value = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
assert isinstance(value, float) and value >= 0
```

[Общий EDF-источник](https://doi.org/10.1016/S1385-7258(60)50006-0) применяется через преобразование
известной непрерывной CDF; это не источник о подгонке гиперболической модели.
Нулевой закон зависит от размера выборки, но не от фиксированных параметров.
Встроенный Monte Carlo resolver не имеет гиперболического генератора.
Для внешнего моделирования используйте `genhyperbolic(p=1, a=alpha*delta,
b=beta*delta, loc=mu, scale=delta)` и вызывайте `execute_statistic` на каждой выборке.

[Полный аудит, ограничения источников и требования к калибровке](mathematical-validation.md).

[Каталог критериев](index.md).

Проверенная вычислительная формула: [Lemeshko et al. (2014), формула (1)](https://doi.org/10.3103/S875669901401004X), [полный текст](https://ami.nstu.ru/~headrd/seminar/publik_html/Awt_1_2014_eng.pdf).

При `n=1` статистика тождественно равна `1` и не различает альтернативы.
