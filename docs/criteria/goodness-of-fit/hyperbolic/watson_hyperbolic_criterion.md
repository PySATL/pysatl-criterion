# Watson: гиперболическое распределение

`WatsonHyperbolicGofStatistic` проверяет iid-выборку из полностью заданного
гиперболического распределения на R. Параметры фиксированы:
все четыре значения `alpha`, `beta`, `delta`, `mu` необходимо задать явно;
`alpha > abs(beta)`, `delta > 0`, все параметры — конечные вещественные скаляры.
Подгонки параметров нет.

Для отсортированных наблюдений `u_i=F(x_(i))`, `i=1,...,n`:

`U²=1/(12n)+Σ(d_i−mean(d))², d_i=u_i−(2i−1)/(2n)`.

Дополнительных поправок нет. Критическая область — большие значения.
Минимум одно конечное вещественное наблюдение; повторы и константы допустимы.
Невалидные выборки вызывают `ValueError`, обнаруженные численные ошибки CDF —
`FloatingPointError`. Вызов возвращает `float`, вход не изменяется,
дополнительные kwargs игнорируются.


```python
from pysatl_criterion.distribution import HyperbolicDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.hyperbolic import (
    WatsonHyperbolicGofStatistic,
)

parameters = HyperbolicDistributionDescriptor.DEFAULT.parse(
    {"alpha": 1.5, "beta": 0.2, "delta": 1, "mu": 0}
)
statistic = WatsonHyperbolicGofStatistic(parameters)
value = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
assert isinstance(value, float) and value >= 0
```

[Общий EDF-источник](https://doi.org/10.1093/biomet/48.1-2.109) применяется через преобразование
известной непрерывной CDF; это не источник о подгонке гиперболической модели.
Нулевой закон зависит от размера выборки, но не от фиксированных параметров.
Встроенный Monte Carlo resolver не имеет гиперболического генератора.
Для внешнего моделирования используйте `genhyperbolic(p=1, a=alpha*delta,
b=beta*delta, loc=mu, scale=delta)` и вызывайте `execute_statistic` на каждой выборке.

[Полный аудит, ограничения источников и требования к калибровке](mathematical-validation.md).

[Каталог критериев](index.md).

Проверенная вычислительная формула: [Lemeshko et al. (2014), формула (6)](https://doi.org/10.3103/S875669901401004X), [полный текст](https://ami.nstu.ru/~headrd/seminar/publik_html/Awt_1_2014_eng.pdf).

При `n=1` статистика тождественно равна `1/12` и не различает альтернативы.
