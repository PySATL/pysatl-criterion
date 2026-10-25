# KolmogorovSmirnov: гиперболическое распределение

`KolmogorovSmirnovHyperbolicGofStatistic` проверяет iid-выборку из полностью заданного
гиперболического распределения на R. Параметры фиксированы:
все четыре значения `alpha`, `beta`, `delta`, `mu` необходимо задать явно;
`alpha > abs(beta)`, `delta > 0`, все параметры — конечные вещественные скаляры.
Подгонки параметров нет.

Для отсортированных наблюдений `u_i=F(x_(i))`, `i=1,...,n`:

`D=max(D+,D−), D+=max(i/n−u_i), D−=max(u_i−(i−1)/n)`.

Дополнительных поправок нет. Критическая область — большие значения.
Минимум одно конечное вещественное наблюдение; повторы и константы допустимы.
Невалидные выборки вызывают `ValueError`, обнаруженные численные ошибки CDF —
`FloatingPointError`. Вызов возвращает `float`, вход не изменяется,
дополнительные kwargs игнорируются.

Класс генерируется из `AbstractKSStatistic` и вычисляет двустороннее расстояние D.
Конструктор принимает только `parameters`; прежние `alternative_type` и `mode`
удалены при унификации с общей EDF-фабрикой.

```python
from pysatl_criterion.distribution import HyperbolicDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.hyperbolic import (
    KolmogorovSmirnovHyperbolicGofStatistic,
)

parameters = HyperbolicDistributionDescriptor.DEFAULT.parse(
    {"alpha": 1.5, "beta": 0.2, "delta": 1, "mu": 0}
)
statistic = KolmogorovSmirnovHyperbolicGofStatistic(parameters)
value = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
assert isinstance(value, float) and value >= 0
```

[Общий EDF-источник](https://doi.org/10.1214/aoms/1177730256) применяется через преобразование
известной непрерывной CDF; это не источник о подгонке гиперболической модели.
Нулевой закон зависит от размера выборки, но не от фиксированных параметров.
Встроенный Monte Carlo resolver не имеет гиперболического генератора.
Для внешнего моделирования используйте `genhyperbolic(p=1, a=alpha*delta,
b=beta*delta, loc=mu, scale=delta)` и вызывайте `execute_statistic` на каждой выборке.

[Полный аудит, ограничения источников и требования к калибровке](mathematical-validation.md).

[Каталог критериев](index.md).

Проверенная вычислительная формула: [Zhang (2002), §2.1](https://doi.org/10.1111/1467-9868.00337), [полный текст](https://anakena.dcc.uchile.cl/~mnmonsal/eso.pdf).
