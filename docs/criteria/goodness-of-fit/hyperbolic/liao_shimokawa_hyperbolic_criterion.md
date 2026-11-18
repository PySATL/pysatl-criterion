# Liao–Shimokawa для гиперболического распределения

`LiaoShimokawaHyperbolicGofStatistic` создаётся через `_edf_factory.py`
и использует общее ядро `AbstractLiaoShimokawaStatistic` из `edf.py`.
Код: `LIAO_SHIMOKAWA_HYPERBOLIC_GOODNESS_OF_FIT`.

Все четыре параметра гиперболического распределения фиксированы заранее.
Для отсортированных наблюдений и значений CDF $u_i=F(x_{(i)})$:

$$
L_n=\frac1{\sqrt n}\sum_{i=1}^n
\frac{\max(i/n-u_i,\ u_i-(i-1)/n)}{\sqrt{u_i(1-u_i)}}.
$$

Большие значения статистики служат основанием для отклонения гипотезы.
Для вычисления весов используются логарифмы CDF и функции выживания.
Калибровка должна соответствовать полностью заданной гиперболической гипотезе.

```python
from pysatl_criterion.distribution import HyperbolicDistributionDescriptor as Distribution
from pysatl_criterion.statistics.goodness_of_fit.continuous.hyperbolic import (
    LiaoShimokawaHyperbolicGofStatistic,
)

parameters = Distribution.DEFAULT.parse({"alpha": 1.5, "beta": 0.25, "delta": 2, "mu": 0})
criterion = LiaoShimokawaHyperbolicGofStatistic(parameters)
value = criterion.execute_statistic([-2, -0.5, 0, 0.7, 2])
```

Описание общего ядра: Liao and Shimokawa (1999),
[статья](https://doi.org/10.1080/00949659908811965).
