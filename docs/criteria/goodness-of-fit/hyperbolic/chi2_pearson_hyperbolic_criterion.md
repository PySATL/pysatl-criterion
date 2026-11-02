# Pearson χ²: гиперболическое распределение

`Chi2PearsonHyperbolicGofStatistic` (`CHI2_PEARSON_HYPERBOLIC_GOODNESS_OF_FIT`) проверяет
полностью заданную гиперболическую iid-модель на вещественной оси.
Все параметры заданы заранее: конечные `alpha > abs(beta)`, `delta > 0`, `mu`.
Неизвестные параметры и их оценивание этим классом не поддерживаются.

Пусть `O_j` — наблюдаемая частота, `E_j=n p_j>0` — ожидаемая частота в
j-й ячейке, `p_j` — её вероятность под нулевой гипотезой.
Обе суммы частот равны n. Статистика:

`T = Σ(O_j−E_j)²/E_j`.

Критическая область правая: большие T свидетельствуют против гипотезы.
Пустая ячейка вносит E_j.

## Разбиение и калибровка

Конструктор принимает `parameters` и необязательный `binning`.
По умолчанию используется `EqualProbabilityBins()`: наблюдения преобразуются
известной CDF, а `[0,1]` разбивается на равновероятные ячейки; `E_j=n/k`.
Количество ячеек выбирается общим правилом `PowerCount`, минимум две.
Пример ниже фиксирует две ячейки явно.

Можно передать другие [стратегии разбиения](../binning/index.md), включая
`ExplicitBins` с границами, покрывающими весь носитель. Внутренняя граница
принадлежит ячейке справа; последняя включает правый конец.
Правила разбиения, числа ячеек и объединения входят в конфигурацию калибровки.

Нужна непустая одномерная конечная вещественная выборка. Повторы и константы
допустимы; вход не изменяется. Это не гарантирует точности асимптотической
калибровки при малом числе наблюдений. Для заранее фиксированных k ячеек с
положительными вероятностями и известными параметрами применима асимптотика
χ² с k−1 степенями свободы. Для малых ожидаемых частот или адаптивных границ
нужно моделирование, воспроизводящее весь алгоритм разбиения.
Встроенный resolver не имеет гиперболического генератора: моделируйте внешне
через `scipy.stats.genhyperbolic(p=1, a=alpha*delta, b=beta*delta, loc=mu, scale=delta)`.
Класс возвращает статистику, а не p-value. Численные ошибки вероятностей
при построении таблицы приводят к исключению, не к скрытому clipping.

## Пример

```python
from pysatl_criterion.distribution import HyperbolicDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous.hyperbolic import (
    Chi2PearsonHyperbolicGofStatistic,
)

parameters = HyperbolicDistributionDescriptor.DEFAULT.parse(
    {"alpha": 1.5, "beta": 0, "delta": 1, "mu": 0}
)
criterion = Chi2PearsonHyperbolicGofStatistic(parameters, binning=EqualProbabilityBins(count=2))
value = criterion.execute_statistic([-2.0, -1.0, -0.5, 1.0])
assert isinstance(value, float) and value >= 0
```

## Научный источник

[Cressie N., Read T. R. C. (1984), Multinomial Goodness-of-Fit Tests,
JRSS B 46(3), 440–464](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x):
семейство power divergence при `λ=1`.
[Полный текст](https://pages.stern.nyu.edu/~dbackus/Exotic/1Robustness/CressieRead%20JRSSB%2084.pdf).
Применение к гиперболической модели использует вероятности её ячеек;
выбор конкретной стратегии разбиения является настройкой библиотеки.

[Все критерии](index.md) · [Математическая проверка](mathematical-validation.md).
