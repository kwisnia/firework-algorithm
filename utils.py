# Gei bi
# M - liczebność populacji
# D - wymiar
# pm - prawdopodobieństwo mutacji

from random import uniform
from numpy import exp, sqrt, sum, cos, pi, abs, sin
import numpy as np

rosenbrock_bounds = (-2.048, 2.048)
ackley_bounds = (-50.0, 50.0)
sphere_bounds = (-100.0, 100.0)
rastrigin_bounds = (-5.12, 5.12)
alpine_bounds = (-10.0, 10.0)
solomon_bounds = (-100.0, 100.0)
schwefel_bounds = (-100.0, 100.0)


def rosenbrock_function(x: list[float]) -> float:
    return sum(
        [100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1)]
    )


def ackley_function(x: list[float]) -> float:
    return (
        -20 * exp(-0.2 * sqrt(1 / len(x) * sum([i ** 2 for i in x])))
        - exp(1 / len(x) * sum([cos(2 * pi * i) for i in x]))
        + 20
        + np.e
    )


def sphere_function(x: list[float]) -> float:
    return sum(x ** 2)


def rastrigin_function(x: list[float]) -> float:
    return sum(x ** 2 - 10 * cos(2 * pi * x) + 10)


def alpine_function(x: list[float]) -> float:
    return sum(abs(x * sin(x) + 0.1 * x))


def solomon_function(x: list[float]) -> float:
    return 1 - cos(2 * pi * sqrt(sum(x ** 2))) + 0.1 * sqrt(sum(x ** 2))


def schwefel_function(x: list[float]) -> float:
    return sum(x * sin(sqrt(abs(x))))


def create_population_member(bounds: tuple, dim: int) -> list:
    return [uniform(bounds[0], bounds[1]) for _ in range(dim)]
