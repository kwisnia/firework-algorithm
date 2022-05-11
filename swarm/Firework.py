from numpy import clip
from numpy.random import default_rng

from typing import Callable

import sys
from copy import deepcopy

from swarm.AlgorithmVariant import AlgorithmVariant

random = default_rng()

class Firework:
    def __init__(
        self,
        initial_positions: list[float],
        maximum_amplitude: float,
        a: float,
        b: float,
        total_sparks: int,
        adaptation_function: Callable[[list[float]], float],
        bounds: tuple,
        dimensions: int,
        algorithm_variant: AlgorithmVariant
    ):
        self.dimensions = dimensions
        self.maximum_amplitude = maximum_amplitude
        self.a = a
        self.b = b
        self.total_sparks = total_sparks
        self.adaptation_function = adaptation_function
        self.bounds = bounds
        self.algorithm_variant = algorithm_variant

        self.positions = initial_positions

        self.actual_adaptation = self.calculate_adaptation()

    def calculate_adaptation(self) -> float:
        return self.adaptation_function(self.positions)

    def calculate_number_of_sparks(
        self, worst_firework_adaptation: float, sum: float
    ) -> int:
        number_of_sparks = (
            self.total_sparks
            * (
                worst_firework_adaptation
                - self.actual_adaptation
                + sys.float_info.epsilon
            )
            / sum
        )
        if number_of_sparks < self.a * self.total_sparks:
            self.number_of_sparks = round(self.a * self.total_sparks)
        elif number_of_sparks > self.b * self.total_sparks:
            self.number_of_sparks = round(self.b * self.total_sparks)
        else:
            self.number_of_sparks = round(number_of_sparks)

    def calculate_amplitude(self, best_firework_adaptation: float, sum: float) -> float:
        self.amplitude = (
            self.maximum_amplitude
            * (
                self.actual_adaptation
                - best_firework_adaptation
                + sys.float_info.epsilon
            )
            / sum
        )

    def create_spark(self, gaussian: bool, best_position=None) -> None:
        spark = deepcopy(self)
        number_of_displacements = round(self.dimensions * random.random())
        selected_dimensions = random.choice(
            range(self.dimensions), number_of_displacements, replace=False)
        if gaussian:
            if self.algorithm_variant == AlgorithmVariant.NEW_GAUSSIAN:
                coeffecient = random.normal(0, 1)
                for i in selected_dimensions:
                    spark.positions[i] += (best_position[i] - spark.positions[i]) * coeffecient
            else:
                coeffecient = random.normal(1, 1)
                for i in selected_dimensions:
                    spark.positions[i] = spark.positions[i] * coeffecient
        else:
            displacement = self.maximum_amplitude * random.uniform(-1, 1)
            for i in selected_dimensions:
                spark.positions[i] += displacement
        spark.positions = clip(spark.positions, self.bounds[0], self.bounds[1])
        spark.update_adaptation()
        return spark

    def update_adaptation(self) -> None:
        self.actual_adaptation = self.calculate_adaptation()
