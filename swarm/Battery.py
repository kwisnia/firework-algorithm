# Paper I'm referring to: Fireworks Algorithm for Optimization

from typing import Callable
from swarm.AlgorithmVariant import AlgorithmVariant
from swarm.Firework import Firework
import numpy as np
from numpy.random import default_rng
from scipy.spatial.distance import cityblock
import sys


random = default_rng()


class Battery:
    def __init__(
        self,
        maximum_amplitude: float,
        dim: int,
        iterations: int,
        population_size: int,
        a: float,
        b: float,
        total_sparks: int,
        gaussian_fireworks: int,
        objective_fun: Callable[[list[float]], float],
        bounds: tuple,
        algorithm_variant: AlgorithmVariant,
        starting_vectors: list[list[float]] = None,
    ) -> None:
        if starting_vectors is not None:
            self.fireworks = [
                Firework(
                    np.array(starting_vector),
                    maximum_amplitude,
                    a,
                    b,
                    total_sparks,
                    objective_fun,
                    bounds,
                    dim,
                    algorithm_variant,
                )
                for _, starting_vector in zip(range(population_size), starting_vectors)
            ]
        else:
            self.fireworks = [
                Firework(
                    np.array(
                        [random.uniform(bounds[0], bounds[1]) for _ in range(dim)]
                    ),
                    maximum_amplitude,
                    a,
                    b,
                    total_sparks,
                    objective_fun,
                    bounds,
                    dim,
                    algorithm_variant,
                )
                for _ in range(population_size)
            ]
        self.iterations = iterations
        self.gaussian_fireworks = gaussian_fireworks
        self.population_size = population_size

    def find_minimum(self):
        best_adaptations = []
        best_actual_adaptation = np.inf
        best_positions = self.fireworks[0].positions
        sparks = []
        for i in range(self.iterations):
            print(i)
            best_firework_adaptation = self.fireworks[0].actual_adaptation
            worst_firework_adaptation = self.fireworks[0].actual_adaptation
            for firework in self.fireworks:
                # Checking if global best changes
                if firework.actual_adaptation < best_actual_adaptation:
                    best_actual_adaptation = firework.actual_adaptation
                    best_positions = firework.positions
                # Checking for worst and best in current population to use in amplitude and number of sparks calculation
                if firework.actual_adaptation > worst_firework_adaptation:
                    worst_firework_adaptation = firework.actual_adaptation
                elif firework.actual_adaptation < best_firework_adaptation:
                    best_firework_adaptation = firework.actual_adaptation

            best_adaptations.append(best_actual_adaptation)
            # Used in equation 2 - number of sparks
            distance_from_worst_sum = (
                np.sum(
                    [
                        worst_firework_adaptation - f.actual_adaptation
                        for f in self.fireworks
                    ]
                )
                + sys.float_info.epsilon
            )
            # Used in equation 4 - amplitude of explosion
            distance_to_best_sum = (
                np.sum(
                    [
                        f.actual_adaptation - best_firework_adaptation
                        for f in self.fireworks
                    ]
                )
                + sys.float_info.epsilon
            )

            for firework in self.fireworks:
                firework.calculate_number_of_sparks(
                    worst_firework_adaptation, distance_from_worst_sum
                )
                firework.calculate_amplitude(
                    best_firework_adaptation, distance_to_best_sum
                )
                # Creating normal sparks for each firework (Algorithm 1 in the paper)
                sparks.append(firework.create_spark(False))
            # Creating gaussian fireworks (Algorithm 2 in the paper)
            for _ in range(self.gaussian_fireworks):
                random_firework = random.choice(self.fireworks)
                sparks.append(random_firework.create_spark(True))
            # Merging sparks with current fireworks to create new explosions generation
            sparks.extend(self.fireworks)
            self.create_next_generation(sparks)

        return best_positions, best_adaptations

    def create_next_generation(self, sparks: list[Firework]) -> None:
        next_generation = []
        spark_probabilities = []
        spark_distance_sums = []
        # Appending the best firework/spark to the next generation
        next_generation.append(
            sparks[np.argmin([firework.actual_adaptation for firework in sparks])]
        )
        for spark in sparks:
            distance_sum = 0.0
            for inner_spark in sparks:
                distance_sum += cityblock(spark.positions, inner_spark.positions)
            spark_distance_sums.append(distance_sum)
        spark_distance_sums_sum = np.sum(spark_distance_sums)
        for distance_sum in spark_distance_sums:
            spark_probabilities.append(distance_sum / spark_distance_sums_sum)
        # Creating new fireworks based on the spark probabilities
        next_generation.extend(
            random.choice(
                sparks,
                size=self.population_size - 1,
                replace=False,
                p=spark_probabilities,
            )
        )
        self.fireworks = next_generation
