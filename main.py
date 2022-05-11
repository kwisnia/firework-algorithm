from swarm.AlgorithmVariant import AlgorithmVariant
from swarm.Battery import Battery
from utils import *
import matplotlib.pyplot as plt

POPULATION_SIZE = 5
DIMENSIONS = 30
MAX_SPARKS = 50
A = 0.04
B = 0.8
MAX_AMPLITUDE = 40
GAUSSIAN_SPARKS = 5
ITERATIONS = 50
CHOSEN_FUNCTION = sphere_function
CHOSEN_BOUNDS = sphere_bounds


def plot(
    values: list[float],
    xlabel: str,
    ylabel: str,
    title: str,
    legend: list[str],
    interation_list: list[int],
):
    for index, value_list in enumerate(values):
        plt.plot(range(interation_list[index]), value_list)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(legend)


if __name__ == "__main__":
    results_1 = []
    results_2 = []
    swarm = Battery(
            MAX_AMPLITUDE,
            DIMENSIONS,
            ITERATIONS,
            POPULATION_SIZE,
            A,
            B,
            MAX_SPARKS,
            GAUSSIAN_SPARKS,
            CHOSEN_FUNCTION,
            CHOSEN_BOUNDS,
            AlgorithmVariant.VANILLA,
        )

    best_pos, best_min = swarm.find_minimum()
    results_1.append(best_min[-1])
    print(f"Minimum FWA {best_min[-1]}")
    plot(
        [best_min],
        "Numer iteracji",
        "Najlepsze przystosowanie w populacji",
        "Optymalizacja funkcji algorytmem fajerwerków",
        ["Minimum w baterii"],
        [ITERATIONS],
    )
    plt.show()
