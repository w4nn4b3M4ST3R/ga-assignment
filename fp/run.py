import json
import os
import random
import sys
import time
from functools import partial, reduce

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ga import evaluate_population, evolve_generation, init_population
from src.problems import (
    fitness_knapsack,
    fitness_onemax,
    fitness_tuning,
    get_knapsack_items,
)

import config


def run_experiment():
    cfg = config.GA_CONFIG
    random.seed(cfg["random_seed"])  # For reproducibility

    items, capacity = get_knapsack_items(
        num_items=cfg["chromosome_length"], seed=cfg["random_seed"]
    )

    fitness_funcs = {
        "onemax": (fitness_onemax, cfg["chromosome_length"]),
        "knapsack": (
            partial(fitness_knapsack, items=items, capacity=capacity),
            cfg["chromosome_length"],
        ),
        "tuning": (fitness_tuning, cfg["hyperparameter_length"]),
    }

    all_results = {}

    for name, (fitness_func, length) in fitness_funcs.items():
        print(f"Running FP GA for {name}...")

        run_config = {
            "pop_size": cfg["population_size"],
            "tournament_k": cfg["tournament_k"],
            "crossover_prob": cfg["crossover_prob"],
            "mutation_prob": 1 / length,
            "elitism_count": cfg["elitism_count"],
            "fitness_func": fitness_func,
        }

        start_time = time.time()

        # Initial state
        initial_population = init_population(run_config["pop_size"], length)
        initial_state = (initial_population, (), run_config)

        # Evolve generations
        final_state = reduce(
            evolve_generation, range(cfg["num_generations"]), initial_state
        )

        final_population, history, _ = final_state
        final_fitnesses = evaluate_population(final_population, fitness_func)
        best_fitness = max(final_fitnesses)

        runtime = time.time() - start_time

        print(f"[{name}] Best fitness: {best_fitness:.4f}, Runtime: {runtime:.2f}s")

        all_results[name] = {
            "best_fitness": best_fitness,
            "runtime": runtime,
            "history": list(history),  # Convert tuple to list so that JSON can read it
        }

        os.makedirs("reports", exist_ok=True)
        plt.figure()
        plt.plot(history)
        plt.title(f"FP GA - {name.capitalize()} Evolution")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.savefig(f"reports/{name}_curve_fp.png")
        plt.close()

    os.makedirs("reports", exist_ok=True)
    with open("reports/results_fp.json", "w") as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    run_experiment()
