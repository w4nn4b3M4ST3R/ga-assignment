import json
import os
import random
import sys
import time

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ga import (
    BitFlipMutation,
    GeneticAlgorithm,
    OnePointCrossover,
    TournamentSelection,
)
from src.problems import KnapsackProblem, OneMaxProblem

import config


def save_results(name, best_fitness, runtime, history):
    # Ensure the reports directory exists
    os.makedirs("reports", exist_ok=True)

    # Save results to a JSON file
    result_data = {
        "best_fitness": best_fitness,
        "runtime": runtime,
        "history": history,
    }
    with open(f"reports/results_oop_{name}.json", "w") as f:
        json.dump(result_data, f, indent=4)

    plt.figure()
    plt.plot(history)
    plt.title(f"OOP GA - {name.capitalize()} Evolution")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.savefig(f"reports/{name}_curve_oop.png")
    plt.close()


def run_experiment():
    cfg = config.GA_CONFIG
    random.seed(cfg["random_seed"])  # For reproducibility

    problems = {
        "onemax": OneMaxProblem(length=cfg["chromosome_length"]),
        "knapsack": KnapsackProblem(
            num_items=cfg["chromosome_length"], random_seed=cfg["random_seed"]
        ),
    }

    for name, problem in problems.items():
        print(f"Running OOP GA for {name}...")
        length = problem.length if hasattr(problem, "length") else problem.num_items

        # Configure strategies
        selection = TournamentSelection(k=cfg["tournament_k"])
        crossover = OnePointCrossover(prob=cfg["crossover_prob"])
        mutation = BitFlipMutation(prob=1 / length)

        ga = GeneticAlgorithm(
            problem=problem,
            pop_size=cfg["population_size"],
            num_generations=cfg["num_generations"],
            selection_strat=selection,
            crossover_strat=crossover,
            mutation_strat=mutation,
            elitism_count=cfg["elitism_count"],
        )

        start_time = time.time()
        best_solution, history = ga.run()
        runtime = time.time() - start_time

        print(
            f"[{name}] Best fitness: {best_solution.fitness}, Runtime: {runtime:.2f}s"
        )
        save_results(name, best_solution.fitness, runtime, history)


if __name__ == "__main__":
    run_experiment()
