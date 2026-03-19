import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import unittest

from src.ga import (
    crossover,
    evolve_generation,
    init_population,
    mutate,
    select_tournament,
)
from src.problems import fitness_onemax


class TestGA(unittest.TestCase):
    def test_fitness_evaluation(self):
        # Ensure counting 1s works correctly
        chromosome = (1, 1, 1, 0, 0, 1, 1, 0, 0, 0)
        self.assertEqual(fitness_onemax(chromosome), 5)

    def test_selection(self):
        # Create a mock population with known fitness values
        population = ((0, 0, 0), (1, 1, 1), (1, 0, 1))
        fitnesses = (10, 30, 20)

        # If we take k=3 (draw all 3 for competition), the individual with fitness 30 must win
        selected = select_tournament(population, fitnesses, k=3)

        self.assertEqual(selected, (1, 1, 1))

    def test_crossover(self):
        p1 = (0,) * 10
        p2 = (1,) * 10
        random.seed(42)  # Set seed to ensure deterministic crossover point
        c1, c2 = crossover(p1, p2, prob=1.0)

        self.assertEqual(len(c1), 10)
        self.assertEqual(len(c2), 10)
        self.assertIn(1, c1)
        self.assertIn(0, c1)

    def test_mutate(self):
        c = (0, 0, 0, 0)
        mutated_c = mutate(c, prob=1.0)
        self.assertEqual(mutated_c, (1, 1, 1, 1))
        self.assertEqual(c, (0, 0, 0, 0))  # Ensure original tuple is not modified

    def test_evolve_generation_improvement(self):
        # Initialize a very small state to test one evolution step
        config = {
            "pop_size": 10,
            "tournament_k": 2,
            "crossover_prob": 0.9,
            "mutation_prob": 0.05,
            "elitism_count": 1,
            "fitness_func": fitness_onemax,
        }
        pop = init_population(config["pop_size"], 10)
        initial_state = (pop, (), config)

        next_state = evolve_generation(initial_state, 0)
        next_pop, next_history, _ = next_state

        self.assertEqual(len(next_pop), 10)
        self.assertEqual(len(next_history), 1)


if __name__ == "__main__":
    unittest.main()
