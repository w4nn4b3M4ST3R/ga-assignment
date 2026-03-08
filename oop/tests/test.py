import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from src.ga import (
    BitFlipMutation,
    Chromosome,
    GeneticAlgorithm,
    OnePointCrossover,
    TournamentSelection,
)
from src.problems import OneMaxProblem


class TestGA(unittest.TestCase):
    def test_fitness_evaluation(self):
        # Ensure counting 1s works correctly
        problem = OneMaxProblem(length=10)
        chromosome = Chromosome(genes=[1, 0, 1, 1, 0, 0, 1, 0, 0, 1])
        self.assertEqual(problem.fitness(chromosome.genes), 5)

    def test_selection(self):
        # Create some chromosomes with known fitness values
        c1 = Chromosome([0, 0, 0])
        c1.fitness = 10

        c2 = Chromosome([1, 1, 1])
        c2.fitness = 30

        c3 = Chromosome([1, 0, 1])
        c3.fitness = 20

        # Create a mock population class to pass into the select function
        class MockPopulation:
            def __init__(self):
                self.chromosomes = [c1, c2, c3]

        pop = MockPopulation()
        strategy = TournamentSelection(k=3)

        # If we take k=3 (draw all 3 for competition), the individual with fitness 30 must win
        selected = strategy.select(pop)

        self.assertEqual(selected.fitness, 30)
        self.assertEqual(selected.genes, [1, 1, 1])

    def test_crossover(self):
        # Force crossover probability to 1 to ensure it happens
        p1 = Chromosome([0] * 10)
        p2 = Chromosome([1] * 10)
        strategy = OnePointCrossover(prob=1.0)

        c1, c2 = strategy.crossover(p1, p2)

        self.assertEqual(len(c1.genes), 10)
        self.assertEqual(len(c2.genes), 10)
        # Children should be a mix of 0s and 1s
        self.assertIn(1, c1.genes)
        self.assertIn(0, c1.genes)

    def test_mutation(self):
        # Force mutation probability to 1 to ensure it happens
        c = Chromosome([0] * 10)
        strategy = BitFlipMutation(prob=1.0)
        strategy.mutate(c)
        self.assertEqual(c.genes, [1] * 10)  # All bits should be flipped to 1

    def test_improve_over_generations(self):
        # Test that fitness improves over generations
        problem = OneMaxProblem(length=20)
        ga = GeneticAlgorithm(
            problem=problem,
            pop_size=10,
            num_generations=5,
            selection_strat=TournamentSelection(k=2),
            crossover_strat=OnePointCrossover(prob=0.9),
            mutation_strat=BitFlipMutation(prob=0.05),
            elitism_count=1,
        )
        best_solution, history = ga.run()

        self.assertEqual(len(history), 5)
        # New fitness should be greater than or equal to the initial fitness
        self.assertGreaterEqual(history[-1], history[0])


if __name__ == "__main__":
    unittest.main()
