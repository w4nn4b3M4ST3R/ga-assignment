import random
from abc import ABC, abstractmethod


# --- Main genetic algorithm's components ---
class Chromosome:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = (
            0  # Fitness will be calculated based on the problem's evaluation function
        )

    def calculate_fitness(self, problem):
        self.fitness = problem.fitness(self.genes)


class Population:
    def __init__(self, size, chromosome_length):
        self.chromosomes = [
            Chromosome([random.randint(0, 1) for _ in range(chromosome_length)])
            for _ in range(size)
        ]  # Initialize population with random chromosomes

    def evaluate(self, problem):
        for chromo in self.chromosomes:
            chromo.calculate_fitness(problem)

    def get_best(self):
        return max(self.chromosomes, key=lambda c: c.fitness)


class GeneticAlgorithm:
    def __init__(
        self,
        problem,
        pop_size,
        num_generations,
        selection_strat,
        crossover_strat,
        mutation_strat,
        elitism_count=2,
    ):
        self.problem = problem
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.selection_strat = selection_strat
        self.crossover_strat = crossover_strat
        self.mutation_strat = mutation_strat
        self.elitism_count = elitism_count

    def run(self):
        population = Population(
            self.pop_size,
            self.problem.length
            if hasattr(self.problem, "length")
            else self.problem.num_items,
        )  # Initialize population based on problem's requirements
        population.evaluate(self.problem)

        history = []
        # Initialize population
        for gen in range(self.num_generations):
            population.chromosomes.sort(key=lambda c: c.fitness, reverse=True)
            next_gen = population.chromosomes[
                : self.elitism_count
            ]  # Elitism: carry forward the best individuals

            while len(next_gen) < self.pop_size:
                p1 = self.selection_strat.select(population)
                p2 = self.selection_strat.select(population)

                c1, c2 = self.crossover_strat.crossover(p1, p2)

                self.mutation_strat.mutate(c1)
                self.mutation_strat.mutate(c2)

                next_gen.extend([c1, c2])

            # Cut off if exceeds population size
            population.chromosomes = next_gen[: self.pop_size]
            population.evaluate(self.problem)

            best_fitness = population.get_best().fitness
            history.append(best_fitness)

        return population.get_best(), history


# --- Strategies ---
class SelectionStrategy(ABC):
    @abstractmethod
    def select(self, population):
        pass


class CrossoverStrategy(ABC):
    @abstractmethod
    def crossover(self, parent1, parent2):
        pass


class MutationStrategy(ABC):
    @abstractmethod
    def mutate(self, individual):
        pass


# --- Operators implementations ---
class TournamentSelection(SelectionStrategy):
    def __init__(self, k=3):
        self.k = k  # Number of individuals to compete in the tournament

    def select(self, population):
        # Randomly select k individuals and return the one with the highest fitness
        tournament = random.sample(population.chromosomes, self.k)
        return max(tournament, key=lambda chromo: chromo.fitness)


class OnePointCrossover(CrossoverStrategy):
    def __init__(self, prob=0.9):
        self.prob = prob  # Crossover probability

    def crossover(self, parent1, parent2):
        if random.random() < self.prob:
            point = random.randint(1, len(parent1.genes) - 1)
            child1_genes = parent1.genes[:point] + parent2.genes[point:]
            child2_genes = parent2.genes[:point] + parent1.genes[point:]
            return Chromosome(child1_genes), Chromosome(child2_genes)
        else:
            return Chromosome(parent1.genes.copy()), Chromosome(parent2.genes.copy())


class BitFlipMutation(MutationStrategy):
    def __init__(self, prob):
        self.prob = prob  # Mutation probability

    def mutate(self, chromosome):
        # Flip each bit with a pre-defined probability
        for i in range(len(chromosome.genes)):
            if random.random() < self.prob:
                chromosome.genes[i] = 1 - chromosome.genes[i]  # Flip bit
