import random
from functools import reduce


def init_population(pop_size, length):
    return tuple(
        tuple(random.randint(0, 1) for _ in range(length)) for _ in range(pop_size)
    )


def evaluate_population(population, fitness_func):
    # Use map to apply the fitness function to each chromosome in the population
    return tuple(map(fitness_func, population))


def select_tournament(population, fitnesses, k=3):
    indices = random.sample(range(len(population)), k)
    # Find the chromosome with the highest fitness
    best_idx = reduce(
        lambda best, current: current if fitnesses[current] > fitnesses[best] else best,
        indices,
    )
    return population[best_idx]


def crossover(p1, p2, prob=0.9):
    if random.random() < prob:
        point = random.randint(1, len(p1) - 1)
        # Concatenate the segments of the parents to create two new offspring
        return (p1[:point] + p2[point:], p2[:point] + p1[point:])
    else:
        return (p1, p2)


def mutate(chromosome, prob):
    # Use map to apply mutation to each bit in the chromosome
    return tuple(
        map(lambda bit: 1 - bit if random.random() < prob else bit, chromosome)
    )


def evolve_generation(state, _):
    """
    Pure function receives the previous state and returns a new state.
    state = (population, history, config)
    """
    population, history, config = state
    fitness = evaluate_population(population, config["fitness_func"])

    # Sort population to get the elites
    pop_with_fit = sorted(zip(population, fitness), key=lambda x: x[1], reverse=True)
    best_fitness = pop_with_fit[0][1]
    elites = tuple(ind[0] for ind in pop_with_fit[: config["elitism_count"]])

    # Lambda function to create 2 children
    def make_childrens(_):
        p1 = select_tournament(population, fitness, config["tournament_k"])
        p2 = select_tournament(population, fitness, config["tournament_k"])
        c1, c2 = crossover(p1, p2, config["crossover_prob"])
        return mutate(c1, config["mutation_prob"]), mutate(c2, config["mutation_prob"])

    num_children = config["pop_size"] - config["elitism_count"]
    num_pairs = (
        num_children + 1
    ) // 2  # Number of pairs needed to create the required number of children

    # Use map to create children pairs and then flatten the list of pairs into a single tuple of children
    children_pairs = tuple(map(make_childrens, range(num_pairs)))
    children = reduce(lambda acc, pair: acc + pair, children_pairs, ())

    # New state with the next generation population
    next_population = elites + children[:num_children]
    next_history = history + (best_fitness,)

    return (next_population, next_history, config)
