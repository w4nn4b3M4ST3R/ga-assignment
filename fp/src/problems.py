import random
from functools import reduce


def get_knapsack_items(num_items=100, seed=42):
    # Initialize random seed for reproducibility
    random.seed(seed)

    items = tuple(
        {"weight": random.uniform(1, 50), "value": random.uniform(10, 100)}
        for _ in range(num_items)
    )
    capacity = 0.4 * sum(
        item["weight"] for item in items
    )  # Set capacity to 40% of total weight
    return items, capacity


def fitness_onemax(chromosome):
    # Calculate fitness as the number of 1s in the chromosome
    return reduce(lambda acc, val: acc + val, chromosome, 0)


def fitness_knapsack(chromosome, items, capacity):

    selected_items = tuple(
        map(lambda x: x[1], filter(lambda x: x[0] == 1, zip(chromosome, items)))
    )

    total_weight = reduce(lambda acc, item: acc + item["weight"], selected_items, 0)
    total_value = reduce(lambda acc, item: acc + item["value"], selected_items, 0)

    if total_weight > capacity:
        return 0  # Invalid solution, exceeds capacity

    return total_value


def fitness_tuning(genes):
    """
    Bonus: Extensible Design - Hyperparameter Tuning for FP.
    Evaluates a 16-bit tuple representing learning rate (alpha) and regularization (lambda).
    """
    # Slice the tuple
    alpha_bits = genes[:8]
    lambda_bits = genes[8:]

    # Convert binary tuples to integers
    alpha_int = sum(bit * (2**i) for i, bit in enumerate(reversed(alpha_bits)))
    lambda_int = sum(bit * (2**i) for i, bit in enumerate(reversed(lambda_bits)))

    # Map integers to hyperparameter ranges: alpha in [0.0001, 0.1], lambda in [0.0, 1.0]
    alpha = 0.0001 + (alpha_int / 255.0) * (0.1 - 0.0001)
    l2_lambda = (lambda_int / 255.0) * 1.0

    # Simulated optimal hyperparameters
    optimal_alpha = 0.015
    optimal_lambda = 0.2

    # Calculate pseudo-loss
    loss = ((alpha - optimal_alpha) ** 2) * 1000 + (
        (l2_lambda - optimal_lambda) ** 2
    ) * 10

    # Return fitness (inverse of loss)
    return 1.0 / (loss + 1e-6)
