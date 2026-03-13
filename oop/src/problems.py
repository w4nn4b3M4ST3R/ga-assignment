import random


class OneMaxProblem:
    def __init__(self, length=100):
        self.length = length

    def fitness(self, genes):
        return sum(genes)


class KnapsackProblem:
    def __init__(self, num_items=100, random_seed=42):
        random.seed(random_seed)
        self.num_items = num_items

        # initialize items with random weights and values
        self.items = [
            {"weight": random.uniform(1, 50), "value": random.uniform(10, 100)}
            for _ in range(num_items)
        ]

        # set capacity to 40% of total weight of all items
        total_weight = sum(item["weight"] for item in self.items)
        self.capacity = total_weight * 0.4

    def fitness(self, genes):
        total_value = 0
        total_weight = 0
        for i, bit in enumerate(genes):
            if bit == 1:
                total_value += self.items[i]["value"]
                total_weight += self.items[i]["weight"]

        if total_weight > self.capacity:
            return 0  # invalid solution, exceeds capacity
        return total_value


class HyperparameterTuningProblem:
    """
    Bonus: Extensible Design - Hyperparameter Tuning.
    Uses GA to optimize Learning Rate (alpha) and L2 Regularization (lambda) for a ML model.
    """

    def __init__(self):
        # 16-bit chromosome: 8 bits for alpha, 8 bits for lambda
        self.length = 16

    def decode(self, genes):
        """Decodes binary genes into continuous float values."""
        alpha_bits = genes[:8]
        lambda_bits = genes[8:]

        # Convert binary arrays to integers (0 to 255)
        alpha_int = sum(bit * (2**i) for i, bit in enumerate(reversed(alpha_bits)))
        lambda_int = sum(bit * (2**i) for i, bit in enumerate(reversed(lambda_bits)))

        # Map integers to actual hyperparameter ranges
        # alpha range: [0.0001, 0.1]
        alpha = 0.0001 + (alpha_int / 255.0) * (0.1 - 0.0001)

        # lambda range: [0.0, 1.0]
        l2_lambda = (lambda_int / 255.0) * 1.0

        return alpha, l2_lambda

    def fitness(self, genes):
        alpha, l2_lambda = self.decode(genes)

        # Simulated optimal hyperparameters for testing
        optimal_alpha = 0.015
        optimal_lambda = 0.2

        # Calculate pseudo-loss (simulating a model's validation loss)
        loss = ((alpha - optimal_alpha) ** 2) * 1000 + (
            (l2_lambda - optimal_lambda) ** 2
        ) * 10

        # Maximize fitness by returning the inverse of the loss
        # Add 1e-6 to prevent division by zero errors
        return 1.0 / (loss + 1e-6)
