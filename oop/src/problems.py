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
