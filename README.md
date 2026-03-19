# **[Extended Assignment]** Genetic Algorithm (GA) — Object-Oriented vs Functional Programming

**Instructor:** Nguyen Thanh Cong, PhD

**Student Name:** Lai Tran Tri

**Student ID:** 2413631

## 1. Project Overview

This project is an extended major assignment exploring the implementation of a Genetic Algorithm (GA) through two distinct software engineering paradigms: **Object-Oriented Programming (OOP)** and **Functional Programming (FP)**.

The primary objective is to evaluate the trade-offs between mutability and immutability, statefulness and pure functions, and execution speed versus code safety. To demonstrate the robustness and extensibility of the architecture, the GA is applied to three distinct optimization problems, ranging from classical computer science puzzles to practical machine learning applications.

## 2. Implemented Optimization Problems

1. **OneMax Problem:** The baseline test. A pure discrete optimization task aiming to maximize the number of `1`s in a binary chromosome of length `L=100`.
2. **0/1 Knapsack Problem:** A constrained combinatorial optimization problem. The GA must maximize the total value of items without exceeding a strict weight capacity (`n=100`).
3. **Hyperparameter Tuning (Bonus - Extensible Design):** Applying GA to the Machine Learning domain. The algorithm decodes a 16-bit chromosome into continuous floating-point values to optimize the Learning Rate (`alpha` in [0.0001, 0.1]) and L2 Regularization (`lambda` in [0.0, 1.0]) for a predictive model, minimizing the simulated validation loss.

## 3. Installation & Execution

### Prerequisites

- Python 3.8+
- `matplotlib` (for generating evolution curves)

```bash
pip install matplotlib
```

### Running the Experiments

Both paradigms are centrally controlled by `config.py` to ensure identical hyperparameter constraints (Population: 100, Generations: 300, Mutation Rate: 1/L).

To execute the Object-Oriented pipeline:

```bash
python oop/run.py
```

To execute the Functional Programming pipeline:

```bash
python fp/run.py
```

_Note: Execution will automatically generate fitness evolution plots (`.png`) and detailed performance logs (`.json`) in the `reports/` directory._

### Running Unit Tests

The project features strict test coverage for selection, crossover, mutation, and generational improvement.

```bash
python -m unittest oop/tests/test_ga.py
python -m unittest fp/tests/test_ga.py
```

## 4. Reflection: OOP vs. FP Trade-offs

Transitioning the GA engine between OOP and FP paradigms revealed significant architectural and computational trade-offs:

- **Object-Oriented Programming (OOP):** The OOP implementation utilizes the Strategy Pattern to decouple genetic operators (Selection, Crossover, Mutation) from the core engine. Modeling biological processes natively aligns with OOP; a `Chromosome` object "mutates" by altering its internal state in place. Because Python lists are mutable, in-place bit-flipping is highly memory-efficient. This resulted in superior raw execution speed, solving the Knapsack problem in approximately `~0.26s` and the complex Tuning problem in `~0.12s`.
- **Functional Programming (FP):** The FP pipeline completely discards classes and mutable states, relying strictly on pure functions (`map`, `reduce`, `filter`) and immutable `tuples`. While this eliminates side-effects and race conditions—making the codebase theoretically perfect for parallel or distributed computing—it introduces a heavy performance penalty. Python's garbage collection struggles with continuous memory allocation for new tuples in every generation, causing the FP execution time to be roughly 2x to 3x slower than OOP (e.g., `~0.53s` for Knapsack).

Ultimately, OOP proved superior for the raw iterative speed required by heuristic searches, while FP enforced a safer, side-effect-free data transformation pipeline.

## 5. Extra Analysis: Genetic Algorithms vs. Gradient-Based Optimization

By successfully extending the GA to solve the **Hyperparameter Tuning** problem, this project highlights specific scenarios where derivative-free optimization (like GA) outperforms standard gradient-based methods (like Gradient Descent or Adam):

1. **Handling Discrete & Non-Differentiable Spaces:** Gradient Descent mathematically requires continuous, differentiable loss landscapes to compute weight updates. It entirely fails on discrete problems like the 0/1 Knapsack. GA, using binary representations, seamlessly bridges the gap between discrete logic and continuous values via binary decoding.
2. **Escaping Local Optima:** In highly non-convex loss landscapes (common in complex hyperparameter tuning), gradient methods can easily get trapped in local minima. GA mitigates this through stochastic **Mutation** and **Crossover**, allowing the algorithm to globally explore the search space and jump out of suboptimal valleys.
3. **Population-Based Global Search:** Unlike gradient methods, which move a single solution iteratively down a slope, GA maintains a parallel population of 100 candidate solutions. This explores multiple regions of the solution space simultaneously before converging on the global optimum.
