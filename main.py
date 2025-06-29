import random
import argparse
import numpy as np
from run_trained_ai import Game
from config import *
import os


class Individual:
    """
    Individual organism in the genetic algorithm population.

    Represents a single candidate solution containing neural network weights
    and associated performance metrics. Each individual can be evaluated
    by playing the snake game and computing fitness based on performance.

    Attributes:
        genes (np.array): Neural network weight parameters that define behavior
        score (int): Number of food items collected during game evaluation
        steps (int): Total number of moves made during game evaluation
        fitness (float): Computed fitness value for selection purposes
        seed (int): Random seed of the game used for evaluation (for reproduction)
    """

    def __init__(self, genes):
        """
        Initialize a new individual with given genetic material.

        Args:
            genes (np.array): Weight parameters for neural network
        """
        self.genes = genes
        self.score = 0
        self.steps = 0
        self.fitness = 0
        self.seed = None

    def get_fitness(self):
        """
        Evaluate individual's performance by playing snake game.

        Creates a game instance with this individual's neural network weights,
        runs a complete game session, and computes fitness based on score
        and efficiency (steps taken). Fitness formula balances high scores
        with efficient movement patterns.
        """
        # Create game environment with this individual's genes
        game = Game([self.genes])

        # Run complete game and collect performance metrics
        self.score, self.steps, self.seed = game.play()

        # Compute fitness: reward high scores and penalize inefficient movement
        # Formula: (score + movement_efficiency) * scaling_factor
        # Higher score = better, fewer steps per score = better efficiency
        self.fitness = (self.score + 1 / self.steps) * 100000


class GA:
    """
    Genetic Algorithm implementation for evolving neural network weights.

    Manages population of neural network individuals, handles selection,
    crossover, mutation, and evolution over multiple generations to optimize
    snake-playing behavior through evolutionary computation.

    Attributes:
        p_size (int): Size of parent population maintained each generation
        c_size (int): Number of children generated each generation
        genes_len (int): Length of genetic material (neural network weights)
        mutate_rate (float): Probability of mutation for each gene
        population (list): Current population of Individual instances
        best_individual (Individual): Individual with highest fitness in population
        avg_score (float): Average score across current population
    """

    def __init__(self, p_size=P_SIZE, c_size=C_SIZE, genes_len=GENES_LEN, mutate_rate=MUTATE_RATE):
        """
        Initialize genetic algorithm with specified parameters.

        Args:
            p_size (int): Parent population size (survivors each generation)
            c_size (int): Child population size (offspring generated each generation)
            genes_len (int): Length of genetic material per individual
            mutate_rate (float): Mutation probability per gene (0.0 to 1.0)
        """
        self.p_size = p_size
        self.c_size = c_size
        self.genes_len = genes_len
        self.mutate_rate = mutate_rate
        self.population = []
        self.best_individual = None
        self.avg_score = 0

    def generate_ancestor(self):
        """
        Create initial population with random genetic material.

        Generates p_size individuals with randomly initialized neural network
        weights uniformly distributed between -1 and 1. This creates genetic
        diversity for the evolutionary process to work with.
        """
        for i in range(self.p_size):
            # Generate random weights uniformly distributed in [-1, 1]
            genes = np.random.uniform(-1, 1, self.genes_len)
            self.population.append(Individual(genes))

    def inherit_ancestor(self):
        """
        Load existing population from saved genetic material files.

        Reads neural network weights from './genes/all/{i}' files where i
        represents the i-th individual in the saved population. Useful for
        continuing evolution from a previously saved generation.
        """
        for i in range(self.p_size):
            # Load genetic material from file
            pth = os.path.join("genes", "all", str(i))
            with open(pth, "r") as f:
                genes = np.array(list(map(float, f.read().split())))
                self.population.append(Individual(genes))

    def crossover(self, c1_genes, c2_genes):
        """
        Perform single-point crossover between two parent individuals.

        Genetic crossover operation that combines genetic material from two
        parents to produce offspring. Uses single-point crossover where
        genetic material is swapped at a randomly chosen point.

        Args:
            c1_genes (np.array): Genetic material of first parent (modified in-place)
            c2_genes (np.array): Genetic material of second parent (modified in-place)
        """
        # Choose random crossover point in genetic material
        point = np.random.randint(0, self.genes_len)

        # Swap genetic material from start to crossover point
        c1_genes[:point + 1], c2_genes[:point + 1] = c2_genes[:point + 1], c1_genes[:point + 1]

    def mutate(self, c_genes):
        """
        Apply Gaussian mutation to genetic material.

        Introduces random variations to genetic material by adding Gaussian
        noise to selected genes. Mutation rate controls probability of each
        gene being mutated, while Gaussian distribution provides realistic
        small-scale variations.

        Args:
            c_genes (np.array): Genetic material to mutate (modified in-place)
        """
        # Determine which genes will be mutated based on mutation rate
        mutation_array = np.random.random(c_genes.shape) < self.mutate_rate

        # Generate Gaussian noise for mutations
        mutation = np.random.normal(size=c_genes.shape)

        # Scale mutation magnitude and apply only to selected genes
        mutation[mutation_array] *= 0.2
        c_genes[mutation_array] += mutation[mutation_array]

    def elitism_selection(self, size):
        """
        Select top-performing individuals using elitist selection strategy.

        Sorts population by fitness and selects the specified number of
        best-performing individuals. Ensures that high-quality genetic
        material is preserved across generations.

        Args:
            size (int): Number of individuals to select

        Returns:
            list: Top 'size' individuals sorted by fitness (highest first)
        """
        # Sort population by fitness in descending order
        population = sorted(self.population, key=lambda individual: individual.fitness, reverse=True)
        return population[:size]

    def roulette_wheel_selection(self, size):
        """
        Select individuals using fitness-proportionate roulette wheel selection.

        Probabilistic selection method where individuals with higher fitness
        have proportionally higher chances of being selected. Simulates
        spinning a roulette wheel where each individual occupies a slice
        proportional to their fitness.

        Args:
            size (int): Number of individuals to select

        Returns:
            list: Selected individuals (may contain duplicates)
        """
        selection = []

        # Calculate total fitness of population (size of roulette wheel)
        wheel = sum(individual.fitness for individual in self.population)

        # Perform selection 'size' times
        for _ in range(size):
            # Generate random selection point on wheel
            pick = np.random.uniform(0, wheel)
            current = 0

            # Find individual corresponding to selection point
            for individual in self.population:
                current += individual.fitness
                if current > pick:
                    selection.append(individual)
                    break

        return selection

    def evolve(self):
        """
        Execute one complete generation of genetic algorithm evolution.

        Core evolutionary process that evaluates all individuals, selects
        parents, generates offspring through crossover and mutation, and
        updates population for next generation. Tracks best individual
        and population statistics.
        """
        # Evaluate fitness for entire population
        sum_score = 0
        for individual in self.population:
            individual.get_fitness()
            sum_score += individual.score

        # Calculate population statistics
        self.avg_score = sum_score / len(self.population)

        # Select best individuals as parents for next generation
        self.population = self.elitism_selection(self.p_size)
        self.best_individual = self.population[0]  # Best individual is first after sorting

        # Randomize parent order to avoid bias in mating
        random.shuffle(self.population)

        # Generate offspring through crossover and mutation
        children = []
        while len(children) < self.c_size:
            # Select two parents using roulette wheel selection
            p1, p2 = self.roulette_wheel_selection(2)

            # Create copies of parent genetic material for offspring
            c1_genes, c2_genes = p1.genes.copy(), p2.genes.copy()

            # Apply crossover to combine parent genes
            self.crossover(c1_genes, c2_genes)

            # Apply mutation to introduce variation
            self.mutate(c1_genes)
            self.mutate(c2_genes)

            # Create new individuals from modified genetic material
            c1, c2 = Individual(c1_genes), Individual(c2_genes)
            children.extend([c1, c2])

        # Add offspring to population for evaluation in next generation
        random.shuffle(children)
        self.population.extend(children)

    def save_best(self):
        """
        Save the best individual's genetic material and game seed to files.

        Preserves the neural network weights and random seed of the best
        performing individual for later replay and analysis. Files are
        organized by the score achieved.
        """
        score = self.best_individual.score

        # Save neural network weights
        genes_pth = os.path.join("genes", "best", str(score))
        with open(genes_pth, "w") as f:
            for gene in self.best_individual.genes:
                f.write(str(gene) + " ")

                # Save random seed for game reproduction
        seed_pth = os.path.join("seed", str(score))
        with open(seed_pth, "w") as f:
            f.write(str(self.best_individual.seed))

    def save_all(self):
        """
        Save entire current population to files for later continuation.

        Evaluates and sorts population, then saves the top p_size individuals'
        genetic material to files. Enables pausing and resuming evolution
        process across multiple program runs.
        """
        # Ensure all individuals have been evaluated
        for individual in self.population:
            individual.get_fitness()

        # Select top individuals to save
        population = self.elitism_selection(self.p_size)

        # Save genetic material for each individual
        for i in range(len(population)):
            pth = os.path.join("genes", "all", str(i))
            with open(pth, "w") as f:
                for gene in self.population[i].genes:
                    f.write(str(gene) + " ")


if __name__ == '__main__':
    # Command line argument parsing for program configuration
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--inherit', action="store_true",
                        help="Whether to load genes from path ./genes/all.")
    parser.add_argument('-s', '--show', action="store_true",
                        help='whether to show the best individual to play snake after each evolve.')
    args = parser.parse_args()

    # Initialize genetic algorithm instance
    ga = GA()

    # Create initial population (either random or loaded from files)
    if args.inherit:
        # Continue evolution from saved population
        ga.inherit_ancestor()
    else:
        # Start fresh evolution with random population
        ga.generate_ancestor()

    # Main evolution loop - runs indefinitely until manually stopped
    generation = 0
    record = 0  # Track best score ever achieved

    while True:
        generation += 1

        # Execute one generation of evolution
        ga.evolve()

        # Display progress information
        print("generation:", generation, ", record:", record,
              ", best score:", ga.best_individual.score, ", average score:", ga.avg_score)

        # Save new record-breaking individuals
        if ga.best_individual.score >= record:
            record = ga.best_individual.score
            ga.save_best()

        # Optionally display best individual playing game
        if args.show:
            genes = ga.best_individual.genes
            seed = ga.best_individual.seed
            game = Game(show=True, genes_list=[genes], seed=seed)
            game.play()

        # Periodically save entire population for backup/continuation
        if generation % 20 == 0:
            ga.save_all()
