import numpy as np
import random
from tqdm import tqdm


def GeneticAlgorithm(func, LB, UB, dimension, pop_size=100, generations=1000, mutation_rate=0.01):
    """
    Genetic Algorithm to minimize a given function.

    Parameters:
        func (callable): The function to minimize. Must take a 2D numpy array of shape (n, 2) as input.
        LB (Lower Bound): The lower bound for each variable x_i.
        UB (Upper Bound): The upper bound for each variable x_i
        dimension: The dimension of the input x.
        pop_size (int): Size of the population.
        generations (int): Number of generations to evolve.
        mutation_rate (float): Probability of mutation.

    Returns:
        tuple: Best solution and its function value.
    """
    population = np.random.uniform(low=LB, high=UB, size=(pop_size, dimension))

    def EvaluatePopulation(pop):
        """Evaluate the fitness of the population."""
        return np.array([func(*individual) for individual in pop])

    def SelectParents(pop, Fitness, TournamentSize=10):
        """Select parents using tournament selection."""
        Parents = []
        for _ in range(len(pop)):  # len(pop) // 2
            Chosen = np.random.choice(len(pop), size=TournamentSize, replace=False)
            BestIndex = Chosen[0]
            for index in Chosen:
                if Fitness[index] < Fitness[BestIndex]:
                    BestIndex = index
            Parents.append(pop[BestIndex])
        return np.array(Parents)


    def crossover(Parent1, Parent2):
        """Perform crossover between two parents."""
        crossover_point = dimension // 2
        child = np.concatenate((Parent1[:crossover_point], Parent2[crossover_point:]))
        return child

    def mutate(child):
        """Mutate a child with given mutation rate."""
        for index in range(dimension):
            if random.random() < mutation_rate:
                child[index] = random.uniform(LB, UB)
        return child

    BestFitness = None
    BestSolution = None
    for _ in tqdm(range(generations)):
        fitness = EvaluatePopulation(population)
        parents = SelectParents(population, fitness)
        BestFitness = np.min(fitness)
        BestSolution = population[np.argmin(fitness)]
        NextGeneration = []
        for i in range(0, pop_size, 2):
            parent1, parent2 = parents[i], parents[min(i + 1, pop_size - 1)]
            child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            NextGeneration.append(mutate(child1))
            NextGeneration.append(mutate(child2))
        population = np.array(NextGeneration[:pop_size])

    return BestSolution, BestFitness, generations
