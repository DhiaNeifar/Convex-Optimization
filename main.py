import numpy as np
import time

from utils import InitDisplay, InitFunction, DisplayResult
from Functions import QuadraticFunction
from GradientDescent import GradientDescent
from GeneticAlgorithm import GeneticAlgorithm



def Optimization():

    InitDisplay()

    DisplayFunctions = ["x² + y² + z²", "(x - 5)² + (y + 7)² + (z - 100)²", "x² + y² + z² - 98"]
    Functions = [lambda x, y, z: (x ** 2 - 5) + (y ** 2 + 7) + (z ** 2 - 100)]

    dimension = 3
    LB, UB = -5, 5

    for index, function in enumerate(Functions):
        InitFunction(DisplayFunctions[index])
        quad_func = QuadraticFunction(function)
        GDParameters = {
            "UB": UB,
            "LB": LB,
            "dimension": dimension,
            "learning_rate": 0.1,
            "tolerance": 1e-6,
            "max_iterations": 1000,
        }
        DisplayResult(quad_func, GradientDescent, "Gradient Descent", GDParameters)
        GAParameters = {
            "UB": UB,
            "LB": LB,
            "dimension": dimension,
            "pop_size": 100,
            "generations": 1000,
            "mutation_rate": 0.01,
        }
        DisplayResult(quad_func, GeneticAlgorithm, "Genetic Algorithm", GAParameters)


if __name__ == '__main__':
    Optimization()
