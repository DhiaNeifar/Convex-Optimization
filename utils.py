import time


def ColorText(color, text):
    if color == 'r':
        return f'\033[31m{text}\033[0m'
    if color == 'g':
        return f'\033[32m{text}\033[0m'
    if color == 'y':
        return f'\033[33m{text}\033[0m'
    if color == 'b':
        return f'\033[34m{text}\033[0m'


def InitDisplay():
    InitMessage = """
            ##################################################################################################
            ##                                                                                              ##
            ##                     ####################################################                     ##
            ##                     # DHIA NEIFAR IS THE SOLE PROPRIETARY OF THIS CODE #                     ##
            ##                     ####################################################                     ##
            ##                                                                                              ##
            ##                                                                                              ##
            ##                     ####################################################                     ##
            ##                     # THIS CODE HAS BEEN DEVELOPED FOR CIS 505 PROJECT #                     ##
            ##                     ####################################################                     ##
            ##                                                                                              ##
            ##################################################################################################"""
    print(ColorText("g", InitMessage))


def InitFunction(Function):
    InitFunctionMessage = f"""
            -------------------------------------------------------------------------------------------------

                                        Function f: (x, y, z) --> {Function}
    """
    print(ColorText("y", InitFunctionMessage))


def DisplayOptimizer(Optimizer):
    OptimizerMessage = f"""  
                                                    {Optimizer}
    """
    print(ColorText("r", OptimizerMessage))


def DisplayResults(OptimalVector, OptimalResult, NumberIterations, GDAlgEndTime, GDAlgStartTime):
    ResultsMessages = f"""            
                        Optimal Vector:
                        x --> {OptimalVector[0]} 
                        y --> {OptimalVector[1]}
                        z --> {OptimalVector[2]}
                        
                        Optimal Result --> {OptimalResult}
                        
                        Number of Iterations --> {NumberIterations}
                        
                        Convergence Speed in Seconds --> {GDAlgEndTime - GDAlgStartTime}
                        
    """
    print(ColorText("b", ResultsMessages))


def DisplayResult(function, Optimizer, OptimizerName, parameters):
    DisplayOptimizer(OptimizerName)
    GDAlgStartTime = time.time()
    OptimalVector, OptimalResult, NumberIterations = Optimizer(function.function, **parameters)
    GDAlgEndTime = time.time()
    DisplayResults(OptimalVector, OptimalResult, NumberIterations, GDAlgEndTime, GDAlgStartTime)