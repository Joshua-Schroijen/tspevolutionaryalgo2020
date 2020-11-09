import Reporter
import numpy as np

class r0123456:
    def __init__(self, k = 5, no_individuals_to_keep):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

        self._k = k
        self._no_individuals_to_keep = no_individuals_to_keep
        
        self._tsp = None

    def optimize(self, filename):		
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()
        
        self._tsp = TSP(distanceMatrix)

        # Your code here.

        while( yourConvergenceTestsHere ):

            # Your code here.

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution 
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

        # Your code here.
        return 0

    def _initialize(self):
        self._population = Population.initialize(self.knapsack_problem, self.population_size)

    def _selection(self):
        selected = list(np.random.choice(self._population.individuals, self._k))
        fitnesses = [self._tsp.fitness(individual) for individual in selected]
        i = fitnesses.index(max(fitnesses))
        return selected[i]

    def _recombination(self, first_parent, second_parent):
        pass
        
    def _mutation(self, individual):
        pass
            
    def _elimination(self, offspring):
        combined = []
        combined.extend(self._population.individuals)
        combined.extend(offspring)
        combined = np.array(combined)
        
        selected = np.flip(np.argsort(np.array(list(map(lambda individual: self._tsp.fitness(individual), combined)))))[0:self._no_individuals_to_keep]
        return list(combined[selected])

class TSP:
    def __init__(self, distance_matrix):
        self._distance_matrix = distance_matrix

    def fitness(self, individual):
        total_distance = 0
        for a, b in zip(individual.permutation[0:(self.no_vertices - 1)], individual.permutation[1:self.no_vertices]):
            total_distance += self._distance_matrix[a, b]

        total_distance += self._distance_matrix[b, individual.permutation[0]]

        return total_distance

    @property
    def distance_matrix(self):
        return self._distance_matrix

    @distance_matrix.setter
    def distance_matrix(self, distance_matrix):
        self._distance_matrix = distance_matrix

    @property
    def no_vertices(self):
        return self._distance_matrix.shape[0]

class Individual:
    @staticmethod
    def get_random_instance(info, mutation_chance = 0.05):
        if   isinstance(info, int):
            return Individual(np.random.permutation(np.arange(info)), mutation_chance)
        else:
            return Individual(np.random.permutation(np.arange(info.no_vertices)), mutation_chance)

    @staticmethod
    def get_random_instances(no_instances, info, mutation_chance = 0.05):
        return [Individual.get_random_instance(info, mutation_chance) for _ in range(no_instances)]

    def __init__(self, permutation, mutation_chance = 0.05):
        self._permutation = order
        self._mutation_chance = mutation_chance

    @property
    def permutation(self):
        return self._permutation

    @order.setter
    def permutation(self, permutation):
        self._permutation = permutation

    @property
    def mutation_chance(self):
        return self._mutation_chance

    @mutation_chance.setter
    def mutation_chance(self, mutation_chance):
        self._mutation_chance = mutation_chance
        
class Population:
    @staticmethod
    def initialize(no_individuals, tsp, mutation_chance = 0.05):
        return Population(Individual.get_random_instances(no_individuals, tsp, mutation_chance))

    def __init__(self, individuals):
        self._individuals = individuals

    def __iter__(self):
        return iter(self._individuals)

    @property
    def individuals(self):
        return self._individuals

    @individuals.setter
    def individuals(self, individuals):
        self._individuals = individuals