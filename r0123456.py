import Reporter
import numpy as np

class r0123456:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	def optimize(self, filename):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

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

class Individual:
    @staticmethod
    def get_random_instance(info, mutation_chance = 0.05):
        if   isinstance(info, int):
            return Individual(np.random.permutation(np.arange(1, info + 1)), mutation_chance)
        else:
            return Individual(np.random.permutation(np.arange(1, info.no_vertices + 1)), mutation_chance)

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