import math
import random
import statistics
import collections
import itertools
import logging
import timeit
import cProfile, pstats
import sys
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import Reporter
from enum import Enum

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(f"r0486848.log", mode="w")
stream_handler = logging.StreamHandler()
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
    
def profile(filename):
    def _profile(function):
        def wrapped(*args, **kwargs):
            profile = cProfile.Profile()
            profile.enable()
            retval = function(*args, **kwargs)
            profile.disable()
            output_file = open(filename, "w", encoding="utf-8")
            stats = pstats.Stats(profile, stream=output_file).sort_stats("cumulative")
            stats.print_stats()
            output_file.close()

            return retval

        return wrapped

    return _profile

def _get_swapped(permutation, a, b):
    """
    Swaps the elements at indices a and b of the given permutation

    :param permutation: a 1D-Numpy array representing the permutation we want to apply a swap to
    :param a: first index of elements to swap
    :param b: second index of elements to swap
    :return: a new permutation based on the given one with the elements at indices a and b swapped
    """
        
    # Create a copy of the permutation
    permutation_copy = np.copy(permutation)
    # Use Numpy's indexing-with-a-list feature to easily swap the elements in the copied permutation
    permutation_copy[[b, a]] = permutation_copy[[a, b]]
    # Return the copied permutation with the given elements swapped
    return permutation_copy

def _get_pairs_cycle(l):
    return [(l[i], l[(i + 1) % len(l)]) for i in range(len(l))]

class PopulationGenerationScheme(Enum):
    RANDOM                  = 1
    NEAREST_NEIGHBOUR_BASED = 2

class PopulationGenerationSettings:
    def __init__(self, no_random_indvs, no_base_nn_indvs, no_derived_nn_indvs):
        self._no_random_individuals = no_random_indvs
        self._no_base_nearest_neighbour_individuals = no_base_nn_indvs
        self._no_derived_nearest_neighbour_individuals = no_derived_nn_indvs

    @property
    def no_random_individuals(self):
        return self._no_random_individuals
    
    @property
    def no_base_nearest_neighbour_individuals(self):
        return self._no_base_nearest_neighbour_individuals
        
    @property
    def no_derived_nearest_neighbour_individuals(self):
        return self._no_derived_nearest_neighbour_individuals
    
    @property
    def total_population_size(self):
        return self._no_random_individuals + self._no_base_nearest_neighbour_individuals + self._no_derived_nearest_neighbour_individuals

    def __repr__(self):
        return f"PopulationGenerationSettings(no_random_indvs = {self._no_random_individuals}, no_base_nn_indvs = {self._no_base_nearest_neighbour_individuals}, no_derived_nn_indvs = {self._no_derived_nearest_neighbour_individuals})"

    def __str__(self):
        return f"PopulationGenerationSettings(no_random_indvs = {self._no_random_individuals}, no_base_nn_indvs = {self._no_base_nearest_neighbour_individuals}, no_derived_nn_indvs = {self._no_derived_nearest_neighbour_individuals})"

class RecombinationOperator(Enum):
    PMX   = 1
    HGREX = 2

class EliminationScheme(Enum):
    LAMBDAPLUSMU           = 1
    LAMBDAPLUSMU_WCROWDING = 2
    LAMBDA_MU              = 3

class r0486848:
    """
    This class manages the solving of a TSP with r0486848's evolutionary algorithm
    """
    def __init__(self):
        """
        Constructs the r0486848 TSP solver object that will be used for running the evolutionary algorithm
        """
        # Initialize and save Reporter class instance
        self.reporter = Reporter.Reporter(self.__class__.__name__)

        self._optimize = self.optimize
        self._tsp = None

        self._parameters_set = False
        
    def set_parameters(self, population_generation_settings, recombination_operator, elimination_scheme, no_islands, island_swap_rate, island_no_swapped_individuals, default_k, mu, no_individuals_to_keep, default_mutation_chance, mutation_chance_feedback, mutation_chance_self_adaptivity, stopping_ratio, tolerances, provide_analytics):
        """
        Sets the evolutionary algorithm's parameters
        
        :param population_generation_settings: the population generation settings (how many individuals to generate through different schemes)
        :param reombination_operator: the recombination operator to use
        :param elimination_scheme: the elimination scheme
        :param no_islands: the number of islands to use (this evolutionary algorithm uses the island model)
        :param default_k: default tournament size for k-tournament selection
        :param mu: number of offspring to generate from the population
        :param no_individuals_to_keep: number of individuals to keep in elimination steps
        :param default_mutation_chance: number between 0 and 1 representing the default chance of mutation
        :param mutation_chance_feedback: if set to True, mutation chance feedback is enabled
        :param mutation_chance_self_adaptivity: if set to True, mutation chance self-adaptivity is enabled
        :param stopping_ratio: the relative improvement in the current iteration compared to the previous one below which, after tolerances iterations, to stop optimization
        :param tolerances: the number of iterations ran below the stopping ratio before optimization is stopped
        :return: an initialized evolutionary algorithm object of class r0486848
        """
        # Copy given evolutionary algorithm parameters to attributes
        self._population_generation_settings = population_generation_settings
        self._recombination_operator = recombination_operator
        self._elimination_scheme = elimination_scheme
        self._no_islands = no_islands
        self._island_swap_rate = island_swap_rate
        self._island_no_swapped_individuals = island_no_swapped_individuals
        self._default_k = default_k
        self._mu = mu
        self._no_individuals_to_keep = no_individuals_to_keep
        self._default_mutation_chance = default_mutation_chance
        self._mutation_chance_self_adaptivity = mutation_chance_self_adaptivity
        self._mutation_chance_feedback = mutation_chance_feedback
        self._stopping_ratio = stopping_ratio
        self._tolerances = tolerances
    
        self._provide_analytics = provide_analytics
        if self._provide_analytics == True:
            self.optimize = profile("algorithm_profile.txt")(self._optimize)
        else:
            self.optimize = self._optimize

        self._parameters_set = True

    def optimize(self, source, verbose=False):   
        if verbose == True:
            logger.info("Starting evolutionary algorithm ...")
        # Start timer for assessing optimization speed
        start_time = timeit.default_timer()

        if isinstance(source, TSP):
            self._tsp = source
        else:
            # Import TSP instance
            file = open(source)
            distanceMatrix = np.loadtxt(file, delimiter=",")
            file.close()
        
            # Set up TSP instance representation
            self._tsp = TSP(distanceMatrix)
        
        if self._parameters_set == False:
            self.set_parameters(population_generation_settings = PopulationGenerationSettings(0, self._tsp.no_vertices, 3), recombination_operator = RecombinationOperator.HGREX, elimination_scheme = EliminationScheme.LAMBDAPLUSMU, no_islands = 7, island_swap_rate = 3, island_no_swapped_individuals = 8, default_k = int((self._tsp.no_vertices * 3) / (7 * 4)), mu = int((self._tsp.no_vertices * 3) / 7), no_individuals_to_keep = int((self._tsp.no_vertices * 3) / 7), default_mutation_chance = 0.1, mutation_chance_feedback = False, mutation_chance_self_adaptivity = False, stopping_ratio = 0.001, tolerances = 3, provide_analytics = False)
        
        # Report TSP instance heuristic benchmark performance
        nn_mean_fitness, nn_best_fitness = self._get_benchmarks()
        if verbose == True:
            logger.info(f"Benchmarks:\n\tMean heuristic fitness = {nn_mean_fitness:.5f}\n\tBest heuristic fitness = {nn_best_fitness:.5f}")

        # Initialize the population
        complete_initial_population = self._get_initial_population()
        if verbose == True:
            logger.info("Initial population generated")
        
        # Run evolutionary algorith on islands, keeping track of performance
        iteration_number = 1
        iteration_numbers = [(iteration_number - 1)]
        current_mean_fitness = self._tsp.mean_fitness(complete_initial_population.individuals, True)
        current_best_fitness = self._tsp.best_fitness(complete_initial_population.individuals)
        mean_fitnesses = [current_mean_fitness]
        best_fitnesses = [current_best_fitness]
        
        self.reporter.report(current_mean_fitness, current_best_fitness, self._tsp.best_individual(complete_initial_population.individuals).permutation)
        if self._provide_analytics == True: mean_distances_to_others = [complete_initial_population.get_mean_distance_to_others()]

        initial_subpopulations = complete_initial_population.get_subpopulations(self._no_islands, False)
        evolutionary_algorithms = [self.__get_EA(subpopulation) for subpopulation in initial_subpopulations]
        with multiprocessing.Pool() as executor_pool:
            while(any([not ea.converged for ea in evolutionary_algorithms])):
                if ( iteration_number % self._island_swap_rate ) == 0:
                    pairs = _get_pairs_cycle(evolutionary_algorithms)
                    for pair in pairs:
                        pair[0].swap_individuals_with(pair[1], self._island_no_swapped_individuals)

                current_subpopulation_asyncresults = [executor_pool.apply_async(ea.run_iteration) for ea in evolutionary_algorithms]
                
                current_population = Population([individual for current_subpopulation_asyncresult in current_subpopulation_asyncresults for individual in current_subpopulation_asyncresult.get()], self._tsp.no_vertices)

                current_mean_fitness = self._tsp.mean_fitness(current_population.individuals, True)
                current_best_fitness = self._tsp.best_fitness(current_population.individuals)
                no_converged_algorithms = sum(1 for ea in evolutionary_algorithms if ea.converged)
                iteration_number += 1
                iteration_numbers.append(iteration_number - 1)
                mean_fitnesses.append(current_mean_fitness)
                best_fitnesses.append(current_best_fitness)
                
                if self._provide_analytics == True: mean_distances_to_others.append(current_population.get_mean_distance_to_others())
               
                # Call the reporter with:
                #  - the mean objective function value of the population
                #  - the best objective function value of the population
                #  - a 1D numpy array in the cycle notation containing the best solution 
                #    with city numbering starting from 0
                timeLeft = self.reporter.report(current_mean_fitness, current_best_fitness, self._tsp.best_individual(current_population.individuals).permutation)
                # Report iteration results
                if verbose == True:
                     logger.info(f"Iteration complete. {no_converged_algorithms} out of {len(evolutionary_algorithms)} islands converged, time left = {timeLeft:.3f} seconds")
                     logger.info(f"\tCurrent mean fitness = {current_mean_fitness:.5f}, current best fitness = {current_best_fitness:.5f}")
                # Stop optimizing if out of time
                if timeLeft < 0:
                    break
        
        # Report optimization speed to screen
        elapsed = timeit.default_timer() - start_time
        if verbose == True:
            logger.info(f"Evolutionary algorithm finished in {elapsed:.3f} seconds")
        # Report performance compared to heuristic benchmarks to screen
        last_mean_performance_difference_with_heuristic = 1 - (current_mean_fitness / nn_mean_fitness)
        last_best_performance_difference_with_heuristic = 1 - (current_best_fitness / nn_best_fitness)
        report = f"Last iteration mean fitness was {abs(last_mean_performance_difference_with_heuristic) * 100:.2f}% "
        report += "better" if last_mean_performance_difference_with_heuristic >= 0 else "worse"
        report += " than mean heuristic solution fitness"
        if verbose == True:
            logger.info(report)
        report = f"Last iteration best fitness was {abs(last_best_performance_difference_with_heuristic) * 100:.2f}% "
        report += "better" if last_best_performance_difference_with_heuristic >= 0 else "worse"
        report += " than best heuristic solution fitness"
        if verbose == True:
            logger.info(report)
        
        if self._provide_analytics == True:
            ticks_step = ((iteration_number + 1) // 10) + 1
            
            # Generate plots of the mean and best fitnesses as the iterations progress and save them to r0486848_means.png and r0486848_bests.png respectively
            plt.figure()
            plt.plot(iteration_numbers, mean_fitnesses, label="Mean fitness")
            plt.hlines(nn_mean_fitness, 0, len(iteration_numbers) - 1, label="Mean heuristic fitness", colors="r")
            plt.title('Mean fitness vs. iteration')
            plt.legend()
            plt.xlabel("Iteration")
            plt.ylabel("Fitness")
            plt.xlim([0, len(iteration_numbers) - 1])
            lower_y_bound = min(itertools.chain(mean_fitnesses, [nn_mean_fitness])) * 0.8
            upper_y_bound = max(itertools.chain(mean_fitnesses, [nn_mean_fitness])) * 1.2
            plt.ylim([lower_y_bound, upper_y_bound])
            plt.xticks(range(0, len(iteration_numbers), ticks_step))
            plt.savefig('r0486848_means.png')
            plt.figure()
            plt.plot(iteration_numbers, best_fitnesses, label="Best fitness")
            plt.hlines(nn_best_fitness, 0, len(iteration_numbers) - 1, label="Best heuristic fitness", colors="r")
            plt.title('Best fitness vs. iteration')
            plt.legend()
            plt.xlabel("Iteration")
            plt.ylabel("Fitness")
            plt.xlim([0, len(iteration_numbers) - 1])
            lower_y_bound = min(itertools.chain(best_fitnesses, [nn_best_fitness])) * 0.8
            upper_y_bound = max(itertools.chain(best_fitnesses, [nn_best_fitness])) * 1.2
            plt.ylim([lower_y_bound, upper_y_bound])
            plt.xticks(range(0, len(iteration_numbers), ticks_step))
            plt.savefig('r0486848_bests.png')

            # Plot the evolution of the mean distance from individuals to eachother in the population
            plt.figure()
            plt.plot(iteration_numbers, mean_distances_to_others, label="Mean distance from individuals to eachother")
            plt.title('Mean distance from individuals to eachother vs. iteration')
            plt.legend()
            plt.xlabel("Iteration")
            plt.ylabel("Mean distance from individuals to eachothe")
            plt.xlim([0, len(iteration_numbers) - 1])
            plt.ylim([0, self._tsp.no_vertices - 1])
            plt.xticks(range(0, len(iteration_numbers), ticks_step))
            plt.savefig('r0486848_mean_distance_from_individuals_to_eachother.png')
        
            # Plot the final population distribution and save it to r0486848_last_distribution.png
            plt.figure()
            plt.bar(range(self._tsp.no_vertices), current_population.get_distribution())
            plt.title('Distribution of individuals')
            plt.xlabel('Distance to identity permutation')
            plt.ylabel('# individuals')
            plt.savefig('r0486848_last_distribution.png')

            # Write the last population's contents to r0486848_last_population.txt        
            current_population.write_to_file("r0486848_last_population.txt")

        # Return performance results of the optimization
        return (current_mean_fitness, current_best_fitness)    

    def _get_benchmarks(self):
        """
        Returns the benchmark mean and best fitness of the set of solutions obtained by applying the nearest neighbour heuristic at each possible starting vertex

        :return: a tuple with the benchmark mean and best fitness for the final solution population
        """

        nn_individuals = []

        # Loop over all possible starting vertices
        for vertex in range(self._tsp.no_vertices):
            # Save its nearest neighbour solution
            nn_individuals.append(self.__get_nearest_neighbour_solution(vertex))
        
        # Return the mean and best fitness of the set of nearest neighbour solutions at each possible starting vertex
        return (self._tsp.mean_fitness(nn_individuals, True), self._tsp.best_fitness(nn_individuals))

    def _get_initial_population(self):
        """
        Creates a population based on the population generation settings
        
        :return: a Population object containing the generated initial population
        """
        individuals = []
        individuals.extend(self._get_random_initial_individuals(self._population_generation_settings.no_random_individuals))
        individuals.extend(self._get_nearest_neighbour_based_initial_individuals(self._population_generation_settings.no_base_nearest_neighbour_individuals, self._population_generation_settings.no_derived_nearest_neighbour_individuals))
        return Population(individuals, self._tsp.no_vertices)

    def _get_random_initial_individuals(self, no_random_individuals):
        """
        Generates no_random_individuals completely random individuals for an initial population
        
        :return: a list of the generated completely random individuals
        """
        return [Individual(np.random.permutation(self._tsp.no_vertices)) for _ in range(no_random_individuals)]

    def _get_nearest_neighbour_based_initial_individuals(self, no_base_nearest_neighbour_individuals, no_derived_nearest_neighbour_individuals):
        """
        Generates individuals by taking no_base_nearest_neighbour_individuals nearest neighbour solutions and
        extending them with no_derived_nearest_neighbour_individuals randomly swap mutated versions of them
        
        :return: a list of nearest neighbour based initial individuals
        """
        starting_individuals = []
        # Add the individuals for each vertex to the list (in parallel)
        with multiprocessing.Pool() as executor_pool:
            individuals_lists = [asyncresult.get() for asyncresult in [executor_pool.apply_async(self._generate_initial_nn_individuals_for, [vertex, no_derived_nearest_neighbour_individuals]) for vertex in np.random.default_rng().choice(self._tsp.no_vertices, size=no_base_nearest_neighbour_individuals, replace=False)]]
            for individuals_list in individuals_lists:
                starting_individuals.extend(individuals_list)

        # Return the generated initial individuals
        return starting_individuals

    def __get_EA(self, population):
        return EvolutionaryAlgorithm(self._tsp, self._recombination_operator, self._elimination_scheme, self._default_k, True, self._mu, self._no_individuals_to_keep, self._default_mutation_chance, self._mutation_chance_feedback, self._mutation_chance_self_adaptivity, self._stopping_ratio, self._tolerances, population)

    def __get_nearest_neighbour_solution(self, starting_vertex):
        """
        Returns an Individual object representing the solution constructed starting from a certain vertex with the nearest neighbour heuristic
        (https://en.wikipedia.org/wiki/Nearest_neighbour_algorithm)
    
        :param starting_vertex: integer representing the starting vertex
        :return: Individual object representing nearest neighbour solution starting at starting_vertex
        """

        nn_solution = np.empty(self._tsp.no_vertices)
        visited = []

        # Start at starting_vertex
        current_vertex = starting_vertex
        # Mark starting_vertex as visited
        visited.append(current_vertex)
        # Record starting_vertex in solution
        nn_solution[0] = current_vertex
        # Select next vertex as closest unvisited one
        edge_weights = np.copy(self._tsp.distance_matrix[current_vertex, :])
        edge_weights[visited] = np.Inf
        
        if np.isinf(np.min(edge_weights)):
            for v in range(np.size(edge_weights)):
                 if v not in visited:
                     next_vertex = v
                     break
        else:
            next_vertex = np.argmin(edge_weights)
        # Traverse the remaining vertices
        for i in range(1, self._tsp.no_vertices):
            # Go to the next vertex
            current_vertex = next_vertex
            # Mark next vertex as visited
            visited.append(current_vertex)
            # Record next vertex in solution
            nn_solution[i] = current_vertex
            # Select next vertex as closest unvisited one
            edge_weights = np.copy(self._tsp.distance_matrix[current_vertex, :])
            edge_weights[visited] = np.Inf
            if np.isinf(np.min(edge_weights)):
                for v in range(np.size(edge_weights)):
                    if v not in visited:
                        next_vertex = v
                        break
            else:
                next_vertex = np.argmin(edge_weights)
        
        return Individual(nn_solution, self._default_mutation_chance)

    def _generate_initial_nn_individuals_for(self, vertex, no_derived_nearest_neighbour_individuals):
        individuals = (no_derived_nearest_neighbour_individuals + 1) * [0]
         
        nn_solution = self.__get_nearest_neighbour_solution(vertex)
        # Add the nearest neighbour solution starting at the given vertex to the set
        individuals[0] = nn_solution

        for i in range(no_derived_nearest_neighbour_individuals):
            permutation = nn_solution.permutation
            # Sample the number of random swaps from a λ/4 - Poisson distribution
            # This way we will get a lot of desired variation, but we mostly won't jump too far away from probably-good solutions
            no_random_swaps = max(1, min(np.random.poisson(math.floor(self._population_generation_settings.total_population_size / 4)), self._population_generation_settings.total_population_size))
                
            # Execute the random swaps
            for _ in range(no_random_swaps):
                a = np.random.randint(0, self._tsp.no_vertices)
                b = np.random.randint(0, self._tsp.no_vertices)
                permutation = _get_swapped(permutation, a, b)

            # Add the new, mutated version of the nearest neighbour solution to the set
            individuals[(i + 1)] = Individual(permutation, self._default_mutation_chance)

        return individuals

class EvolutionaryAlgorithm:
    """
    This class implements the evolutionary algorithm - its main loop and all of its components (initialization, selection, mutation, recombination, elimination)
    """
    def __init__(self, tsp, recombination_operator, elimination_scheme, default_k, enable_k_adaptivity, mu, no_individuals_to_keep, default_mutation_chance, mutation_chance_feedback, mutation_chance_self_adaptivity, stopping_ratio, tolerances, population):
        """
        Constructs the evolutionary algorithm object
        
        :param reombination_operator: the recombination operator to use
        :param elimination_scheme: the elimination scheme
        :param k: tournament size for k-tournament selection
        :param mu: number of offspring to generate from the population
        :param no_individuals_to_keep: number of individuals to keep in elimination steps
        :param mutation_chance: number between 0 and 1 representing the chance of mutation
        :param mutation_chance_self_adaptivity: if set to True, mutation chance self-adaptivity is enabled
        :param stopping_ratio: the relative improvement in the current iteration compared to the previous one below which, after tolerances iterations, to stop optimization
        :param tolerances: the number of iterations ran below the stopping ratio before optimization is stopped
        :return: an initialized evolutionary algorithm object of class r0486848
        """
        # Copy given evolutionary algorithm arguments to attributes
        self._tsp = tsp
        self._recombination = {
            RecombinationOperator.PMX: self._recombination_PMX,
            RecombinationOperator.HGREX: self._recombination_HGreX
        }[recombination_operator]
        self._elimination = {
            EliminationScheme.LAMBDAPLUSMU: self._elimination_lambdaplusmu,
            EliminationScheme.LAMBDAPLUSMU_WCROWDING: self._elimination_lambdaplusmu_with_crowding,
            EliminationScheme.LAMBDA_MU: self._elimination_lambdamu
        }[elimination_scheme]
        self._default_k = default_k
        self._enable_k_adaptivity = enable_k_adaptivity
        self._mu = mu
        self._no_individuals_to_keep = no_individuals_to_keep
        self._default_mutation_chance = default_mutation_chance
        self._mutation_chance_feedback = mutation_chance_feedback
        self._mutation_chance_self_adaptivity = mutation_chance_self_adaptivity
        self._stopping_ratio = stopping_ratio
        self._tolerances = tolerances
        self._population = population
        
        # Set up convergence tracking
        self._current_mean_fitness = self._tsp.mean_fitness(self._population.individuals, True)
        self._current_change = math.nan
        self._change_ratio = float('inf')
        self._change_ratio_became_number = False
        self._change_ratios = FixedSizeStack(self._tolerances)
        self._change_ratios.push(self._change_ratio)
        
        self._converged = False
        
    def run_iteration(self):
        """
        Runs one iteration of the evolutionary algorithm
        
        :return: the population after the iteration of the evolutionary algorithm
        """       
        self._previous_mean_fitness = self._current_mean_fitness

        # Create μ offspring of the current population
        offspring = []

        while len(offspring) < self._mu:
            # Select two parents
            first_parent = self._selection()
            second_parent = self._selection()
            # Recombine them, mutate the recombination and save the resulting offspring
            recombinations = self._recombination(first_parent, second_parent)
            for recombination in recombinations:
                offspring.append(self._mutation(recombination)) 

        # Apply random mutation to each member of the population
        for idx, individual in enumerate(self._population):
            self._population.individuals[idx] = self._mutation(individual)

        # Apply elimination to the current population and its offspring, forming the new population
        self._population = Population(self._elimination(offspring), self._tsp.no_vertices)

        # Determine change ratio
        self._current_mean_fitness = self._tsp.mean_fitness(self._population.individuals, True)

        self._previous_change = self._current_change
        self._current_change = self._previous_mean_fitness - self._current_mean_fitness
        if math.isnan(self._previous_change):
            self._change_ratio = float('inf')
        else:
            self._change_ratio = abs(self._current_change) / (abs(self._previous_change) + sys.float_info.epsilon)
            if self._change_ratio_became_number == False:
                self._change_ratio_became_number = True
                self._first_change_ratio_number = self._change_ratio

        self._change_ratios.push(self._change_ratio)

        self._converged = not any([cr > self._stopping_ratio for cr in self._change_ratios])

        return self._population

    def swap_individuals_with(self, other_ea, no_individuals_to_swap):
        to_give_indices = np.random.choice(range(len(self._population.individuals)), no_individuals_to_swap)
        to_give = [self._population.individuals[index] for index in to_give_indices]
        to_get_indices = np.random.choice(range(len(other_ea._population.individuals)), no_individuals_to_swap)
        to_get = [other_ea._population.individuals[index] for index in to_get_indices]
        
        self._population.individuals.extend(to_get)
        for index in np.unique(to_get_indices)[::-1]:
            del other_ea._population.individuals[index]
        other_ea._population.individuals.extend(to_give)
        for index in np.unique(to_give_indices)[::-1]:
            del self._population.individuals[index]
        
    def _selection(self):
        """
        Performs k-tournament selection

        :return: a random, best-out-of-k Individual object
        """
        
        # Select k random individuals from the population
        selected = list(np.random.choice(self._population.individuals, self.k))
        # Map them to their fitness with a list comprehension
        fitnesses = [self._tsp.fitness(individual) for individual in selected]
        # Select the individual with the lowest (=best) fitness
        i = fitnesses.index(min(fitnesses))
        # Return said individual
        return selected[i]

    def _recombination_HGreX(self, first_parent, second_parent):
        """
        Performs a HGreX recombination on the two given Individual objects
    
        :param first_parent: the Individual object representing the first parent
        :param second_parent: the Individual object representing the second parent
        :return: a new Individual object representing the recombination of the parents
        """
        child_permutation = []
        
        # Choose a random vertex to start the child permutation at
        current_vertex = np.random.randint(self._tsp.no_vertices)
        # Save the starting vertex in the child permutation
        child_permutation.append(current_vertex)
        # Loop over the remaining positions in the child permutation
        for _ in range(self._tsp.no_vertices - 1):
            # Get information on the edges starting at the current vertex in the parents
            first_parent_current_vertex_idx = np.where(first_parent.permutation == current_vertex)[0][0]
            second_parent_current_vertex_idx = np.where(second_parent.permutation == current_vertex)[0][0]
            first_parent_edge_endpoint = first_parent.permutation[first_parent_current_vertex_idx + 1] if first_parent_current_vertex_idx < (self._tsp.no_vertices - 1) else first_parent.permutation[0]
            second_parent_edge_endpoint = second_parent.permutation[second_parent_current_vertex_idx + 1] if second_parent_current_vertex_idx < (self._tsp.no_vertices - 1) else second_parent.permutation[0]
            first_parent_edge_length = self._tsp.distance_matrix[int(current_vertex), int(first_parent_edge_endpoint)]
            second_parent_edge_length = self._tsp.distance_matrix[int(current_vertex), int(second_parent_edge_endpoint)]

            # If both parents' edges lead to a vertex that is already in the child permutation,
            # append the nearest, unused vertex to the child permutation and select it as the next vertex to visit         
            if (first_parent_edge_endpoint in child_permutation) and (second_parent_edge_endpoint in child_permutation):
                # Get the unused vertices in the child permutation
                possible_endpoints = [edge_endpoint for edge_endpoint in range(self._tsp.no_vertices) if edge_endpoint not in child_permutation and edge_endpoint != current_vertex]
                # Get the lengths of the corresponding edges starting at the current vertex
                possible_edge_lenghts = [self._tsp.distance_matrix[int(current_vertex), int(possible_endpoint)] for possible_endpoint in possible_endpoints]
                # Choose the vertex with the shortest corresponding edge starting at the current vertex as the next vertex
                chosen_endpoint = possible_endpoints[np.argmin(possible_edge_lenghts)]
                # Save the next vertex in the child permutation
                child_permutation.append(chosen_endpoint)
                # Go to next vertex
                current_vertex = chosen_endpoint
            
            # If only one of the parents' edges leads to a vertex that is already in the child permutation,
            # append the endpoint of the other's one to the child permutation and select it as the next vertex to visit
            elif (first_parent_edge_endpoint not in child_permutation) and (second_parent_edge_endpoint in child_permutation):
                # Save the next vertex in the child permutation
                child_permutation.append(first_parent_edge_endpoint)
                # Go to next vertex
                current_vertex = first_parent_edge_endpoint
            elif (first_parent_edge_endpoint in child_permutation) and (second_parent_edge_endpoint not in child_permutation):
                # Save the next vertex in the child permutation
                child_permutation.append(second_parent_edge_endpoint)
                # Go to next vertex
                current_vertex = second_parent_edge_endpoint
            else:
                if first_parent_edge_length <= second_parent_edge_length:
                    # Save the next vertex in the child permutation
                    child_permutation.append(first_parent_edge_endpoint)
                    # Go to next vertex
                    current_vertex = first_parent_edge_endpoint
                else:
                    # Save the next vertex in the child permutation
                    child_permutation.append(second_parent_edge_endpoint)
                    # Go to next vertex
                    current_vertex = second_parent_edge_endpoint
                  
        # Recombine the child's mutation chance with a blend recombination if self-adaptivity is enabled             
        if self._mutation_chance_self_adaptivity:
          child_mutation_chance = first_parent.mutation_chance + ((2 * np.random.rand() - 0.5) * abs(second_parent.mutation_chance - first_parent.mutation_chance))
        else:
          child_mutation_chance = self._default_mutation_chance
        # Return a new Individual object based on the recombined child permutation and mutation chance
        return [Individual(np.array(child_permutation), child_mutation_chance)]
        
    def _recombination_PMX(self, first_parent, second_parent):
        def PMX_core_logic():
            child_permutation = [None] * self._tsp.no_vertices   # child permutation will be a list of integers (not numpy list)
            # Copy the segment of parent 1 into the offspring
            child_permutation[index_begin:index_end + 1] = first_parent.permutation[index_begin:index_end + 1]
            covered_area = [*range(index_begin, index_end + 1)]   # keep track of what is already covered in child_permutation
            # loop over the elements of parent 2 at the segment locations
            for idx in range(index_begin,index_end+1):
                if second_parent.permutation[idx] in first_parent.permutation[index_begin:index_end + 1]:
                    pass   # this value has already found a place in the new child.
                else:
                    idx_search = idx
                    location_found = False
                    while not location_found:
                        idx_found = np.where(second_parent.permutation == child_permutation[idx_search])[0][0]
                        if child_permutation[idx_found] != None:    # found location in offspring is already occupied
                            idx_search = idx_found   # set index to be able to look for the new found value.
                        else:
                            location_found = True
                            covered_area.append(idx_found)
                    child_permutation[idx_found] = second_parent.permutation[idx]  # assign the new location of this value.
            # Determine which are the remaining elements of parent 2 to be copied to the offspring.
            idx_to_be_copied = list(set([*range(0, self._tsp.no_vertices)]) - set(covered_area))  # full list minus covered area
            # copy the remaining elements into the child.
            for ind in idx_to_be_copied:
                child_permutation[ind] = second_parent.permutation[ind]

            # Recombine the child's mutation chance with a blend recombination if self-adaptivity is enabled
            if self._mutation_chance_self_adaptivity:
                child_mutation_chance = first_parent.mutation_chance + ((2 * np.random.rand() - 0.5) * abs(second_parent.mutation_chance - first_parent.mutation_chance))
            else:
                child_mutation_chance = self._default_mutation_chance
            return Individual(np.array(child_permutation), child_mutation_chance)

        # Choose a random index vertex
        index_vertex1 = np.random.randint(self._tsp.no_vertices)
        # Choose another random index vertex
        index_vertex2 = index_vertex1
        # Make sure that the second crossoverpoint is different from the first corssoverpoint
        while index_vertex1 == index_vertex2:
            index_vertex2 = np.random.randint(self._tsp.no_vertices) # Choose another random index vertex
        # begin index has to be the smallest value, so sort the indices
        index_begin, index_end = np.sort(np.array([index_vertex1, index_vertex2]))
        # ##### check if the same index, then decide what to do... #####
        ############################
        # determine first and second child respectively, which are kind of symmetric
        child1 = PMX_core_logic()
        child2 = PMX_core_logic()
        return [child1, child2]

    def _mutation(self, individual):
        """
        Performs a random swap mutation on an Individual object with its mutation chance
    
        :param individual: the Individual object to mutate
        :return: a new Individual object representing the possibly mutated version of the individual
        """

        # Do the mutation with random chance individual.mutation_chance
        if np.random.rand() <= self.__get_mutation_chance(individual):
            # Draw two random numbers representing the indices of the vertices to be swapped in the permutation
            a = np.random.randint(0, self._tsp.no_vertices)
            b = np.random.randint(0, self._tsp.no_vertices)
            # Return a new Individual object with the swap
            return Individual(_get_swapped(individual.permutation, a, b), individual.mutation_chance)
        else:
            # The mutation should not happen here, return a copy of the original Individual
            return Individual(individual.permutation, individual.mutation_chance)

    def _elimination_lambdaplusmu(self, offspring):
        """
        Performs (λ + μ)-elimination on the current population extended with the given offspring
    
        :param offspring: a Python list of Individual objects representing the newly created offspring
        :return: a Python list of Individual objects representing a new, (λ + μ)-eliminated population
        """
        
        # Create the combined collection of the current population and given offspring
        combined = []
        combined.extend(self._population.individuals)
        combined.extend(offspring)
        combined = np.array(combined)
        
        # Get the no_individuals_to_keep shortest route individuals' indices in combined
        selected = np.argsort(np.array([self._tsp.fitness(individual) for individual in combined]))[0:self._no_individuals_to_keep]
        # Return the individuals at those indices
        return list(combined[selected])

    def _elimination_lambdaplusmu_with_crowding(self, offspring):
        """
        Performs (λ + μ)-elimination with crowding on the current population extended with the given offspring
    
        :param offspring: a Python list of Individual objects representing the newly created offspring
        :return: a Python list of Individual objects representing a new, (λ + μ)-crowding-eliminated population
        """
        k = 3
        
        selected = []

        # Create the combined collection of the current population and given offspring
        combined = []
        combined.extend(self._population.individuals)
        combined.extend(offspring)
        
        for _ in range(self._no_individuals_to_keep):
            if len(combined) == 0:
                break
            
            currently_selected_individual = np.argmin([self._tsp.fitness(individual) for individual in combined])
            selected.append(combined[currently_selected_individual])
            if len(combined) > 1:
                random_others = np.random.choice([*range(currently_selected_individual), *range(currently_selected_individual + 1, len(combined))], k)
                individual_to_delete = random_others[np.argmin([combined[other].distance_to_other(combined[currently_selected_individual]) for other in random_others])]
                del combined[max(currently_selected_individual, individual_to_delete)]
                del combined[min(currently_selected_individual, individual_to_delete)]
            else:
                del combined[currently_selected_individual]

        return list(selected)

    def _elimination_lambdamu(self, offspring):
        """
        Performs (λ, μ)-elimination
    
        :param offspring: a Python list of Individual objects representing the newly created offspring
        :return: a Python list of Individual objects representing a new, (λ, μ)-eliminated population
        """
        offspring_array = np.array(offspring)
        
        # Get the no_individuals_to_keep shortest route individuals' indices in offspring
        selected = np.argsort(np.array([self._tsp.fitness(individual) for individual in offspring_array]))[0:self._no_individuals_to_keep]
        # Return the individuals at those indices
        return list(offspring_array[selected])

    def __get_mutation_chance(self, individual):
        if self._mutation_chance_self_adaptivity == True:
            return individual.mutation_chance
        else:
            if self._mutation_chance_feedback == True:
                if self._change_ratio_became_number == False:
                    return self._default_mutation_chance
                else:
                    b = self._first_change_ratio_number
                    a = math.log(((2 * self._default_mutation_chance) - 0.01) / 0.01) / (b - self._stopping_ratio)
                    return math.ceil((2 * self._default_mutation_chance) / (1 + math.exp(-a * (self._change_ratio - b))))
            else:
                return self._default_mutation_chance
    
    @property
    def converged(self):
        return self._converged

    @property
    def k(self):
        if self._enable_k_adaptivity == True:
            if self._change_ratio_became_number == False:
                return self._default_k
            else:
                b = self._first_change_ratio_number
                a = math.log(((2 * self._default_k) - 1) / 1) / (b - self._stopping_ratio)
                return math.ceil((2 * self._default_k) / (1 + math.exp(-a * (self._change_ratio - b))))
        else:
            return self._default_k

class TSP:

    """
    This class represents the TSP problem instance we're trying to solve

    :attribute distance_matrix: a 2D Numpy array where element [i, j] = d(i, j) in the TSP graph, not symmetric!
    :attribute no_vertices: the number of vertices (cities) in the problem
    """
    @staticmethod
    def get_random(no_vertices, infinite_edge_chance=0.1):
        distance_matrix = np.random.rand(no_vertices, no_vertices)
        distance_matrix_reshaped = distance_matrix.reshape(no_vertices ** 2)
        for idx, _ in np.ndenumerate(distance_matrix_reshaped):
            if (idx[0] % (no_vertices + 1)) == 0:
                distance_matrix_reshaped[idx] = 0
            elif np.random.rand() <= infinite_edge_chance:
                distance_matrix_reshaped[idx] = math.inf

        return TSP(distance_matrix)

    def __init__(self, distance_matrix):
        self._distance_matrix = distance_matrix

    def fitness(self, individual):
        """
        Returns the total length of the tour represented by the individual

        :param individual: the Individual object representing the individual of interest
        :return: the total length of the tour represented by the individual
        """
        
        total_distance = 0
        # This loops over all the edges in the individual
        # a and b are the edge endpoints
        for a, b in zip(individual.permutation[0:(self.no_vertices - 1)], individual.permutation[1:self.no_vertices]):
            total_distance += self._distance_matrix[int(a), int(b)]

        # We have to go back to our starting point, so we have to count that the distance from end to start too
        total_distance += self._distance_matrix[int(b), int(individual.permutation[0])]

        return total_distance

    def mean_fitness(self, individuals, filter_infeasibles=False):
        """
        Returns the mean total length of the tours represented by the different individuals

        :param individuals: a Python list of Individual objects
        :return: the mean total length of the tours represented by the individuals
        """
        if filter_infeasibles:
            filtered_fitnesses = list(filter(lambda x: not np.isinf(x), [self.fitness(individual) for individual in individuals]))
            if len(filtered_fitnesses) != 0:
                return statistics.mean(filtered_fitnesses)
            else:
                return math.inf
        else:
            return statistics.mean([self.fitness(individual) for individual in individuals])

    def best_fitness(self, individuals):
        """
        Returns the shortest total length of the tours represented by the different individuals

        :param individuals: a Python list of Individual objects
        :return: the shortest total length of the tours represented by the individuals
        """
        
        return min([self.fitness(individual) for individual in individuals])

    def best_individual(self, individuals):
        """
        Returns the shortest total length of the tours represented by the different individuals

        :param individuals: a Python list of Individual objects
        :return: the shortest total length of the tours represented by the individuals
        """

        return individuals[np.argmin([self.fitness(individual) for individual in individuals])]

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
    def _distance_to_identity(permutation):
        """
        Returns the number of swaps needed to transform the identity permutation into the given permutation
        
        :return: the number of swaps needed to transform the identity permutation into the given permutation
        """
        return np.size(permutation) - Individual._no_cycles(permutation)

    @staticmethod
    def _no_cycles(permutation):
        """
        Returns the number of cycles in the given permutation
        
        :return: the number of cycles in the given permutation
        """
        # The key idea of this implementation is the fact that the number of swaps required to transform the identity
        # permutation into a given permutation is equal to the length of the permutation - its number of cycles
        no_cycles = 0
        identity_vertices = [{ "vertex": vertex, "visited": False } for vertex in range(np.size(permutation))]
        for current_identity_vertex in identity_vertices:
            if current_identity_vertex["visited"] == False:
                # Count the cycles
                current_vertex = permutation[current_identity_vertex["vertex"]]
                while current_vertex != current_identity_vertex["vertex"]:
                    identity_vertices[int(current_vertex)]["visited"] = True
                    current_vertex = permutation[int(current_vertex)]

                current_identity_vertex["visited"] = True
                
                no_cycles += 1
        
        return no_cycles

    """
    This class represents an individual based on a permutation stored in a 1D Numpy array

    :attribute permutation: a 1D Numpy array containing integers representing the permutation
    :attribute mutation_chance: the chance that this individual will mutate
    """
    def __init__(self, permutation, mutation_chance = 0.05):
        self._permutation = permutation
        self._mutation_chance = mutation_chance                                                                                     # 

    def distance_to_identity(self):
        """
        Returns the number of swaps needed to transform the identity permutation into the individual's permutation
        
        :return: the number of swaps needed to transform the identity permutation into the individual's permutation
        """
        return self._distance_to_identity(self._permutation)

    def distance_to_other(self, other):
        def compose(first, second):
            n = np.size(first, 0)
            composed = np.zeros(n)

            for i in range(n):
                composed[i] = second[int(first[i])]

            return composed

        def inverse(permutation):
            n = np.size(permutation, 0)
            inverse_permutation = np.zeros(n)

            for i in range(n):
                inverse_permutation[int(permutation[i])] = i

            return inverse_permutation
            
        return self._no_cycles(compose(inverse(other.permutation), self._permutation))

    @property
    def permutation(self):
        return self._permutation

    @permutation.setter
    def permutation(self, permutation):
        self._permutation = permutation

    @property
    def mutation_chance(self):
        return self._mutation_chance

    @mutation_chance.setter
    def mutation_chance(self, mutation_chance):
        self._mutation_chance = mutation_chance
        
class Population:
    """
    This class holds a collection of individuals

    :attribute individuals: a Python list of Individual objects
    :attribute no_vertices: the number of vertices in an Individual
    """
    def __init__(self, individuals, no_vertices):
        self._individuals = individuals
        self._no_vertices = no_vertices

    def __iter__(self):
        return iter(self._individuals)

    def get_distribution(self):
        """
        Returns the number of individuals at different Hamming distances from the identity permutation

        :return: the number of individuals at different Hamming distances from the identity permutation as a Python list
        """
        bins = [0] * self._no_vertices
        
        for distance_to_identity in [individual.distance_to_identity() for individual in self._individuals]:
            bins[distance_to_identity] += 1

        return bins

    def get_mean_distance_to_others(self):
        no_individuals = len(self._individuals)
        distance_matrix = np.zeros((no_individuals, no_individuals))
        
        for individual_index in range(no_individuals):
            for other_individual_index in range((individual_index + 1), no_individuals):
                d = self._individuals[individual_index].distance_to_other(self._individuals[other_individual_index])
                distance_matrix[individual_index, other_individual_index] = d
                distance_matrix[other_individual_index, individual_index] = d
        
        return (np.mean(distance_matrix) * ((no_individuals ** 2)/((no_individuals ** 2) - no_individuals)))
    
    def get_stdev_distance_to_identity(self):
        """
        Returns the standard deviation of the Hamming distances to the identity permutation

        :return: the standard deviation of the Hamming distances to the identity permutation
        """
        return statistics.stdev([individual.distance_to_identity() for individual in self._individuals])

    def write_to_file(self, filename):
        """
        Writes the population content out to the file specified in filename
        """
        file = open(filename, "w")
        for individual in self._individuals:
            file.write(np.array_str(individual.permutation))

        file.close()

    def get_subpopulations(self, m, return_rest = False):
        shuffled_individuals = self._individuals.copy()
        random.shuffle(shuffled_individuals)
        e = int((len(shuffled_individuals) - (len(shuffled_individuals) % m)) / m)
        sublists = [shuffled_individuals[(i * e):((i + 1) * e)] for i in range(m)]
        if return_rest == True:
            sublists.append(shuffled_individuals[(len(shuffled_individuals) - (len(shuffled_individuals) % m)):])
        return [Population(sublist, self._no_vertices) for sublist in sublists]
    
    @property
    def size(self):
        return len(self._individuals)
        
    @property
    def individuals(self):
        return self._individuals

    @individuals.setter
    def individuals(self, individuals):
        self._individuals = individuals

class FixedSizeStack:
    """
    This class implements a very simple iterable fixed size stack (to which you can only push) of variable size N
    
    Once more than N elements are pushed onto this stack, elements from the bottom are discarded to make space for them
    """

    def __init__(self, N):
        self._N = N
        self._internal_deque = collections.deque()
        self._elements_pushed = 0

    def push(self, e):
        """
        Pushes an element onto the fixed size stack

        :param e: object to push onto the fixed size stack
        """
   
        if self._elements_pushed >= self._N:
            self._internal_deque.popleft()

        self._internal_deque.append(e)     
        
        self._elements_pushed += 1
        
    # Method needed to make objects of this class iterable
    def __iter__(self):
        # Just return the iterator provided by the deque class
        return iter(self._internal_deque)

    @property
    def N(self):
        return self._N
        
class Combinations:
    def __init__(self, *args):
        self._lists = [arg for arg in args if len(arg) > 0]
        self._no_lists = len(self._lists)
        self._lists_lengths = [len(l) for l in self._lists]
        self._cur_pos = self._no_lists * [0]
        self._max_pos = [(length - 1) for length in self._lists_lengths]

        self._finished = False

    def __iter__(self):
        return self

    def __len__(self):
        return np.prod(self._lists_lengths)

    def __next__(self):
        if self._finished == True:
            raise StopIteration
        else:
            to_return = ()
            for l, i in enumerate(self._cur_pos):
                to_return += (self._lists[l][i], )

            if self._cur_pos == self._max_pos:
                 self._finished = True
            else:
                for i in range((self._no_lists - 1), -1, -1):
                    if self._cur_pos[i] != (self._lists_lengths[i] - 1):
                        self._cur_pos[i] += 1
                        break
                    else:
                        self._cur_pos[i] = 0
                
            return to_return
