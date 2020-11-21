import math
import statistics
import collections
import itertools
import timeit
import sys
import numpy as np
import matplotlib.pyplot as plt
import profile_decorator
import Reporter

class r0123456:
    """
    This class implements the evolutionary algorithm - its main loop and all of its components (initialization, selection, mutation, recombination, elimination)
    """

    @staticmethod
    def __get_swapped(permutation, a, b):
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

    def __init__(self, population_size_factor, k, mu, no_individuals_to_keep, stopping_ratio, tolerances):
        """
        Constructs the evolutionary algorithm object
        
        :param population_size_factor: population size is number of vertices in problem * this parameter (default = 5)
        :param k: tournament size for k-tournament selection (default = 5)
        :param mu: number of offspring to generate from the population (default = 100)
        :param no_individuals_to_keep: number of individuals to keep in elimination steps (default = 100)
        :param stopping_ratio: the relative improvement in the current iteration compared to the previous one below which, after tolerances iterations, to stop optimization (default = 0.001)
        :param tolerances: the number of iterations ran below the stopping ratio before optimization is stopped (default = 0.001)
        :return: an initialized evolutionary algorithm object of class r0123456
        """
        
        # Initialize and save Reporter class instance
        self.reporter = Reporter.Reporter(self.__class__.__name__)

        # Copy given evolutionary algorithm parameters to attributes
        self._population_size_factor = population_size_factor
        self._k = k
        self._mu = mu
        self._no_individuals_to_keep = no_individuals_to_keep
        self._stopping_ratio = stopping_ratio
        self._tolerances = tolerances
        # print(population_size_factor, k, mu, no_individuals_to_keep, stopping_ratio, tolerances)
        # Initialize remaining attributes
        self._population = None
        self._tsp = None
        self._population_size = math.nan
        self._type_recombination = 'PMX1'   # PMX1 or  HGreX1
        
    @profile_decorator.profile("algorithm_profile.txt")
    def optimize(self, filename):
        """
        Runs the evolutionary algorithm and reports its' results
        
        :param filename: a string containing the filename of a CSV-file containing the TSP instance
        :return: a tuple with the mean and best fitness of the last population
        """
        
        print("Starting evolutionary algorithm ...")
        # Start timer for assessing optimization speed
        start_time = timeit.default_timer()

        # Import TSP instance
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()
        
        # Set up TSP instance representation
        self._tsp = TSP(distanceMatrix)
        self._population_size = self._population_size_factor * self._tsp.no_vertices
        
        # Report TSP instance heuristic benchmark performance
        nn_mean_fitness, nn_best_fitness = self._get_benchmarks()
        print(f"Benchmarks:\n\tMean heuristic fitness = {nn_mean_fitness:.5f}\n\tBest heuristic fitness = {nn_best_fitness:.5f}")
        
        # Initialize the population
        self._initialize_population()
        
        # Set up main loop variables
        current_mean_fitness = self._tsp.mean_fitness(self._population.individuals)
        current_best_fitness = self._tsp.best_fitness(self._population.individuals)
        iteration_number = 0
        iteration_numbers = [iteration_number]
        mean_fitnesses = [current_mean_fitness]
        best_fitnesses = [current_best_fitness]
        current_change = math.nan
        change_ratio = float('inf')
        change_ratios = FixedSizeStack(self._tolerances)
        change_ratios.push(change_ratio)
        stdev_hamming_distances = [self._population.get_stdev_distance_to_identity()]
        print("Entering main loop")
        while( any([cr > self._stopping_ratio for cr in change_ratios]) ): # Keep optimizing while change ratio is large enough
            previous_mean_fitness = current_mean_fitness
            previous_best_fitness = current_best_fitness

            # Create μ offspring of the current population
            offspring = []
            if self._type_recombination == 'HGreX1':
                for _ in range(self._mu):
                    # Select two parents
                    first_parent = self._selection()
                    second_parent = self._selection()
                    # Recombine them, mutate the recombination and save the resulting offspring
                    offspring.append(self._mutation(self._recombination(first_parent, second_parent)))
            elif self._type_recombination == 'PMX1':
                two_individuals = []
                for _ in range(int(self._mu/2)):
                    # Select two parents
                    first_parent = self._selection()
                    second_parent = self._selection()
                    # Recombine them, mutate the recombination and save the resulting offspring
                    two_individuals = self._recombinationPMX1(first_parent, second_parent)   #  PMX recombination results in two symmetric childs
                    for ind, individual in enumerate(two_individuals):
                        offspring.append(self._mutation(individual))
            else:
                pass

            # Apply random mutation to each member of the population
            for idx, individual in enumerate(self._population):
                self._population.individuals[idx] = self._mutation(individual)

            # Apply elimination to the current population and its offspring, forming the new population
            self._population = Population(self._elimination(offspring), self._tsp.no_vertices)

            # Determine change ratio
            current_mean_fitness = self._tsp.mean_fitness(self._population.individuals)
            current_best_fitness = self._tsp.best_fitness(self._population.individuals)

            iteration_number += 1
            iteration_numbers.append(iteration_number)
            mean_fitnesses.append(current_mean_fitness)
            best_fitnesses.append(current_best_fitness)
        
            previous_change = current_change
            current_change = previous_mean_fitness - current_mean_fitness
            if math.isnan(previous_change):
                change_ratio = float('inf')
            else:
                change_ratio = abs(current_change) / (abs(previous_change) + sys.float_info.epsilon)
            change_ratios.push(change_ratio)

            stdev_hamming_distances.append(self._population.get_stdev_distance_to_identity())

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution 
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(current_mean_fitness, current_best_fitness, self._tsp.best_individual(self._population.individuals).permutation)
            # Report iteration results
            print(f"Iteration complete. Change ratio = {change_ratio:.5f}, time left = {timeLeft:.3f} seconds")
            print(f"\tCurrent mean fitness = {current_mean_fitness:.5f}, current best fitness = {current_best_fitness:.5f}")
            # Stop optimizing if out of time
            if timeLeft < 0:
                break

        # Report optimization speed to screen
        elapsed = timeit.default_timer() - start_time
        print(f"Evolutionary algorithm finished in {elapsed:.3f} seconds")
        # Report performance compared to heuristic benchmarks to screen
        last_mean_performance_difference_with_heuristic = 1 - (current_mean_fitness / nn_mean_fitness)
        last_best_performance_difference_with_heuristic = 1 - (current_best_fitness / nn_best_fitness)
        report = f"Last iteration mean fitness was {last_mean_performance_difference_with_heuristic * 100:.2f}% "
        report += "better" if last_mean_performance_difference_with_heuristic >= 0 else "worse"
        report += " than mean heuristic solution fitness"
        print(report)
        report = f"Last iteration best fitness was {last_best_performance_difference_with_heuristic * 100:.2f}% "
        report += "better" if last_best_performance_difference_with_heuristic >= 0 else "worse"
        report += " than best heuristic solution fitness"
        print(report)

        # Generate plots of the mean and best fitnesses as the iterations progress and save them to r0123456_means.png and r0123456_bests.png respectively
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
        plt.xticks(range(0, len(iteration_numbers) , 1))
        plt.savefig('r0123456_means.png')
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
        plt.xticks(range(0, len(iteration_numbers), 1))
        plt.savefig('r0123456_bests.png')

        # Plot the evolution of the standard deviation of the Hamming distances to the identity permutations
        plt.figure()
        plt.plot(iteration_numbers, stdev_hamming_distances, label="σ of Hamming distances to the identity permutation")
        plt.title('σ of Hamming distances to the identity permutation vs. iteration')
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("σ")
        plt.xlim([0, len(iteration_numbers) - 1])
        plt.ylim([0, self._tsp.no_vertices - 1])
        plt.xticks(range(0, len(iteration_numbers), 1))
        plt.savefig('r0123456_stdev_distances.png')
        
        # Plot the final population distribution and save it to r0123456_last_distribution.png
        plt.figure()
        plt.bar(range(self._tsp.no_vertices), self._population.get_distribution())
        plt.title('Distribution of individuals')
        plt.xlabel('Distance to identity permutation')
        plt.ylabel('# individuals')
        plt.savefig('r0123456_last_distribution.png')

        # Write the last population's contents to r0123456_last_population.txt        
        self._population.write_to_file("r0123456_last_population.txt")

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
        return (self._tsp.mean_fitness(nn_individuals), self._tsp.best_fitness(nn_individuals))

    def _initialize_population(self):
        """
        Initializes the population by taking the set of nearest neighbour solutions starting at each possible vertex and
        extending it with (population_size_factor - 1) randomly swap mutated versions of each member
        """
        
        starting_individuals = []
        
        # Add the set of nearest neighbour solutions starting at each possible vertex
        for vertex in range(self._tsp.no_vertices):
            starting_individuals.append(self.__get_nearest_neighbour_solution(vertex))
        
        # Create the (population_size_factor - 1) randomly swap mutated versions of each member
        for _ in range(self._population_size_factor - 1):
            for vertex in range(self._tsp.no_vertices):
                permutation = starting_individuals[vertex].permutation
                # Sample the number of random swaps from a λ/4 - Poisson distribution
                # This way we will get a lot of desired variation, but we mostly won't jump too far away from probably-good solutions
                no_random_swaps = max(1, min(np.random.poisson(math.floor(self._population_size / 4)), self._population_size))
                
                # Execute the random swaps
                for _ in range(no_random_swaps):
                    a = np.random.randint(0, self._tsp.no_vertices)
                    b = np.random.randint(0, self._tsp.no_vertices)
                    permutation = self.__get_swapped(permutation, a, b)
                
                # Add the new, mutated version of the current member to the population
                starting_individuals.append(Individual(permutation))

        # Set the population attribute to the initial population generated in this method
        self._population = Population(starting_individuals, self._tsp.no_vertices)

    def _selection(self):
        """
        Performs k-tournament selection

        :return: a random, best-out-of-k Individual object
        """
        
        # Select k random individuals from the population
        selected = list(np.random.choice(self._population.individuals, self._k))
        # Map them to their fitness with a list comprehension
        fitnesses = [self._tsp.fitness(individual) for individual in selected]
        # Select the individual with the lowest (=best) fitness
        i = fitnesses.index(min(fitnesses))
        # Return said individual
        return selected[i]

    def _recombination(self, first_parent, second_parent):
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
                  
        # Recombine the child's mutation chance with a blend recombination                  
        child_mutation_chance = first_parent.mutation_chance + ((2 * np.random.rand() - 0.5) * abs(second_parent.mutation_chance - first_parent.mutation_chance))

        # Return a new Individual object based on the recombined child permutation and mutation chance
        return Individual(np.array(child_permutation), child_mutation_chance)
        
    def _recombinationPMX1(self,first_parent,second_parent):
        # Choose a random index vertex
        index_vertex1 = np.random.randint(self._tsp.no_vertices)
        # Choose another random index vertex
        index_vertex2 = index_vertex1
        # Make sure that the second crossoverpoint is different from the first corssoverpoint
        while index_vertex1 == index_vertex2:
            index_vertex2 = np.random.randint(self._tsp.no_vertices) # Choose another random index vertex
        # begin index has to be the smallest value, so sort the indices
        index_begin, index_end = np.sort(np.array([index_vertex1,index_vertex2]))
        # ##### check if the same index, then decide what to do... #####
        ############################
        # determine first and second child respectively, which are kind of symmetric
        child1 = self._PMXimplementation(first_parent,second_parent,index_begin,index_end)
        child2 = self._PMXimplementation(second_parent, first_parent, index_begin, index_end)
        return [child1,child2]

    def _PMXimplementation(self,first_parent,second_parent,index_begin,index_end):
        child_permutation = [None]*self._tsp.no_vertices   # child permutation will be a list of integers (not numpy list)
        # Copy the segment of parent 1 into the offspring
        child_permutation[index_begin:index_end+1] = first_parent.permutation[index_begin:index_end+1]
        covered_area = [*range(index_begin,index_end+1)]   # keep track of what is already covered in child_permutation
        # loop over the elements of parent 2 at the segment locations
        for idx in range(index_begin,index_end+1):
            if second_parent.permutation[idx] in first_parent.permutation[index_begin:index_end+1]:
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
        idx_to_be_copied = list(set([*range(0,self._tsp.no_vertices)]) - set(covered_area))  # full list minus covered area
        # copy the remaining elements into the child.
        for ind in idx_to_be_copied:
            child_permutation[ind] = second_parent.permutation[ind]

        # Recombine the child's mutation chance with a blend recombination
        child_mutation_chance = first_parent.mutation_chance + ((2 * np.random.rand() - 0.5) * abs(second_parent.mutation_chance - first_parent.mutation_chance))
        return Individual(np.array(child_permutation), child_mutation_chance)

    def _mutation(self, individual):
        """
        Performs a random swap mutation on an Individual object with its mutation chance
    
        :param individual: the Individual object to mutate
        :return: a new Individual object representing the possibly mutated version of the individual
        """

        # Do the mutation with random chance individual.mutation_chance
        if np.random.rand() <= individual.mutation_chance:
            # Draw two random numbers representing the indices of the vertices to be swapped in the permutation
            a = np.random.randint(0, self._tsp.no_vertices)
            b = np.random.randint(0, self._tsp.no_vertices)
            # Return a new Individual object with the swap
            return Individual(self.__get_swapped(individual.permutation, a, b), individual.mutation_chance)
        else:
            # The mutation should not happen here, return a copy of the original Individual
            return Individual(individual.permutation, individual.mutation_chance)
      
    def _elimination(self, offspring):
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
            next_vertex = np.argmin(edge_weights)

        return Individual(nn_solution)

class TSP:
    """
    This class represents the TSP problem instance we're trying to solve

    :attribute distance_matrix: a 2D Numpy array where element [i, j] = d(i, j) in the TSP graph, not symmetric!
    :attribute no_vertices: the number of vertices (cities) in the problem
    """

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

    def mean_fitness(self, individuals):
        """
        Returns the mean total length of the tours represented by the different individuals

        :param individuals: a Python list of Individual objects
        :return: the mean total length of the tours represented by the individuals
        """
        
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
        # The key idea of this implementation is the fact that the number of swaps required to transform the identity
        # permutation into a given permutation is equal to the length of the permutation - its number of cycles
        no_cycles = 0
        identity_vertices = [{ "vertex": vertex, "visited": False } for vertex in range(np.size(self._permutation))]
        
        for current_identity_vertex in identity_vertices:
            if current_identity_vertex["visited"] == False:
                # Count the cycles
                current_vertex = self._permutation[current_identity_vertex["vertex"]]
                while current_vertex != current_identity_vertex["vertex"]:
                    identity_vertices[int(current_vertex)]["visited"] = True
                    current_vertex = self._permutation[int(current_vertex)]
                
                current_identity_vertex["visited"] = True
                
                no_cycles += 1
        
        return np.size(self._permutation) - no_cycles

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