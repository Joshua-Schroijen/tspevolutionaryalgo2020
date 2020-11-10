import math
import statistics
import itertools
import timeit
import numpy as np
import matplotlib.pyplot as plt
import Reporter

class r0123456:
    @staticmethod
    def __get_swapped(permutation, a, b):
        permutation_copy = np.copy(permutation)
        permutation_copy[[b, a]] = permutation_copy[[a, b]]
        return permutation_copy

    def __init__(self, population_size_factor = 5, k = 5, mu = 100, no_individuals_to_keep = 100, stopping_ratio = 0.01):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

        self._population_size_factor = population_size_factor
        self._k = k
        self._mu = mu
        self._no_individuals_to_keep = no_individuals_to_keep
        self._stopping_ratio = stopping_ratio

        self._population = None
        self._tsp = None
        self._population_size = math.nan
        
    def optimize(self, filename):
        print("Starting evolutionary algorithm ...")
        start_time = timeit.default_timer()

        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()
        
        self._tsp = TSP(distanceMatrix)
        self._population_size = self._population_size_factor * self._tsp.no_vertices
        
        nn_mean_fitness, nn_best_fitness = self._get_benchmarks()
        print(f"Benchmarks:\n\tMean heuristic fitness = {nn_mean_fitness:.5f}\n\tBest heuristic fitness = {nn_best_fitness:.5f}")
         
        self._initialize_population()
        
        current_mean_fitness = self._tsp.mean_fitness(self._population.individuals)
        current_best_fitness = self._tsp.best_fitness(self._population.individuals)
        iteration_number = 0
        iteration_numbers = [iteration_number]
        mean_fitnesses = [current_mean_fitness]
        best_fitnesses = [current_best_fitness]
        current_change = math.nan
        change_ratio = float('inf')
        print("Entering main loop")
        while( change_ratio > self._stopping_ratio ):
            previous_mean_fitness = current_mean_fitness
            previous_best_fitness = current_best_fitness

            offspring = []
            for _ in range(self._mu):
                first_parent = self._selection()
                second_parent = self._selection()
                offspring.append(self._mutation(self._recombination(first_parent, second_parent)))

            for idx, individual in enumerate(self._population):
                self._population.individuals[idx] = self._mutation(individual)

            self._population = Population(self._elimination(offspring))

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
                change_ratio = abs(current_change / previous_change)

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution 
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(current_mean_fitness, current_best_fitness, self._tsp.best_individual(self._population.individuals).permutation)
            print(f"Iteration complete. Change ratio = {change_ratio:.5f}, time left = {timeLeft:.3f} seconds")
            print(f"\tCurrent mean fitness = {current_mean_fitness:.5f}, current best fitness = {current_best_fitness:.5f}")
            if timeLeft < 0:
                break

        elapsed = timeit.default_timer() - start_time
        print(f"Evolutionary algorithm finished in {elapsed:.3f} seconds")
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

        return 0

    def _get_benchmarks(self):
        nn_individuals = []

        for vertex in range(self._tsp.no_vertices):
            nn_individuals.append(self.__get_nearest_neighbour_solution(vertex))
            
        return (self._tsp.mean_fitness(nn_individuals), self._tsp.best_fitness(nn_individuals))

    def _initialize_population(self):
        starting_individuals = []
        
        for vertex in range(self._tsp.no_vertices):
            starting_individuals.append(self.__get_nearest_neighbour_solution(vertex))
        
        for _ in range(self._population_size_factor - 1):
            for vertex in range(self._tsp.no_vertices):
                permutation = starting_individuals[vertex].permutation
                no_random_swaps = max(1, min(np.random.poisson(math.floor(self._population_size / 4)), self._population_size))
                
                for _ in range(no_random_swaps):
                    a = np.random.randint(0, self._tsp.no_vertices)
                    b = np.random.randint(0, self._tsp.no_vertices)
                    permutation = self.__get_swapped(permutation, a, b)
                
                starting_individuals.append(Individual(permutation))

        self._population = Population(starting_individuals)

    def _selection(self):
        selected = list(np.random.choice(self._population.individuals, self._k))
        fitnesses = [self._tsp.fitness(individual) for individual in selected]
        i = fitnesses.index(min(fitnesses))
        return selected[i]

    def _recombination(self, first_parent, second_parent):
        child_permutation = []
        
        current_vertex = np.random.randint(self._tsp.no_vertices)
        child_permutation.append(current_vertex)
        for _ in range(self._tsp.no_vertices - 1):
            first_parent_current_vertex_idx = np.where(first_parent.permutation == current_vertex)[0][0]
            second_parent_current_vertex_idx = np.where(second_parent.permutation == current_vertex)[0][0]
            first_parent_edge_endpoint = first_parent.permutation[first_parent_current_vertex_idx + 1] if first_parent_current_vertex_idx < (self._tsp.no_vertices - 1) else first_parent.permutation[0]
            second_parent_edge_endpoint = second_parent.permutation[second_parent_current_vertex_idx + 1] if second_parent_current_vertex_idx < (self._tsp.no_vertices - 1) else second_parent.permutation[0]
            first_parent_edge_length = self._tsp.distance_matrix[int(current_vertex), int(first_parent_edge_endpoint)]
            second_parent_edge_length = self._tsp.distance_matrix[int(current_vertex), int(second_parent_edge_endpoint)]
            
            if (first_parent_edge_endpoint in child_permutation) and (second_parent_edge_endpoint in child_permutation):
                possible_endpoints = [edge_endpoint for edge_endpoint in range(self._tsp.no_vertices) if edge_endpoint not in child_permutation and edge_endpoint != current_vertex]
                possible_edge_lenghts = [self._tsp.distance_matrix[int(current_vertex), int(possible_endpoint)] for possible_endpoint in possible_endpoints]
                chosen_endpoint = possible_endpoints[np.argmin(possible_edge_lenghts)]
                child_permutation.append(chosen_endpoint)
                current_vertex = chosen_endpoint
            elif (first_parent_edge_endpoint not in child_permutation) and (second_parent_edge_endpoint in child_permutation):
                child_permutation.append(first_parent_edge_endpoint)
                current_vertex = first_parent_edge_endpoint
            elif (first_parent_edge_endpoint in child_permutation) and (second_parent_edge_endpoint not in child_permutation):
                child_permutation.append(second_parent_edge_endpoint)
                current_vertex = second_parent_edge_endpoint
            else:
                if first_parent_edge_length <= second_parent_edge_length:
                    child_permutation.append(first_parent_edge_endpoint)
                    current_vertex = first_parent_edge_endpoint
                else:
                    child_permutation.append(second_parent_edge_endpoint)
                    current_vertex = second_parent_edge_endpoint
                    
        child_mutation_chance = first_parent.mutation_chance + ((2 * np.random.rand() - 0.5) * abs(second_parent.mutation_chance - first_parent.mutation_chance))

        return Individual(np.array(child_permutation), child_mutation_chance)
        
    def _mutation(self, individual):
        if np.random.rand() <= individual.mutation_chance:
            a = np.random.randint(0, self._tsp.no_vertices)
            b = np.random.randint(0, self._tsp.no_vertices)
            return Individual(self.__get_swapped(individual.permutation, a, b), individual.mutation_chance)
        else:
            return individual
            
    def _elimination(self, offspring):
        combined = []
        combined.extend(self._population.individuals)
        combined.extend(offspring)
        combined = np.array(combined)
        
        selected = np.argsort(np.array(list(map(lambda individual: self._tsp.fitness(individual), combined))))[0:self._no_individuals_to_keep]
        return list(combined[selected])

    def __get_nearest_neighbour_solution(self, starting_vertex):
        nn_solution = np.empty(self._tsp.no_vertices)
        visited = []

        current_vertex = starting_vertex
        visited.append(current_vertex)
        nn_solution[0] = current_vertex
        edge_weights = np.copy(self._tsp.distance_matrix[current_vertex, :])
        edge_weights[visited] = np.Inf
        next_vertex = np.argmin(edge_weights)
        for i in range(1, self._tsp.no_vertices):
            current_vertex = next_vertex
            visited.append(current_vertex)
            nn_solution[i] = current_vertex
            edge_weights = np.copy(self._tsp.distance_matrix[current_vertex, :])
            edge_weights[visited] = np.Inf
            next_vertex = np.argmin(edge_weights)

        return Individual(nn_solution)

class TSP:
    def __init__(self, distance_matrix):
        self._distance_matrix = distance_matrix

    def fitness(self, individual):
        total_distance = 0
        for a, b in zip(individual.permutation[0:(self.no_vertices - 1)], individual.permutation[1:self.no_vertices]):
            total_distance += self._distance_matrix[int(a), int(b)]

        total_distance += self._distance_matrix[int(b), int(individual.permutation[0])]

        return total_distance

    def mean_fitness(self, individuals):
        return statistics.mean([self.fitness(individual) for individual in individuals])
    
    def best_fitness(self, individuals):
        return min([self.fitness(individual) for individual in individuals])

    def best_individual(self, individuals):
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
    def __init__(self, permutation, mutation_chance = 0.05):
        self._permutation = permutation
        self._mutation_chance = mutation_chance

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