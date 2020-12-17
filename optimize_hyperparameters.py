import multiprocessing
import os
import statistics
import timeit

import r0486848


def evaluate_combination(tsps, current_combination):
    ea = r0486848.r0486848()
    ea.set_parameters(*current_combination)
    
    return statistics.mean([ea.optimize(tsp)[0] for tsp in tsps])

if __name__ == '__main__':
    np.random.seed(0)

    no_vertices = 140
    
    tsps = [r0486848.TSP.get_random(no_vertices) for _ in range(5)]

    population_generation_schemes = [r0486848.PopulationGenerationScheme.RANDOM, r0486848.PopulationGenerationScheme.NEAREST_NEIGHBOUR_BASED]
    recombination_operators = [r0486848.RecombinationOperator.HGREX, r0486848.RecombinationOperator.PMX]
    elimination_schemes = [r0486848.EliminationScheme.LAMBDA_MU, r0486848.EliminationScheme.LAMBDAPLUSMU, r0486848.EliminationScheme.LAMBDAPLUSMU_WCROWDING]
    # no_islands = [1, 2, 4, 5, 7, 10]
    no_islands = [7, 10]
    island_swap_rates = [1, 3, 6]
    island_no_swapped_individuals = [1, 2, 4, 8]
    population_size_factors = [2, 3, 4, 5, 6]
    default_mutation_chances = [0.05, 0.10]
    mutation_chance_feedbacks = [False, True]
    mutation_chance_self_adaptivities = [False, True]

    output_file = open("r0486848_hyperparameter_optimization_results.txt", "w")

    best_combination = None
    best_combination_average = float('inf')

    output_file.write('-' * 50 + '\n')
    print('-' * 50)

    combinations = r0486848.Combinations(population_generation_schemes, recombination_operators, elimination_schemes, no_islands, island_swap_rates, island_no_swapped_individuals, population_size_factors, default_mutation_chances, mutation_chance_feedbacks, mutation_chance_self_adaptivities)
    no_combinations = len(combinations)
    moving_average_time_per_combination = 0

    for i, c in enumerate(combinations):
        start_time = timeit.default_timer()

        current_combination = c[0:7] + (int((c[6] * no_vertices ) / (4 * c[3])), int((c[6] * no_vertices ) / c[3]), int((c[6] * no_vertices ) / c[3])) + c[7:] + (0.001, 3, False)

        with multiprocessing.Pool() as p:
            res = p.apply_async(evaluate_combination, [tsps, current_combination])
            try:
                current_combination_average = res.get(1000)
            except multiprocessing.TimeoutError:
                current_combination_average = float('inf')

        output_file.write(f'Current combination =\n\t{current_combination}\nCurrent combination average =\n\t{current_combination_average}\n')
        print(f'Current combination =\n\t{current_combination}\nCurrent combination average =\n\t{current_combination_average}')
    
        output_file.write('Current combination new best? ')
        print('Current combination new best? ', end='')
        if current_combination_average < best_combination_average:
            output_file.write('YES\n')
            print('YES')
            best_combination = current_combination
            best_combination_average = current_combination_average
        else:
            output_file.write('NO\n')
            print('NO')

        output_file.flush()
        os.fsync(output_file.fileno())
    
        print(f'Hyperparameter optimization {100 * ((i + 1) / no_combinations):.2f}% finished')

        elapsed = timeit.default_timer() - start_time
    
        if i == 0:
            moving_average_time_per_combination = elapsed
        else:
            moving_average_time_per_combination *= (i / (i + 1))
            moving_average_time_per_combination += (elapsed / (i + 1))
    
        estimated_time_left = (no_combinations - (i + 1)) * moving_average_time_per_combination
        print(f'Estimated time until finished: {estimated_time_left} seconds')

    output_file.write('-' * 50 + '\n')
    print('-' * 50)
    output_file.write(f'Best combination was\n\t{current_combination}\nBest combination average was\n\t{current_combination_average}\n')
    print(f'Best combination was\n\t{current_combination}\nBest combination average was\n\t{current_combination_average}')

    output_file.close()