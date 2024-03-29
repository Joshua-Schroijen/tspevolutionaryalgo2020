import concurrent.futures
import multiprocessing
import queue
import numpy as np
import os
import statistics
import timeit
import logging
import logging.handlers

import r0486848

def get_remaining_time_string(seconds):
    no_days = seconds // 86400
    seconds -= no_days * 86400
    no_hours = seconds // 3600
    seconds -= no_hours * 3600
    no_minutes = seconds // 60
    seconds -= no_minutes * 60
    
    return f"{no_days} days, {no_hours} hours, {no_minutes} minutes & {seconds} seconds"

def logging_process_target(logging_queue, shutdown_event):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.NOTSET)
    file_handler = logging.FileHandler(f"compare_parameters.log", mode="w")
    stream_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    while not shutdown_event.is_set():
        try:
            logger.handle(logging_queue.get(block=True, timeout=1))
        except queue.Empty:
            continue
        
    file_handler.close()
    stream_handler.close()

def evaluate_combination_init(lq):
    global logging_queue
    global logger
    logging_queue = lq
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.handlers.QueueHandler(logging_queue))

def evaluate_combination(tsps, current_combination):
    logger.info(f'Running EA in process with PID {os.getpid()}')
    
    ea = r0486848.r0486848()
    ea.set_parameters(*current_combination)
    
    return statistics.mean([ea.optimize(tsp)[0] for tsp in tsps])

if __name__ == '__main__':
    shutdown_event = multiprocessing.Event()
    logging_queue = multiprocessing.Queue()
    logging_process = multiprocessing.Process(target=logging_process_target, args=(logging_queue, shutdown_event))
    logging_process.start()
    output_file = open("r0486848_parameter_comparison_results.txt", "w")

    try:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.handlers.QueueHandler(logging_queue))

        logger.info(f'Starting parameter comparison. PID = {os.getpid()}')
        np.random.seed(0)

        no_vertices = 30
    
        tsps = [r0486848.TSP.get_random(no_vertices) for _ in range(10)]

        population_generation_settings = [r0486848.PopulationGenerationSettings(no_vertices * 2, 0, 0), r0486848.PopulationGenerationSettings(no_vertices * 3, 0, 0), r0486848.PopulationGenerationSettings(no_vertices * 4, 0, 0), r0486848.PopulationGenerationSettings(no_vertices * 5, 0, 0), r0486848.PopulationGenerationSettings(no_vertices * 6, 0, 0), r0486848.PopulationGenerationSettings(0, no_vertices, 2), r0486848.PopulationGenerationSettings(0, no_vertices, 3), r0486848.PopulationGenerationSettings(0, no_vertices, 4), r0486848.PopulationGenerationSettings(0, no_vertices, 5), r0486848.PopulationGenerationSettings(0, no_vertices, 6)]
        recombination_operators = [r0486848.RecombinationOperator.HGREX, r0486848.RecombinationOperator.PMX]
        elimination_schemes = [r0486848.EliminationScheme.LAMBDA_MU, r0486848.EliminationScheme.LAMBDAPLUSMU, r0486848.EliminationScheme.LAMBDAPLUSMU_WCROWDING]
        no_islands = [7, 10]
        island_swap_rates = [1, 3, 6]
        mutation_chance_self_adaptivities = [False, True]

        combination_average_deltas = []

        island_no_swapped_individuals = [1, 2, 4, 8]
        default_mutation_chances = [0.05, 0.10]
        mutation_chance_feedbacks = [False, True]
        output_file.write('-' * 50 + '\n')
        logger.info('-' * 50)

        combinations = {i : c for i, c in enumerate(r0486848.Combinations(population_generation_settings, recombination_operators, elimination_schemes, no_islands, island_swap_rates, island_no_swapped_individuals, default_mutation_chances, mutation_chance_feedbacks, mutation_chance_self_adaptivities))}
        no_combinations = 2
        moving_average_time_per_combination = 0

        for i in range(no_combinations):
            start_time = timeit.default_timer()

            combination_choice_key = np.random.choice(list(combinations.keys()), 1)[0]
            chosen_combination = combinations[combination_choice_key]
            current_combination_original = chosen_combination[0:6] + (int(chosen_combination[0].total_population_size / (4 * chosen_combination[3])), int(chosen_combination[0].total_population_size / chosen_combination[3]), int(chosen_combination[0].total_population_size / chosen_combination[3])) + chosen_combination[6:] + (0.001, 3)

            print(chosen_combination)
            print(f"{chosen_combination[0].total_population_size} / (4 * {chosen_combination[3]}))")          
            del combinations[combination_choice_key]
        
            # Set up the comparison
            current_combination_with = (r0486848.PopulationGenerationSettings(no_vertices * 2, 0, 0), ) + current_combination_original[1:]
            current_combination_without = (r0486848.PopulationGenerationSettings(no_vertices * 3, 0, 0), ) + current_combination_original[1:]

            with concurrent.futures.ProcessPoolExecutor(initializer=evaluate_combination_init, initargs=(logging_queue,)) as p:
                res_with = p.submit(evaluate_combination, tsps, current_combination_with)
                res_without = p.submit(evaluate_combination, tsps, current_combination_without)
                try:
                    current_combination_average_delta = res_with.result(1000) - res_without.result(1000)
                except (concurrent.futures.TimeoutError, ArithmeticError):
                    current_combination_average_delta = 0
        
            combination_average_deltas.append(current_combination_average_delta)
       
            output_file.write(f'Current combination average delta =\n\t{current_combination_average_delta}\n')
            logger.info(f'Current combination average delta =\n\t{current_combination_average_delta}')

            output_file.flush()
            os.fsync(output_file.fileno())
    
            logger.info(f'Parameter comparison {100 * ((i + 1) / no_combinations):.2f}% finished')

            elapsed = timeit.default_timer() - start_time
    
            if i == 0:
                moving_average_time_per_combination = elapsed
            else:
                moving_average_time_per_combination *= (i / (i + 1))
                moving_average_time_per_combination += (elapsed / (i + 1))
    
            estimated_time_left = (no_combinations - (i + 1)) * moving_average_time_per_combination
            logger.info(f'Estimated time until finished: {get_remaining_time_string(estimated_time_left)}')

        output_file.write('-' * 50 + '\n')
        logger.info('-' * 50)
    
        mean_combination_average_delta = statistics.mean(combination_average_deltas)
    
        output_file.write(f'Mean combination average delta was\n\t{mean_combination_average_delta}\n')
        logger.info(f'Mean combination average delta was\n\t{mean_combination_average_delta}')

    finally:
        shutdown_event.set()
        logging_process.join()
        logging_process.terminate()
        logging_process.close()
        while True:
            try:
                logging_queue.get(block=False)
            except queue.Empty:
                break
        logging_queue.close()
        logging_queue.join_thread()
        output_file.close()