import r0123456
import os


def takeMeanfitness(elem):
  return elem[1]

if __name__ == '__main__':
  os.system('clear')
  test_data_file = 'tour29.csv'
  """
  Define the grid array for each parameter
  """
  population_size_factor = [4, 5, 6]
  k = [5]
  mu = [100, 110]
  no_individuals_to_keep = [100, 110]
  stopping_ratio = [0.001]
  tolerances = [3]
  """
  build the parameter vectors
  """
  parameter_vector = []

  for x in range(len(population_size_factor)):
    for y in range(len(k)):
      for z in range(len(mu)):
        for l in range(len(no_individuals_to_keep)):
          for m in range(len(stopping_ratio)):
            for n in range(len(tolerances)):
              parameter_vector.append((population_size_factor[x], k[y], mu[z], no_individuals_to_keep[l], stopping_ratio[m], tolerances[n]))

  print(len(parameter_vector)," Parameter vectors construction completed!")

  """
  Store the parameter vector and the mean fitness after tested by the EA instances
  """
  parameter_vector_ea_mean_fitness = []
  for p in parameter_vector:
    (current_mean_fitness, current_best_fitness) = r0123456.r0123456(p[0], p[1], p[2], p[3], p[4], p[5]).optimize(test_data_file)
    parameter_vector_ea_mean_fitness.append((p, current_mean_fitness))

  """
  Sort the parameter vector and mean fitness array according to the mean fitness value
  """
  parameter_vector_ea_mean_fitness.sort(key=takeMeanfitness)

  print("The optimal parameter vector and its mean fitness value is: ")
  for p in parameter_vector_ea_mean_fitness[0]:
    print(p)







