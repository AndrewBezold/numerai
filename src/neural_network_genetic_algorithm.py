import pygad
import pygad.kerasga
import utils
import numpy as np
from data_generator import DataGenerator

def create_model(input_size: int, output_size: int, hidden_sizes: "list[int]" = []):
    model = utils.create_neural_net(input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes)
    return model

training_data = utils.load_training_data()
validation_data = utils.load_validation_data()
feature_names = utils.get_feature_names(training_data)
target = utils.get_target(training_data)

input_size = len(feature_names)
output_size = 1
hidden_sizes = [500]
num_solutions = 20
num_generations = 10
num_parents_mating = 5
model = create_model(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)


#training_generator = DataGenerator(training_data, feature_names, target, batch_size=1000000)

def set_model_weights(model, solution):
    solution_weights = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
    model.set_weights(solution_weights)
    return model

def fitness_function(solution, solution_idx):
    # run training data through solution
    set_model_weights(model, solution)
    predictions_list = []
    for era in training_data['era'].unique():
        predictions = utils.predict_neural_net(model, training_data[training_data['era'] == era], feature_names, inplace=False, batch_size=None)
        # rank predictions, with any post-processing (feature neutralization, etc)
        predictions = predictions.rank(pct=True, method='first')
        # compare to rank of known results (using corr?  Not sure how that's done here yet)
        actual = training_data[target].rank(pct=True, method='first')
        corr = predictions.corr(actual, 'spearman')
        predictions_list.append(corr)
    # return correlation
    return np.mean(corr)

def validation_function(solution):
    # run validation data through solution, as per fitness function
    set_model_weights(model, solution)
    predictions_list = []
    for era in validation_data['era'].unique():
        predictions = utils.predict_neural_net(model, validation_data[validation_data['era'] == era], feature_names, inplace=False, batch_size=None)
        predictions = predictions.rank(pct=True, method='first')
        actual = validation_data[target].rank(pct=True, method='first')
        corr = predictions.corr(actual, 'spearman')
        predictions_list.append(corr)
    # return correlation
    return np.mean(corr)

def create_ga(model, num_solutions, num_generations, num_parents_mating) -> pygad.GA:
    keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=num_solutions)
    ga_instance = pygad.GA(num_generations=num_generations, num_parents_mating=num_parents_mating, initial_population=keras_ga.population_weights, fitness_func=fitness_function, on_generation=callback_function)
    return ga_instance

def run_ga(ga_instance: pygad.GA) -> None:
    ga_instance.run()

def callback_function(ga_instance):
    print(f"Generation {ga_instance.generations_completed}")
    best_solution = ga_instance.best_solution()
    validation_result = validation_function(best_solution[0])
    print(f"Best Solution Training Fitness: {best_solution[1]}")
    print(f"Best Solution Validation Fitness: {validation_result}")


def baseline_gann():
    ga_instance = create_ga(model, num_solutions, num_generations, num_parents_mating)
    run_ga(ga_instance)


if __name__ == '__main__':
    baseline_gann()
