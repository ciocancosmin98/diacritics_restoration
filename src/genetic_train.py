from copy import deepcopy
from functools import partial
import hashlib
import math
from types import SimpleNamespace
from typing import Any, Dict, List, OrderedDict, Tuple
import numpy as np
import pygad
from genetic_parameters import (
    FloatGeneType,
    GeneticParameters,
    BoolGeneType,
    IntegerGeneType
)
from train import main, parse_args
from config import Config, BiLSTMConfig
from pandas.core.util.hashing import hash_array


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# transforms = Compose([
#     ToTensor(),
#     CenterCrop(160),
#     Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
# ])
# data_dir = setup_imagenette()
# training_data = ImagenetteDataset(
#     root=data_dir,
#     subset='train',
#     transforms=transforms
# )
# test_data = ImagenetteDataset(
#     root=data_dir,
#     subset='val',
#     transforms=transforms
# )

activations = [None, 'relu', 'tanh', 'sigmoid']
anneal_strats = ['cos', 'linear']
already_computed: Dict[str, float] = {}
spent_effort = 0


def get_hash(solution: np.ndarray, *args) -> str:
    integer_array = hash_array(solution)
    hash_ = hashlib.md5(integer_array.tobytes())
    for arg in args:
        hash_.update(bytes(arg))
    return hash_.hexdigest()


def get_sorted_solutions(ga_instance: pygad.GA, best_first=True):
    fitness_arr = ga_instance.last_generation_fitness
    if fitness_arr is None or len(fitness_arr) == 0:
        raise Exception(
            'Fitness for the last population was not yet computed.'
        )

    population: np.ndarray = ga_instance.population
    pop_size = population.shape[0]

    solutions: List[Tuple[np.ndarray, float]] = []

    for sol_id in range(pop_size):
        solution = population[sol_id]
        fitness = fitness_arr[sol_id]
        solutions.append((solution, fitness))

    solutions = sorted(
        solutions,
        key=lambda x: x[1],
        reverse=best_first
    )

    return solutions

def initialize_config(
    config: OrderedDict[str, Any], config_path: str, n_epochs: int
):
    config_ = Config(config_path, BiLSTMConfig)

    config_.model_config.char_embedding_dim = config['char_embedding_dim']
    config_.model_config.rnn_cell_dim = config['rnn_cell_dim']
    config_.model_config.rnn_n_layers = config['rnn_n_layers']
    config_.model_config.dropout = config['dropout']
    config_.model_config.use_residual = config['use_residual']

    config_.learning_config.dataset_config.take_num_top_chars = \
        config['take_num_top_chars']

    config_.learning_config.optimizer_config.learning_rate = \
        config['learning_rate']
    config_.learning_config.optimizer_config.beta_1 = \
        config['beta_1']
    config_.learning_config.optimizer_config.beta_2 = \
        config['beta_2']
    config_.learning_config.optimizer_config.epsilon = \
        config['epsilon']

    config_.learning_config.running_config.batch_size = \
        config['batch_size']

    config_.learning_config.running_config.num_epochs = n_epochs

    return config_

def fitness_func(
    solution: np.ndarray,
    solution_idx: np.ndarray,
    gp: GeneticParameters,
    n_epochs: int
):
    sol_hash = get_hash(solution, n_epochs)
    fitness = already_computed.get(sol_hash, None)
    if fitness is not None:
        return fitness

    global spent_effort
    spent_effort += n_epochs

    args = parse_args([
        '../data/sample_dataset_config.txt',
        '--config', '../configs/bilstm.yml',
        '--debug'
    ])

    config = gp.from_solution(solution)
    config = initialize_config(config, args.config, n_epochs)

    best_acc = main(
        args=args,
        config=config
    )

    # use f(x) = x / (1 - x) as a fitness function with the best accuracy
    # as the parameter x; it starts at 0 for 0% accuracy and a small increase
    # in accuracy leads to a quite large increase in fitness
    fitness = best_acc / (1 - best_acc)

    already_computed[sol_hash] = fitness

    return fitness

def make_fitness_func(gp: GeneticParameters, n_epochs: int):
    fitness = partial(fitness_func, gp=gp, n_epochs=n_epochs)
    fitness.__code__ = SimpleNamespace()
    fitness.__code__.co_argcount = 2
    return fitness


def callback_gen(ga_instance: pygad.GA, gp: GeneticParameters):
    print(
        'BEST FITNESS CB:',
        ga_instance.best_solution()[1]
    )
    print(
        'BEST SOLUTION CB:',
        gp.from_solution(ga_instance.best_solution()[0])
    )


class GeneticModel:
    def __init__(self):
        self.gp = GeneticParameters()
        self.gp.register_gene(
            'char_embedding_dim', IntegerGeneType([64, 128, 256])
        )
        self.gp.register_gene(
            'rnn_cell_dim', IntegerGeneType([64, 128, 256])
        )
        self.gp.register_gene(
            'rnn_n_layers', IntegerGeneType([1])
        )
        self.gp.register_gene(
            'dropout', FloatGeneType(gene_space=[
                0.0, 0.1, 0.3, 0.5
            ])
        )
        self.gp.register_gene(
            'use_residual', BoolGeneType()
        )
        self.gp.register_gene(
            'take_num_top_chars', IntegerGeneType([25, 50, 100])
        )
        self.gp.register_gene(
            'learning_rate', FloatGeneType(gene_space=[
                1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5
            ])
        )
        self.gp.register_gene(
            'beta_1', FloatGeneType(gene_space=[
                0.85, 0.9, 0.95
            ])
        )
        self.gp.register_gene(
            'beta_2', FloatGeneType(gene_space=[
                0.99, 0.999, 0.9999
            ])
        )
        self.gp.register_gene(
            'epsilon', FloatGeneType(gene_space=[
                1e-6, 1e-7, 1e-8
            ])
        )
        self.gp.register_gene(
            'batch_size', IntegerGeneType([128])
        )

    def run_genetic_algorithm(
        self,
        stages: List[int],
        max_effort: int,
        generations_per_stage: int
    ):
        gene_types, gene_spaces = self.gp.training_parameters()

        cb_gen = partial(callback_gen, gp=self.gp)
        cb_gen.__code__ = SimpleNamespace()
        cb_gen.__code__.co_argcount = 1

        last_solutions = None
        next_population = None
        for stage in stages:
            n_epochs = stage
            pop_size = math.ceil(max_effort / n_epochs)
            if pop_size <= 1:
                raise Exception(
                    f'Stage "{stage}": Population too small ({pop_size})'
                )

            if last_solutions is not None:
                next_population = np.array(
                    [solution[0] for solution in last_solutions[:pop_size]],
                    dtype=object
                )
                kwargs = {'initial_population': next_population}
            else:
                kwargs = {'sol_per_pop': pop_size}

            ga_instance = pygad.GA(
                num_generations=generations_per_stage,
                num_parents_mating=2,
                num_genes=len(gene_types),
                on_generation=cb_gen,
                fitness_func=make_fitness_func(self.gp, n_epochs=n_epochs),
                gene_type=deepcopy(gene_types),
                gene_space=deepcopy(gene_spaces),
                random_seed=666013,
                keep_elitism=1,
                crossover_type="scattered",
                **kwargs
            )

            ga_instance.run()
            last_solutions = get_sorted_solutions(ga_instance, best_first=True)

        print(
            'BEST FITNESS:',
            ga_instance.best_solution()[1]
        )
        print(
            'BEST SOLUTION:',
            self.gp.from_solution(ga_instance.best_solution()[0])
        )


gm = GeneticModel()
gm.run_genetic_algorithm(
    stages=[1, 4], #, 2, 4, 8],
    max_effort=20,
    generations_per_stage=3,
)
print('Effort spent:', spent_effort)
