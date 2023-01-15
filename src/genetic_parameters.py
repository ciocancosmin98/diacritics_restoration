import pygad
import numpy as np
import numpy
from dataclasses import dataclass
from typing import TypeVar, Generic, Dict, Union, List
from collections import OrderedDict


T = TypeVar('T')


@dataclass
class LinSpace(Generic[T]):
    low: T
    high: T
    step: Union[None, T] = None


GeneSpaceType = Union[List[T], LinSpace[T]]


class GeneType(Generic[T]):
    def __init__(
        self,
        gene_space: GeneSpaceType[T],
        dtype: T
    ):
        self.gene_space = gene_space
        self.dtype = dtype

        assert self.dtype in (int, float)
        if isinstance(gene_space, list):
            assert all(isinstance(x, self.dtype) for x in gene_space)
        elif isinstance (gene_space, LinSpace):
            assert isinstance(gene_space.high, self.dtype)
            assert isinstance(gene_space.low, self.dtype)
            step = gene_space.step
            assert step is None or isinstance(step, self.dtype)
        else:
            raise ValueError('Wrong type for "gene_space".')

    def get_type(self):
        return self.dtype

    def get_space(self):
        if isinstance(self.gene_space, list):
            return self.gene_space
        elif isinstance (self.gene_space, LinSpace):
            range = {
                'high': self.gene_space.high,
                'low': self.gene_space.low
            }
            if self.gene_space.step is not None:
                range['step'] = self.gene_space.step
            return range
        else:
            raise ValueError('Wrong type for "gene_space".')


class BoolGeneType(GeneType[int]):
    def __init__(self):
        super().__init__(
            gene_space=[0, 1],
            dtype=int
        )


class IntegerGeneType(GeneType[int]):
    def __init__(self, gene_space: GeneSpaceType[int]):
        super().__init__(
            gene_space=gene_space,
            dtype=int
        )


class FloatGeneType(GeneType[float]):
    def __init__(self, gene_space: GeneSpaceType[float]):
        super().__init__(
            gene_space=gene_space,
            dtype=float
        )


class GeneticParameters:
    def __init__(self):
        self.genes: OrderedDict[str, GeneType] = OrderedDict()

    def register_gene(self, name: str, gene: GeneType):
        self.genes[name] = gene

    def training_parameters(self):
        gene_types = []
        gene_spaces = []

        for gene in self.genes.values():
            gene_types.append(gene.get_type())
            gene_spaces.append(gene.get_space())

        return gene_types, gene_spaces

    def from_solution(self, solution: np.ndarray) -> "OrderedDict[str, any]":
        solution_dict: OrderedDict[str, any] = OrderedDict()
        for index, name in enumerate(self.genes):
            solution_dict[name] = solution[index]
        return solution_dict