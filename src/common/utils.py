import io
import os
from typing import Dict
from dataset import CharToIndexType


def load_vocabulary(filename: str):
    vocabulary: CharToIndexType = dict()
    with io.open(filename, encoding='utf-8', mode='r') as reader:
        keys = reader.read().splitlines()
    for i, key in enumerate(keys):
        vocabulary[key] = i
    return vocabulary


def parse_dataset_file(filename: str):
    dataset: Dict[str, str] = dict()
    with io.open(filename, 'r', encoding='utf-8') as reader:
        for line in reader:
            key, value = line.strip().split(' ')
            dataset[key] = os.path.join(os.path.dirname(filename), value)
    # entries example:
    #   "train_targets" -> "datasets/ro/target_train.txt"
    return dataset
