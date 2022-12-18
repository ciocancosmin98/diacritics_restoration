import os
from typing import Dict, List, Union, Any, TypeVar, Type, Generic
import yaml
import re
import tensorflow as tf


T = TypeVar("T")


def load_yaml(
    path: str,
) -> dict:
    # Fix yaml numbers https://stackoverflow.com/a/30462009/11037553
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u"tag:yaml.org,2002:float",
        re.compile(
            u"""^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list(u"-+0123456789."),
    )
    with open(path, "r", encoding="utf-8") as file:
        return yaml.load(file, Loader=loader)


def get_from_config(config: Dict[str, Any], key: str, dtype: Type[T]):
    value = config.get(key, None)
    if value is None:
        raise Exception(f'Key "{key}" does not exist in config {config}.')
    if not isinstance(value, dtype):
        raise Exception(
            f'Value of key "{key}" is of wrong type: '
            f'Expected "{dtype}", got "{type(value)}".'
        )
    return value


def preprocess_paths(
    paths: Union[List[str], str],
    isdir: bool = False,
) -> Union[List[str], str]:
    """Expand the path to the root "/" and makedirs
    Args:
        paths (Union[List, str]): A path or list of paths
    Returns:
        Union[List, str]: A processed path or list of paths, return None if it's not path
    """
    if isinstance(paths, list):
        paths = [
            os.path.abspath(os.path.expanduser(path))
            for path in paths
        ]
        for path in paths:
            dirpath = path if isdir else os.path.dirname(path)
            if not tf.io.gfile.exists(dirpath):
                tf.io.gfile.makedirs(dirpath)
        return paths
    if isinstance(paths, str):
        paths = os.path.abspath(os.path.expanduser(paths))
        dirpath = paths if isdir else os.path.dirname(paths)
        if not tf.io.gfile.exists(dirpath):
            tf.io.gfile.makedirs(dirpath)
        return paths
    return None


class DatasetConfig:
    def __init__(self, config: dict):
        self.stage = config.pop("stage", None)
        self.data_paths = preprocess_paths(config.pop("data_paths", None))
        self.tfrecords_dir = preprocess_paths(config.pop("tfrecords_dir", None), isdir=True)
        self.tfrecords_shards = config.pop("tfrecords_shards", 16)
        self.shuffle = config.pop("shuffle", False)
        self.cache = config.pop("cache", False)
        self.drop_remainder = config.pop("drop_remainder", True)
        self.buffer_size = config.pop("buffer_size", 100)
        self.use_tf = config.pop("use_tf", False)
        # self.augmentations = Augmentation(config.pop("augmentation_config", {}))
        for k, v in config.items():
            setattr(self, k, v)


class RunningConfig:
    def __init__(self, config: dict):
        self.batch_size = get_from_config(config, "batch_size", int)
        self.num_epochs = get_from_config(config, "num_epochs", int)
        # for k, v in config.items():
        #     setattr(self, k, v)
        #     if k == "checkpoint":
        #         if v and v.get("filepath"):
        #             preprocess_paths(v.get("filepath"))
        #     elif k == "states_dir" and v:
        #         preprocess_paths(v)
        #     elif k == "tensorboard":
        #         if v and v.get("log_dir"):
        #             preprocess_paths(v.get("log_dir"))


class OptimizerConfig:
    def __init__(self, config: dict):
        self.learning_rate = get_from_config(config, "learning_rate", float)
        self.beta_1 = get_from_config(config, "beta_1", float)
        self.beta_2 = get_from_config(config, "beta_2", float)
        self.epsilon = get_from_config(config, "epsilon", float)

class LearningConfig:
    def __init__(self, config: dict):
        self.running_config = RunningConfig(config.pop("running_config", {}))
        self.optimizer_config = OptimizerConfig(config.pop("optimizer_config", {}))
        # for k, v in config.items():
        #     setattr(self, k, v)

class BiLSTMConfig:
    def __init__(self, config: dict):
        self.char_embedding_dim = \
            get_from_config(config, "char_embedding_dim", int)
        self.rnn_cell_dim = get_from_config(config, "rnn_cell_dim", int)
        self.rnn_n_layers = get_from_config(config, "rnn_n_layers", int)
        self.dropout = get_from_config(config, "dropout", float)
        self.use_residual = get_from_config(config, "use_residual", bool)


class Config(Generic[T]):
    """User config class for training, testing or infering"""

    def __init__(self, data: Union[str, dict], model_type: Type[T]):
        config = data if isinstance(data, dict) else load_yaml(preprocess_paths(data))
        # self.speech_config = config.pop("speech_config", {})
        # self.decoder_config = config.pop("decoder_config", {})
        # self.model_config = config.pop("model_config", {})
        self.model_config: T = model_type(config.pop("model_config", {}))
        self.learning_config = LearningConfig(config.pop("learning_config", {}))
        for k, v in config.items():
            setattr(self, k, v)