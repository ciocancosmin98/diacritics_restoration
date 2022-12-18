from dataclasses import dataclass
from enum import Enum
from functools import partial
import io
import math
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from collections import Counter
import tensorflow as tf
from common import constants


CharToIndexType = Dict[str, int]
IndexToCharType = Dict[int, str]


class VocabularyType(Enum):
    INPUT = 1
    TARGET = 2


@dataclass
class Samples:
    inputs: np.ndarray
    input_lens: np.ndarray
    targets: np.ndarray


@dataclass
class BatchedSamples:
    inputs: List[np.ndarray]
    input_lens: List[np.ndarray]
    targets: List[np.ndarray]


def reverse_charmap(charmap: CharToIndexType) -> IndexToCharType:
    reverse: IndexToCharType = {v: k for k, v in charmap.items()}
    return reverse


def read_dataset_files(
    inputs_filepath: str,
    targets_filepath: str,
    sentence_limit: Union[int, None]
):
    inputs: List[str] = []
    targets: List[str] = []
    inputs_reader = io.open(inputs_filepath, 'r', encoding='utf8')
    targets_reader = io.open(targets_filepath, 'r', encoding='utf8')
    for index, (input_line, target_line) in enumerate(zip(inputs_reader, targets_reader)):
        inputs.append(input_line.strip())
        targets.append(target_line.strip())
        if sentence_limit is not None and index >= sentence_limit:
            break
    inputs_reader.close()
    targets_reader.close()
    return inputs, targets


class ParalelSentencesDataset():
    '''
    This class provides methods to load and batch dataset consisting of two
    parallel files: one containing input (uncorrected) sentences and the other
    with target (corrected) sentences.

    This class has two basic usage scenarios:
        - user has only one input and one target file and wants to create
        train, validation and test set (randomly) from them. In this case,
        instantiate this class with appropriate constants (train_perc,
        valid_perc and test_perc) and call build to prepare batches.
        - user has separate train, validation and test input and target files.
        In this case, ignore (set default values) those constants and use
        add_validation_set and use_test_set to add validation and test sets.
        After that, call build to prepare batches.

    '''

    def __init__(
        self,
        batch_size: int,
        max_chars_in_sentence: int,
        input_sentences: List[str],
        target_sentences: List[str],
        input_char_vocabulary: Union[None, CharToIndexType]=None,
        target_char_vocabulary: Union[None, CharToIndexType]=None,
        take_num_top_chars: Union[None, int]=None
    ):
        '''

        :param batch_size: how many samples should each training batch have.
        :param max_chars_in_sentence: maximal amount of characters each sentence might have to be included in the dataset.
        :param input_sentences: List with input (uncorrected) sentences.
        :param target_sentences: List containing target (corrected) sentences, where i-th item of this list contains corrected sentence of i-th item of data_input_file
        :param train_perc: percentage of input_data to use for training.
        :param validation_perc: percentage of input_data to use for validation.
        :param test_perc: percentage of input_data to use for testing.
        :param input_char_vocabulary: vocabulary (dictionary in form char:id) to use for encoding characters, None if new one should be created from the data.
        :param take_num_top_chars: take only this number of most occuring characters -- all other are considered UNK
        '''

        self.batch_size = batch_size
        self.max_chars_in_sentence = max_chars_in_sentence

        self.input_sentences = input_sentences
        self.target_sentences = target_sentences

        self.input_char_vocabulary = input_char_vocabulary
        self.target_char_vocabulary = target_char_vocabulary
        self.take_num_top_chars = take_num_top_chars

        self.validation_input_sentences: Optional[List[str]] = None
        self.validation_target_sentences: Optional[List[str]] = None
        self.test_input_sentences: Optional[List[str]] = None
        self.test_target_sentences: Optional[List[str]] = None

    def add_validation_set(
        self,
        validation_input_sentences: List[str],
        validation_target_sentences: List[str]
    ):
        self.validation_input_sentences = validation_input_sentences
        self.validation_target_sentences = validation_target_sentences

    def add_test_set(
        self,
        test_input_sentences: List[str],
        test_target_sentences: List[str]
    ):
        self.test_input_sentences = test_input_sentences
        self.test_target_sentences = test_target_sentences

    def build(self):
        # first remove too long sentences and create vocabulary
        all_input_characters, all_target_characters = Counter(), Counter()
        num_input_sentences = len(self.input_sentences)
        self.input_sentences, self.target_sentences, train_input_chars, train_target_chars, num_removed = \
            self._remove_long_samples_and_build_vocab(
                self.input_sentences, self.target_sentences
            )
        print(
            '{}/{} train samples were removed due to their length.'.format(
                num_removed, num_input_sentences
            )
        )
        all_input_characters.update(train_input_chars)
        all_target_characters.update(train_target_chars)

        if self.validation_input_sentences is not None:
            val_input_chars, val_target_chars = \
                self._build_vocab(
                    input_sentences=self.validation_input_sentences,
                    target_sentences=self.validation_target_sentences
                )
            all_input_characters.update(val_input_chars)
            all_target_characters.update(val_target_chars)

        if self.test_input_sentences is not None:
            test_input_chars, test_target_chars = \
                self._build_vocab(
                    input_sentences=self.test_input_sentences,
                    target_sentences=self.test_target_sentences
                )
            all_input_characters.update(test_input_chars)
            all_target_characters.update(test_target_chars)

        def build_vocab_from_counter(characters_counter: Counter):
            characters: List[str] = list(sorted(characters_counter.keys()))

            if self.take_num_top_chars is not None:
                characters_tuples = characters_counter.most_common(self.take_num_top_chars)
                characters = map(lambda x: x[0], characters_tuples)
                characters = list(sorted(characters))

            char_vocabulary = {x: i for i, x in enumerate(characters)}
            char_vocabulary[constants.UNKNOWN_SYMBOL] = len(characters)

            return char_vocabulary

        # build vocabulary if no vocabulary is provided
        if self.input_char_vocabulary is None:
            self.input_char_vocabulary = \
                build_vocab_from_counter(all_input_characters)

        if self.target_char_vocabulary is None:
            self.target_char_vocabulary = \
                build_vocab_from_counter(all_target_characters)

        # second step is transforming sentences into sequences of IDs rather than sequences of characters
        input_data, target_data, max_decoder_word_chars = self._preprocess(self.input_sentences, self.target_sentences)
        self.input_sentences, self.target_sentences = None, None  # forget no more necessary data
        self.max_decoder_word_chars = max_decoder_word_chars  # number of characters in the longest word

        self.train_xdata, self.train_ydata = input_data, target_data
        self.train_batches = int(len(input_data) / self.batch_size)  # number of train batches prepared

        # preprocess validation data
        if self.validation_input_sentences is None:
            raise Exception('Must call "add_validation_set" before build.')
        self.validation_xdata, self.validation_ydata, valid_max_decoder_word_chars = self._preprocess(
            self.validation_input_sentences, self.validation_target_sentences)
        self.validation_input_sentences, self.validation_target_sentences = None, None  # forget no more necessary data
        self.max_decoder_word_chars = max(self.max_decoder_word_chars, valid_max_decoder_word_chars)

        # preprocess test data
        if self.test_input_sentences is None:
            raise Exception('Must call "add_test_set" before build.')
        self.test_xdata, self.test_ydata, test_max_decoder_word_chars = self._preprocess(
            self.test_input_sentences, self.test_target_sentences)
        self.test_input_sentences, self.test_target_sentences = None, None  # forget no more necessary data
        self.max_decoder_word_chars = max(self.max_decoder_word_chars, test_max_decoder_word_chars)

    def _remove_long_samples_and_build_vocab(
        self,
        input_sentences: List[str],
        target_sentences: List[str]
    ):
        data_inputs_shortened: List[str] = []
        data_targets_shortened: List[str] = []

        # remove too long (short) samples and split each sentence into words (split by space)
        num_samples_removed = 0
        input_characters = Counter()
        target_characters = Counter()
        for i in range(len(input_sentences)):
            if len(input_sentences[i]) < self.max_chars_in_sentence:
                data_inputs_shortened.append(input_sentences[i])
                data_targets_shortened.append(target_sentences[i])

                if self.input_char_vocabulary is None:
                    input_characters.update(input_sentences[i])
                if self.target_char_vocabulary is None:
                    target_characters.update(target_sentences[i])
            else:
                num_samples_removed += 1

        return (
            data_inputs_shortened,
            data_targets_shortened,
            input_characters,
            target_characters,
            num_samples_removed
        )

    def _build_vocab(
        self,
        input_sentences: List[str],
        target_sentences: List[str]
    ):
        # split each sentence into words (split by space)

        input_characters = Counter()
        target_characters = Counter()
        for i in range(len(input_sentences)):
            if self.input_char_vocabulary is None:
                input_characters.update(input_sentences[i])
            if self.target_char_vocabulary is None:
                target_characters.update(target_sentences[i])

        return (
            input_characters,
            target_characters
        )

    def sentence_to_indices(
        self,
        sentence: str,
        vocabtype: VocabularyType
    ):
        if vocabtype == VocabularyType.INPUT:
            vocabulary_map = self.input_char_vocabulary
        else:
            vocabulary_map = self.target_char_vocabulary

        if vocabulary_map is None:
            raise Exception("Input char to index map is unitialized")

        indices: List[int] = []
        for char in sentence:
            if char in vocabulary_map:
                index = vocabulary_map[char]
            else:
                index = vocabulary_map[constants.UNKNOWN_SYMBOL]
            indices.append(index)

        return indices

    def _preprocess(
        self,
        input_sentences: List[str],
        target_sentences: List[str]
    ):

        max_decoder_word_chars = 0  # maximal number of characters in a single decoder word

        # encode data with the vocabulary
        # input_data(target_data) is a list of numpy arrays, where each numpy array is a sequence od char-IDs representing one word
        input_data: List[np.ndarray] = []
        target_data: List[np.ndarray] = []
        for input_sentence, target_sentence in zip(input_sentences, target_sentences):
            if len(input_sentence) != len(target_sentence):
                raise ValueError(
                    "Input and target sentence do not have same lengths!:\n "
                    f"input: {input_sentence} \n target: {target_sentence}"
                )

            input_indices = self.sentence_to_indices(
                input_sentence, VocabularyType.INPUT
            )
            input_data.append(np.array(input_indices))

            target_indices = self.sentence_to_indices(
                target_sentence, VocabularyType.TARGET
            )
            target_data.append(np.array(target_indices))

            if len(input_sentence) + 1 > max_decoder_word_chars:
                max_decoder_word_chars = len(input_sentence) + 1

        return input_data, target_data, max_decoder_word_chars


    def load_data(
        self,
        batch_size: int,
        data_x: List[np.ndarray],
        data_y: List[np.ndarray]
    ):
        n_entries = len(data_x)

        max_sentence_len = max([
            len(data_x[sent_id])
                for sent_id in range(n_entries)
        ])

        inputs = np.zeros(
            (n_entries, max_sentence_len), np.int32
        )
        targets = np.zeros(
            (n_entries, max_sentence_len), np.int32
        )
        input_lens = np.zeros(
            (n_entries), np.int32
        )

        for sid in range(n_entries):
            input_lens[sid] = len(data_x[sid])

            for cid in range(input_lens[sid]):
                inputs[sid][cid] = data_x[sid][cid]
                targets[sid][cid] = data_y[sid][cid]

        dataset = tf.data.Dataset.from_tensor_slices((
            {
                "inputs": inputs,
                "input_lens": input_lens
            },
            {
                "labels": targets
            }
        ))
        loader = dataset.batch(batch_size=batch_size)
        return loader

    def get_loaders(self, batch_size: int):
        train_loader = self.load_data(
            batch_size=batch_size,
            data_x=self.train_xdata,
            data_y=self.train_ydata
        )
        dev_loader = self.load_data(
            batch_size=batch_size,
            data_x=self.validation_xdata,
            data_y=self.validation_ydata
        )
        return train_loader, dev_loader
