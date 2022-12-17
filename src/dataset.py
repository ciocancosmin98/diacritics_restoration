from dataclasses import dataclass
from enum import Enum
from functools import partial
import io
import math
from typing import Callable, Dict, List, Tuple, Union
import numpy as np
from collections import Counter
import tensorflow as tf
from common import constants


CharToIndexType = Dict[str, int]


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
        train_perc=1.0,
        validation_perc=0.0,
        test_perc=0.0,
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

        self.train_perc = train_perc
        self.validation_perc = validation_perc
        self.test_perc = test_perc

        self.validation_input_sentences, self.validation_target_sentences = None, None
        self.test_input_sentences, test_targets = None, None

    def add_validation_set(self, validation_input_sentences, validation_target_sentences):
        self.validation_input_sentences = validation_input_sentences
        self.validation_target_sentences = validation_target_sentences

    def add_test_set(self, test_input_sentences, test_target_sentences):
        self.test_input_sentences = test_input_sentences
        self.test_target_sentences = test_target_sentences

    def build(self):
        if self.train_perc <= 0:
            print('No training samples!')
        if self.validation_perc == 0 and self.validation_input_sentences == None:
            print('No validation data!')
        if self.test_perc == 0 and self.test_input_sentences == None:
            print('No test data!')

        # first remove too long sentences and create vocabulary
        all_input_characters, all_target_characters = Counter(), Counter()
        num_input_sentences = len(self.input_sentences)
        self.input_sentences, self.target_sentences, train_input_characters, train_target_characters, num_removed = self._remove_long_samples_and_build_vocab(
            self.input_sentences, self.target_sentences)
        print('{}/{} train samples were removed due to their length.'.format(num_removed, num_input_sentences))
        all_input_characters.update(train_input_characters)
        all_target_characters.update(train_target_characters)

        if self.validation_input_sentences != None:
            self.validation_input_sentences, self.validation_target_sentences, validation_input_characters, validation_target_characters = self._build_vocab(
                self.validation_input_sentences, self.validation_target_sentences)
            all_input_characters.update(validation_input_characters)
            all_target_characters.update(validation_target_characters)

        if self.test_input_sentences != None:
            self.test_input_sentences, self.test_target_sentences, test_input_characters, test_target_characters = self._build_vocab(
                self.test_input_sentences, self.test_target_sentences)
            all_input_characters.update(test_input_characters)
            all_target_characters.update(test_target_characters)

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
            self.input_char_vocabulary = build_vocab_from_counter(all_input_characters)

        if self.target_char_vocabulary is None:
            self.target_char_vocabulary = build_vocab_from_counter(all_target_characters)

        # self.char_vocab is a dictionary {"character" : ID}
        # characters is just a list of characters (words) that appeared in the text
        # characters = self.input_char_vocabulary.keys()
        # self.input_char_vocab_size = len(characters)

        # second step is transforming sentences into sequences of IDs rather than sequences of characters
        input_data, target_data, max_decoder_word_chars = self._preprocess(self.input_sentences, self.target_sentences)
        self.input_sentences, self.target_sentences = None, None  # forget no more necessary data
        self.max_decoder_word_chars = max_decoder_word_chars  # number of characters in the longest word

        if self.validation_input_sentences != None or self.test_input_sentences != None:
            self.train_xdata, self.train_ydata = input_data, target_data
            self.train_batches = int(len(input_data) / self.batch_size)  # number of train batches prepared

            if self.validation_input_sentences != None:
                self.validation_xdata, self.validation_ydata, valid_max_decoder_word_chars = self._preprocess(
                    self.validation_input_sentences, self.validation_target_sentences)
                self.validation_input_sentences, self.validation_target_sentences = None, None  # forget no more necessary data
                self.max_decoder_word_chars = max(self.max_decoder_word_chars, valid_max_decoder_word_chars)

            if self.test_input_sentences != None:
                self.test_xdata, self.test_ydata, test_max_decoder_word_chars = self._preprocess(
                    self.test_input_sentences, self.test_target_sentences)
                self.test_input_sentences, self.test_target_sentences = None, None  # forget no more necessary data
                self.max_decoder_word_chars = max(self.max_decoder_word_chars, test_max_decoder_word_chars)
        else:
            self._split_train_data(input_data, target_data, self.train_perc, self.validation_perc, self.test_perc)

        self.reset_batch_pointer()

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
        data_inputs_shortened: List[str] = []
        data_targets_shortened: List[str] = []

        # split each sentence into words (split by space)

        input_characters = Counter()
        target_characters = Counter()
        for i in range(len(input_sentences)):
            data_inputs_shortened.append(input_sentences[i])
            data_targets_shortened.append(target_sentences[i])

            if self.input_char_vocabulary is None:
                input_characters.update(input_sentences[i])
            if self.target_char_vocabulary is None:
                target_characters.update(target_sentences[i])

        return (
            data_inputs_shortened,
            data_targets_shortened,
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

    def _split_train_data(
        self,
        input_data: List[np.ndarray],
        target_data: List[np.ndarray],
        train_percentage: float,
        valid_percentage: float
    ):
        '''
        Splits input_data into train, validation and test sets
        '''

        total_batches = int(len(input_data) / self.batch_size)

        # number of train batches prepared
        self.train_batches = int(round(
            total_batches * train_percentage
        ))
        # number of validation batches prepared
        val_batches = int(
            round(total_batches * valid_percentage)
        )
        # number of test batches prepared
        test_batches = total_batches - self.train_batches - val_batches

        # counts in means of samples
        test_samples = test_batches * self.batch_size
        val_samples = test_batches * self.batch_size
        train_samples = len(input_data) - test_samples - val_samples

        if self.train_batches == 0:
            assert False, "Not enough data. Make batch_size smaller."

        print(str(self.train_batches) + ' train batches available')
        print(str(val_batches) + ' validation batches available')
        print(str(test_batches) + ' test batches available')

        # split tensor into train, validation and test set
        self.test_xdata = input_data[:test_samples]
        self.test_ydata = target_data[:test_samples]

        self.validation_xdata = input_data[
            test_samples: test_samples + val_samples
        ]
        self.validation_ydata = target_data[
            test_samples: test_samples + val_samples
        ]

        self.train_xdata = input_data[test_samples + val_samples:]
        self.train_ydata = target_data[test_samples + val_samples:]

        assert len(self.train_xdata) == train_samples

    def get_loaders(self, batch_size: int):
        n_entries = len(self.train_xdata)

        max_sentence_len = max([
            len(self.train_xdata[sent_id])
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
            input_lens[sid] = len(self.train_xdata[sid])

            for cid in range(input_lens[sid]):
                inputs[sid][cid] = self.train_xdata[sid][cid]
                targets[sid][cid] = self.train_ydata[sid][cid]

        train_dataset = tf.data.Dataset.from_tensor_slices((
            {
                "inputs": inputs,
                "input_lens": input_lens
            },
            {
                "labels": targets
            }
        ))
        train_loader = train_dataset.batch(batch_size=batch_size)
        return train_loader

    def next_batch(self):
        '''
        Returns next train batch.  Each batch consists of three parts.
        '''

        if self.pointer + self.batch_size > len(self.train_xdata):
            raise AssertionError(
                'No more batches. Call reset_batch_pointer() before calling '
                'this method again.'
            )

        max_sentence_len_in_batch = max([
            len(self.train_xdata[sent_id]) for sent_id in range(
                self.pointer, self.pointer + self.batch_size
            )
        ])

        batch_inputs = np.zeros(
            (self.batch_size, max_sentence_len_in_batch), np.int32
        )
        batch_targets = np.zeros(
            (self.batch_size, max_sentence_len_in_batch), np.int32
        )
        batch_input_lens = np.zeros(
            (self.batch_size), np.int32
        )

        for sample_ind in range(self.batch_size):
            batch_input_lens[sample_ind] = len(self.train_xdata[self.pointer])

            for char_ind in range(batch_input_lens[sample_ind]):
                batch_inputs[sample_ind][char_ind] = \
                    self.train_xdata[self.pointer][char_ind]
                batch_targets[sample_ind][char_ind] = \
                    self.train_ydata[self.pointer][char_ind]

            self.pointer += 1

        return batch_inputs, batch_input_lens, batch_targets

    def get_samples(
        self,
        xdata: List[np.ndarray],
        ydata: List[np.ndarray],
        n_samples=-1
    ):
        '''
        Returns numpy array with first n_samples samples. If n_samples == -1,
        all test samples are returned.
        '''

        if n_samples < 0:
            n_samples = len(xdata)

        max_sentence_len = max([
            len(xdata[sid]) for sid in range(n_samples)
        ])

        inputs = np.zeros(
            (n_samples, max_sentence_len), np.int32
        )
        targets = np.zeros(
            (n_samples, max_sentence_len), np.int32
        )
        input_lens = np.zeros(
            (n_samples), np.int32
        )

        for sid in range(n_samples):
            input_lens[sid] = len(xdata[sid])

            for char_idx in range(input_lens[sid]):
                inputs[sid][char_idx] = xdata[sid][char_idx]
                targets[sid][char_idx] = ydata[sid][char_idx]

        return Samples(
            inputs=inputs,
            input_lens=input_lens,
            targets=targets
        )

    def get_evaluation_sets(self):
        eval_sets: List[Tuple[str, Callable[[], Samples]]] = [
            (
                'dev',
                partial(
                    self.get_samples,
                    xdata=self.validation_xdata,
                    ydata=self.validation_ydata
                )
            ),
            (
                'test',
                partial(
                    self.get_samples,
                    xdata=self.test_xdata,
                    ydata=self.test_ydata
                )
            )
        ]
        return eval_sets

    def get_eval_sets(self):
        evaluation_sets: Dict[str, BatchedSamples] = {}
        for evaluation_set_name, evaluation_set_fn in self.get_evaluation_sets():
            eval_bs = self.batch_size * 3
            samples = evaluation_set_fn()

            num_eval_bins = math.ceil(len(samples.inputs) / eval_bs)

            input_sentences = np.array_split(samples.inputs, num_eval_bins)
            sentence_lens = np.array_split(samples.input_lens, num_eval_bins)
            target_sentences = np.array_split(samples.targets, num_eval_bins)

            evaluation_sets[evaluation_set_name] = BatchedSamples(
                inputs=input_sentences,
                input_lens=sentence_lens,
                targets=target_sentences
            )
        return evaluation_sets

    def reset_batch_pointer(self):
        '''
        Resets batch pointer and permutes array. Call this after end of every epoch.
        '''
        permutation = np.random.permutation(len(self.train_xdata))

        # works only with np.arrays
        # self.train_xdata = self.train_xdata[permutation]
        # self.train_ydata = self.train_ydata[permutation]

        self.train_xdata = [self.train_xdata[i] for i in permutation]
        self.train_ydata = [self.train_ydata[i] for i in permutation]

        self.pointer = 0

