#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import io
import logging
import math
import os
import sys
import time
from typing import Dict, List, Union
import tensorflow as tf
from glob import glob
import argparse
from common import constants

import numpy as np
import pickle
from dataclasses import dataclass

from dataset import (
    BatchedSamples,
    CharToIndexType,
    ParalelSentencesDataset,
    reverse_charmap,
    read_dataset_files
)
from common import utils
from config import Config, BiLSTMConfig
from train import load_vocabularies, add_dev_and_test_sets
from models.bilstm import BiLSTM


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input", required=True, type=str,
        help="File path of the text to be restored."
    )
    parser.add_argument(
        "--output", required=True, type=str,
        help="File path of the restored text."
    )
    parser.add_argument(
        "--config", required=True, type=str,
        help="Path to configuration file."
    )
    parser.add_argument(
        "--session_dir", required=True, type=str,
        help="Path to the directory where the session results are stored such as the vocabulary."
    )
    parser.add_argument(
        "--model_path", required=True, type=str,
        help="Path to .h5 saved model file."
    )
    parser.add_argument(
        "--max_entries", default=-1, type=int,
        help="Maximum number of entries to predict on. Set negative or zero to ignore."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Runs the training steps eagerly, allowing for easier debugging."
    )

    args = parser.parse_args()

    if args.max_entries <= 0:
        args.max_entries = None

    return args


def setup_session(args: argparse.Namespace):
    # Set random seed
    np.random.seed(42)

    config = Config(args.config, BiLSTMConfig)

    return config


@dataclass
class RebuildLine:
    # indices to rebuild the line
    indices: List[int]

def split_long_lines(lines: List[str], max_len: int):
    split_lines: List[str] = []
    rebuild_lines: List[RebuildLine] = []
    idx = 0
    for line in lines:
        if len(line) >= max_len:
            words = line.split(' ')
            new_line: List[str] = []
            line_idxs: List[int] = []
            new_size = 0

            def build_line(new_line, split_lines, line_idxs, idx):
                split_lines.append(' '.join(new_line))
                line_idxs.append(idx)
                idx += 1
                return [], 0, idx

            for word in words:
                if new_size + len(word) < max_len:
                    new_line.append(word)
                    new_size += len(word)
                else:
                    new_line, new_size, idx = build_line(new_line, split_lines, line_idxs, idx)
                    new_line.append(word)
                    new_size += len(word)

                if new_size >= max_len // 2:
                    new_line, new_size, idx = build_line(new_line, split_lines, line_idxs, idx)

            if len(new_line) != 0:
                new_line, new_size, idx = build_line(new_line, split_lines, line_idxs, idx)

            rebuild_lines.append(RebuildLine(line_idxs))
        else:
            split_lines.append(line)
            rebuild_lines.append(RebuildLine([idx]))
            idx += 1

    return split_lines, rebuild_lines

def merge_lines(predicted_lines: List[str], rebuild_lines: List[RebuildLine]):
    merged_lines: List[str] = []
    for rebuild in rebuild_lines:
        lines: List[str] = []
        for idx in rebuild.indices:
            lines.append(predicted_lines[idx])
        merged_lines.append(' '.join(lines))
    return merged_lines

def main(args: argparse.Namespace):
    config = setup_session(args)

    input_sentences: List[str] = []
    with open(args.input, 'r', encoding='utf-8') as f:
        input_sentences.extend([line.strip() for line in f.readlines()])

    split_sentences, rebuild_sentences = split_long_lines(
        lines=input_sentences,
        max_len=config.learning_config.dataset_config.max_chars_in_sentence
    )

    input_char_vocab, target_char_vocab = load_vocabularies(
        input_vocab_path=None,
        target_vocab_path=None,
        checkpoint_path=args.session_dir
    )

    assert input_char_vocab is not None
    assert target_char_vocab is not None

    batch_size = config.learning_config.running_config.batch_size
    batch_size = 64

    dataset = ParalelSentencesDataset(
        batch_size=batch_size,
        max_chars_in_sentence=config.learning_config.dataset_config.max_chars_in_sentence,
        input_sentences=split_sentences,
        target_sentences=split_sentences,
        input_char_vocabulary=input_char_vocab,
        target_char_vocabulary=target_char_vocab,
        take_num_top_chars=config.learning_config.dataset_config.take_num_top_chars
    )
    dataset.add_validation_set(split_sentences, split_sentences)
    dataset.add_test_set(split_sentences, split_sentences)

    print('Building dataset')
    dataset.build()

    model = BiLSTM(
        lstm_units=config.model_config.rnn_cell_dim,
        num_rnns=config.model_config.rnn_n_layers,
        input_alphabet_size=len(input_char_vocab.keys()),
        target_alphabet_size=len(target_char_vocab.keys()),
        embedding_dim=config.model_config.char_embedding_dim,
        use_residual=config.model_config.use_residual,
        dropout=config.model_config.dropout
    )

    model.make(batch_size)
    model.summary(line_length=80)
    model.load_weights(args.model_path)

    _, dev_loader = dataset.get_loaders(
        batch_size=batch_size,
        max_entries=args.max_entries
    )

    results = model.predict(
        dev_loader,
        batch_size=batch_size,
    )

    predictions = results['predictions']
    mask = results['mask']


    # extract indices based on the masks which tell us the lengths of the
    # original sentences
    predictions_masked: List[List[int]] = []
    for sentence_id in range(len(predictions)):
        predictions_masked.append(
            list(predictions[sentence_id][mask[sentence_id]])
        )

    # map indices back to characters
    index_map = reverse_charmap(target_char_vocab)
    unk_id = target_char_vocab[constants.UNKNOWN_SYMBOL]
    predicted_sentences: List[str] = []
    for sentence_id in range(len(predictions_masked)):
        indices = predictions_masked[sentence_id]
        sentence = ''
        for char_id in range(len(indices)):
            if indices[char_id] != unk_id:
                sentence += index_map[indices[char_id]]
            else:
                sentence += split_sentences[sentence_id][char_id]
        predicted_sentences.append(sentence)

    merged_lines = merge_lines(predicted_sentences, rebuild_sentences)

    # outputs_dir = os.path.abspath(os.path.join(args.session_dir, 'outputs'))
    # os.makedirs(outputs_dir, exist_ok=True)
    # predicted_dev_path = os.path.join(outputs_dir, 'predictions_dev.txt')
    with open(args.output, 'w', encoding='utf-8') as f:
        for sentence in merged_lines:
            f.write(sentence)
            f.write('\n')

    print(f'Results are available at "{args.output}".')

if __name__ == "__main__":
    main(parse_args())