#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import io
import logging
import math
import os
import sys
import time
from typing import Dict, Union
import tensorflow as tf
from glob import glob
import argparse

import numpy as np
import pickle

from dataset import (
    BatchedSamples,
    CharToIndexType,
    ParalelSentencesDataset,
    read_dataset_files
)
from common import utils
from models.bilstm import BiLSTM


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "dataset", default='', type=str,
        help="Path to dataset configuration file storing files for train, dev and test."
    )
    parser.add_argument(
        "--exp_name", default='', type=str,
        help="Experiment name."
    )
    parser.add_argument(
        "--batch_size", default=128, type=int,
        help="Batch size."
    )
    parser.add_argument(
        "--embedding", default=256, type=int,
        help="Embedding dimension. One hot is used if <1. It is highly recommended that embedding == rnn_cell_dim"
    )
    parser.add_argument(
        "--input_char_vocab", default=None, type=str,
        help="Path to file storing input char vocabulary. If no provided, is automatically computed from data."
    )
    parser.add_argument(
        "--target_char_vocab", default=None, type=str,
        help="Path to file storing target char vocabulary. If no provided, is automatically computed from data."
    )
    parser.add_argument(
        "--num_top_chars", default=None, type=int,
        help="Take only num_top_chars most occuring characters. All other will be considered UNK"
    )

    parser.add_argument(
        "--max_chars", default=200, type=int,
        help="Maximum number of characters in a sentence."
    )
    parser.add_argument(
        "--epochs", default=10, type=int,
        help="Number of epochs."
    )
    parser.add_argument(
        "--logdir", default="logs", type=str,
        help="Logdir name."
    )
    parser.add_argument(
        "--savedir", default="save", type=str,
        help="Savedir name."
    )

    parser.add_argument(
        "--train_perc", default=1.0, type=float,
        help="Percentage of total samples used for training."
    )
    parser.add_argument(
        "--validation_perc", default=0.0, type=float,
        help="Percentage of total samples used for validation set."
    )
    parser.add_argument(
        "--test_perc", default=0.0, type=float,
        help="Percentage of total samples used for testing."
    )

    parser.add_argument(
        "--keep_prob", default=0.8, type=float,
        help="Dropout keep probability used for training."
    )

    parser.add_argument(
        "--rnn_cell", default="gru", type=str,
        help="RNN cell type."
    )
    parser.add_argument(
        "--rnn_cell_dim", default=240, type=int,
        help="RNN cell dimension."
    )
    parser.add_argument(
        "--num_layers", default=1, type=int,
        help="Number of layers."
    )
    parser.add_argument(
        "--learning_rate", default=1e-4, type=float,
        help="Learning rate."
    )
    parser.add_argument(
        "--threads", default=8, type=int,
        help="Maximum number of threads to use."
    )

    parser.add_argument(
        '--use_residual', action='store_true', default=False,
        help="If set, residual connections will be used in the model."
    )

    parser.add_argument(
        "--save_every", default=2000, type=int,
        help="Interval for saving models."
    )
    parser.add_argument(
        "--log_every", default=1000, type=int,
        help="Interval for logging models (Tensorboard)."
    )
    parser.add_argument(
        "--num_test", default=1000, type=int,
        help="Number of samples to test on."
    )

    parser.add_argument(
        "--restore", type=str,
        help="Restore model from this checkpoint and continue training from it. Can be a shell-style wildcard expandable exactly to one directory."
    )

    parser.add_argument(
        "--num_sentences", default=None, type=int,
        help="Number of sentences to read from train file (-1 == read all sentences)."
    )

    parser.add_argument(
        "--debug", action="store_true",
        help="Runs the training steps eagerly, allowing for easier debugging."
    )

    args = parser.parse_args()
    return args


def load_vocabularies(
    input_vocab_path: Union[str, None],
    target_vocab_path: Union[str, None],
    checkpoint_path: Union[str, None]
):
    input_char_vocab: Union[CharToIndexType, None] = None
    target_char_vocab: Union[CharToIndexType, None] = None

    if input_vocab_path is not None and os.path.exists(input_vocab_path):
        input_char_vocab = utils.load_vocabulary(input_vocab_path)
    if target_vocab_path is not None and os.path.exists(target_vocab_path):
        target_char_vocab = utils.load_vocabulary(target_vocab_path)

    if checkpoint_path is not None:
        checkpoint_paths = glob(checkpoint_path)  # expand possible wildcard

        if len(checkpoint_paths) == 0:
            raise ValueError(
                f'Restore parameter provided ({checkpoint_path}), '
                'but no such folder exists.'
            )
        elif len(checkpoint_paths) > 1:
            raise ValueError(
                f'Restore parameter provided ({checkpoint_path}), '
                'but multiple such folders exist.'
            )

        checkpoint_path = checkpoint_paths[0]

        with open(os.path.join(checkpoint_path, 'vocab.pkl'), 'rb') as f:
            input_char_vocab, target_char_vocab = pickle.load(f)

    return input_char_vocab, target_char_vocab


def setup_session(args: argparse.Namespace):
    # Set random seed
    np.random.seed(42)

    experiment_name = args.exp_name
    experiment_name += '_layers{}_dim{}_embedding{}_lr{}'.format(
        args.num_layers, args.rnn_cell_dim, args.embedding, args.learning_rate
    )

    # create save directory for current experiment's data (if not exists)
    save_data_dir = os.path.join(args.savedir, experiment_name)
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    # create subdir of save data directory to store trained models
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    save_model_dir = os.path.join(save_data_dir, timestamp)
    os.makedirs(save_model_dir)

    # configure logger
    logging.basicConfig(
        filename=os.path.join(save_model_dir, 'experiment_log.log'),
        level=logging.DEBUG,
        format='%(asctime)s %(message)s'
    )
    logging.info('Experiment started at: {} and its name: {}'.format(
        timestamp, experiment_name
    ))
    logging.info('Experiment arguments: {}'.format(str(args)))

    # summary_writer = tf.compat.v1.summary.FileWriter(
    #     f"{args.logdir}/{timestamp}-{experiment_name}",
    #     flush_secs=10
    # )
    summary_writer = None

    return save_model_dir, summary_writer


def add_dev_and_test_sets(
    dataset: ParalelSentencesDataset,
    ds_fpaths: Dict[str, str]
):
    if 'dev_inputs' in ds_fpaths:
        print('Loading validation data')
        dev_input_sentences, dev_target_sentences = read_dataset_files(
            inputs_filepath=ds_fpaths['dev_inputs'],
            targets_filepath=ds_fpaths['dev_targets'],
            sentence_limit=None
        )
        dataset.add_validation_set(dev_input_sentences, dev_target_sentences)

    if 'test_inputs' in ds_fpaths:
        print('Loading test data')
        test_input_sentences, test_target_sentences = read_dataset_files(
            inputs_filepath=ds_fpaths['dev_inputs'],
            targets_filepath=ds_fpaths['dev_targets'],
            sentence_limit=None
        )
        dataset.add_test_set(test_input_sentences, test_target_sentences)


def main(args: argparse.Namespace):
    save_model_dir, summary_writer = setup_session(args)

    ds_fpaths = utils.parse_dataset_file(args.dataset)
    print(ds_fpaths, '\nLoading train data')

    input_sentences, target_sentences = read_dataset_files(
        inputs_filepath=ds_fpaths['train_inputs'],
        targets_filepath=ds_fpaths['train_targets'],
        sentence_limit=args.num_sentences
    )

    input_char_vocab, target_char_vocab = load_vocabularies(
        input_vocab_path=args.input_char_vocab,
        target_vocab_path=args.target_char_vocab,
        checkpoint_path=args.restore
    )

    dataset = ParalelSentencesDataset(
        batch_size=args.batch_size,
        max_chars_in_sentence=args.max_chars,
        input_sentences=input_sentences,
        target_sentences=target_sentences,
        train_perc=args.train_perc,
        validation_perc=args.validation_perc,
        test_perc=args.test_perc,
        input_char_vocabulary=input_char_vocab,
        target_char_vocabulary=target_char_vocab,
        take_num_top_chars=args.num_top_chars
    )
    add_dev_and_test_sets(dataset, ds_fpaths)

    print('Building dataset')
    dataset.build()

    input_char_vocab = dataset.input_char_vocabulary
    assert input_char_vocab is not None
    target_char_vocab = dataset.target_char_vocabulary
    assert target_char_vocab is not None

    # dump current configuration and used vocabulary to this model's folder
    with open(os.path.join(save_model_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump((input_char_vocab, target_char_vocab), f)
    with open(os.path.join(save_model_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    model = BiLSTM(
        lstm_units=args.rnn_cell_dim,
        num_rnns=args.num_layers,
        input_alphabet_size=len(input_char_vocab.keys()),
        target_alphabet_size=len(target_char_vocab.keys()),
        embedding_dim=args.embedding,
        use_residual=args.use_residual,
        dropout=0.0
    )

    model.make(args.batch_size)
    model.summary(line_length=80)

    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    model.compile(
        optimizer=optimizer,
        run_eagerly=args.debug,
    )

    train_loader = dataset.get_loaders(batch_size=args.batch_size)

    tf.debugging.disable_traceback_filtering()

    model.fit(
        train_loader,
        # validation_data=eval_data_loader,
        batch_size=args.batch_size,
        epochs=args.epochs,
        # steps_per_epoch=train_dataset.total_steps,
        # validation_steps=eval_dataset.total_steps if eval_data_loader else None,
    )

    # if args.restore:
    #     logging.info('Restoring model from: {}'.format(checkpoint_path))
    #     network.restore(checkpoint_path)


if __name__ == "__main__":
    main(parse_args())