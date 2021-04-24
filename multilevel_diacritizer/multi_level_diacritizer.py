#!/usr/bin/env python3
import sys
from argparse import ArgumentParser, FileType
from logging import getLogger, basicConfig
from pathlib import Path

import tensorflow as tf

# The next two lines are added to avoid the crash of Tensorflow when calling a model on a GPU.
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from multilevel_diacritizer.constants import (
    DATASET_FILE_NAME, DEFAULT_DATA_DIR, DEFAULT_PARAMS_DIR, DEFAULT_TRAIN_STEPS, DEFAULT_BATCH_SIZE,
    DEFAULT_EARLY_STOPPING_STEPS, DEFAULT_WINDOW_SIZE, DEFAULT_SLIDING_STEP, DEFAULT_MONITOR_METRIC,
    DEFAULT_EMBEDDING_SIZE, DEFAULT_LSTM_SIZE, DEFAULT_DROPOUT_RATE, SENTENCE_TOKENIZATION_REGEXP, SENTENCE_SEPARATORS,
    DIACRITICS_PATTERN, DEFAULT_DIACRITIZATION_LINES_COUNT
)

from multilevel_diacritizer.model import MultiLevelDiacritizer

basicConfig(level='INFO', format='%(asctime)s [%(name)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = getLogger(__name__)


def get_dataset_from(data_paths, args):
    files_names = []
    for path in data_paths:
        if path.is_dir():
            files_names.extend(str(x) for x in path.iterdir())
        else:
            files_names.append(str(path))
    return MultiLevelDiacritizer.get_processed_window_dataset(files_names, args.batch_size, args.window_size,
                                                              args.sliding_step)


def get_loaded_model(args):
    model = MultiLevelDiacritizer(window_size=args.window_size, lstm_size=args.lstm_size,
                                  dropout_rate=args.dropout_rate, embedding_size=args.embedding_size,
                                  test_der=args.calculate_der, test_wer=args.calculate_wer)
    model.summary(positions=[.45, .6, .75, 1.], print_fn=logger.info)
    args.params_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.params_dir / Path(
        f'{model.name}-E{args.embedding_size}L{args.lstm_size}W{args.window_size}S{args.sliding_step}.h5'
    )
    if model_path.exists():
        model.load_weights(str(model_path), by_name=True, skip_mismatch=True)
        logger.info('Model weights loaded from %s ...', str(model_path))
    else:
        logger.warning('Random weights initialized for the model %s ...', model.name)
    return model, model_path


def get_sentences(text):
    fragments = list(filter(None, SENTENCE_TOKENIZATION_REGEXP.split(text)))
    u_sentences = []
    for s in fragments:
        if s.strip(SENTENCE_SEPARATORS + '\n') != '':
            u_sentences.append(s)
        else:
            u_sentences[-1] = u_sentences[-1] + s
    return u_sentences


def insert_d_words(u_sentence, d_words):
    start_index = 0
    d_sentence = u_sentence
    for d_word in d_words:
        d_word = d_word.numpy().decode('UTF-8')
        u_word = DIACRITICS_PATTERN.sub('', d_word)
        d_sentence = d_sentence[:start_index] + d_sentence[start_index:].replace(u_word, d_word, 1)
        start_index = d_sentence.find(d_word, start_index) + len(d_word)
    return d_sentence


def diacritize_text(model, args, text):
    text = DIACRITICS_PATTERN.sub('', text)
    u_sentences = get_sentences(text)
    d_sentence_words = model.diacritize_words(u_sentences, args.window_size, args.sliding_step)
    d_sentences = list(map(insert_d_words, u_sentences, d_sentence_words))
    start_index = 0
    for d_sentence, u_sentence in zip(d_sentences, u_sentences):
        text = text[:start_index] + text[start_index:].replace(u_sentence, d_sentence, 1)
        start_index += len(d_sentence)
    return text


def create_server_app(args):
    from flask import Flask, request, make_response

    args = server_parser.parse_args(args)
    args.dropout_rate = 0
    args.calculate_der = False
    args.calculate_wer = False
    model, model_path = get_loaded_model(args)

    app = Flask(__name__)

    @app.route('/', methods=['POST'])
    def home():
        text = request.get_data(as_text=True)
        response = make_response(diacritize_text(model, args, text))
        response.headers['Content-Type'] = 'text/plain; charset=UTF-8'
        return response

    return app


main_parser = ArgumentParser(description='Command-line text diacritics restoration tool.')
subparsers = main_parser.add_subparsers(title='Commands', description='Available operations:',  dest='subcommand')

data_parser = subparsers.add_parser('download-dataset', help='Get and extract the data of training and testing.')
data_parser.add_argument('--data-dir', '-d', type=Path, default=DEFAULT_DATA_DIR,
                         help='Directory used for downloading and extracting the dataset.')
data_parser.add_argument('--url', '-u', help='URL of the Tashkeela-processed dataset or any other dataset written'
                                             ' in the same format.',
                         default='https://sourceforge.net/projects/tashkeela-processed/files/latest/download')

common_args_parser = ArgumentParser(add_help=False)
common_args_parser.add_argument('--params-dir', '-p', type=Path, default=DEFAULT_PARAMS_DIR,
                                help='Directory used to store the model parameters.')
common_args_parser.add_argument('--batch-size', '-b', type=int, default=DEFAULT_BATCH_SIZE,
                                help='Maximum number of elements in a single batch.')
common_args_parser.add_argument('--embedding-size', type=int, default=DEFAULT_EMBEDDING_SIZE,
                                help='The size of the embedding layer.')
common_args_parser.add_argument('--lstm-size', type=int, default=DEFAULT_LSTM_SIZE,
                                help='The size of the lstm layers.')
common_args_parser.add_argument('--window-size', type=int, default=DEFAULT_WINDOW_SIZE,
                                help='The number of characters in a single instance of the data.')
common_args_parser.add_argument('--sliding-step', type=int, default=DEFAULT_SLIDING_STEP,
                                help='The number of characters to skip to generate between the start of two '
                                     'consecutive windows.')

train_parser = subparsers.add_parser('train', help='Train the model on a dataset.', parents=[common_args_parser])
train_parser.add_argument('--train-data', '-t', type=Path, required=True, action='append',
                          help='The file or directory containing the training data.')
train_parser.add_argument('--val-data', '-v', type=Path, required=True, action='append',
                          help='The file or directory containing the validation data.')
train_parser.add_argument('--epochs', '-e', type=int, default=DEFAULT_TRAIN_STEPS,
                          help='Maximum number of iterations before stopping the training process.')
train_parser.add_argument('--dropout-rate', type=float, default=DEFAULT_DROPOUT_RATE,
                          help='The rate of the dropout.')
train_parser.add_argument('--monitor-metric', '-m', default=DEFAULT_MONITOR_METRIC,
                          help='The metric to monitor to estimate the model performance when saving its weights and'
                               ' for early stopping.')
train_parser.add_argument('--early-stopping-epochs', '-s', type=int, default=DEFAULT_EARLY_STOPPING_STEPS,
                          help='Number of training iterations to wait before stopping the training if there is no '
                               'improvement.')
train_parser.add_argument('--calculate-der', '-d', action='store_true',
                          help='Calculate the Diacritization Error Rate on the validation dataset after each'
                               ' iteration.')
train_parser.add_argument('--calculate-wer', '-w', action='store_true',
                          help='Calculate the Word Error Rate on the validation dataset after each iteration.')

diacritization_parser = subparsers.add_parser('diacritization', help='Diacritize some text.',
                                              parents=[common_args_parser])
diacritization_parser.add_argument('--file', '-f', type=FileType('rt', encoding='UTF-8'), default=sys.stdin,
                                   help='The path of the file containing undiacritized arabic text.')
diacritization_parser.add_argument('--out-file', '-o', type=FileType('wt', encoding='UTF-8'), default=sys.stdout,
                                   help='The path of the generated file containing the arabic text diacritized.')
diacritization_parser.add_argument('--lines-at-once', '-l', type=int, default=DEFAULT_DIACRITIZATION_LINES_COUNT,
                                   help='Number of lines to diacritize at once.')
confusion_parser = subparsers.add_parser('confusion-matrix', help='Generate the confusion matrix from a predicted '
                                                                  'diacritized text file and a ground-truth one.')
confusion_parser.add_argument('predicted_file', help='The file generated by the diacritization model.',
                              type=FileType('rt', encoding='UTF-8'))
confusion_parser.add_argument('test_file',  help='The file having the ground-truth diacritics.',
                              type=FileType('rt', encoding='UTF-8'))
server_parser = subparsers.add_parser('server', help='Launch a diacritization web server.',
                                      parents=[common_args_parser])


if __name__ == '__main__':
    args = main_parser.parse_args()
    if args.subcommand == 'download-dataset':
        data_dir = args.data_dir.expanduser()
        dataset_file_path = data_dir.joinpath(DATASET_FILE_NAME)
        from tensorflow.keras.utils import get_file
        get_file(str(dataset_file_path.absolute()), args.url, cache_dir=str(data_dir.absolute()),
                 cache_subdir=str(data_dir.absolute()), extract=True)
        logger.info('Downloaded and extracted.')
    elif args.subcommand == 'train':
        from random import randint
        from tensorflow.keras.optimizers import RMSprop
        from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
        from tensorflow.keras.callbacks import (ModelCheckpoint, TerminateOnNaN, LambdaCallback, EarlyStopping,
                                                TensorBoard)

        model, model_path = get_loaded_model(args)

        logger.info('Loading the training data...')
        train_set = get_dataset_from(args.train_data, args)

        logger.info('Loading the validation data...')
        val_set = get_dataset_from(args.val_data, args)

        last_epoch_path = args.params_dir / Path('last_epoch.txt')

        def write_epoch(epoch, logs):
            with last_epoch_path.open('w') as f:
                print(epoch, file=f)
                print(logs, file=f)

        def get_initial_epoch():
            if last_epoch_path.exists():
                with last_epoch_path.open() as f:
                    return int(f.readline())
            return 0

        def get_diacritization_preview(val_set, sliding_step, model, limit):
            x, (pri, sec, sh, su) = next(iter(
                    val_set['dataset'].skip(randint(1, val_set['size'] - 1)).take(1)
                ))
            x, pri, sec, sh, su = x[:limit], pri[:limit], sec[:limit], sh[:limit], su[:limit]
            predicted = model.predict_sentence_from_input_batch(x, sliding_step).numpy().decode('UTF-8')
            real = model.generate_real_sentence_from_batch(
                (x, [pri, sec, sh, su]),
                sliding_step
            )
            return predicted, real

        model.compile(RMSprop(0.001),
                      [SparseCategoricalCrossentropy(from_logits=True, name='primary_loss'),
                       SparseCategoricalCrossentropy(from_logits=True, name='secondary_loss'),
                       BinaryCrossentropy(from_logits=True, name='shadda_loss'),
                       BinaryCrossentropy(from_logits=True, name='sukoon_loss')])
        model.fit(train_set['dataset'].repeat(), steps_per_epoch=train_set['size'], epochs=args.epochs,
                  initial_epoch=get_initial_epoch(),
                  validation_data=val_set['dataset'].repeat(), validation_steps=val_set['size'],
                  callbacks=[ModelCheckpoint(str(model_path), save_best_only=True, save_weights_only=True,
                                             monitor=args.monitor_metric), TerminateOnNaN(),
                             EarlyStopping(monitor=args.monitor_metric, patience=args.early_stopping_epochs, verbose=1),
                             LambdaCallback(
                                 on_epoch_end=lambda epoch, logs: logger.info(
                                     '\nPred: %s\nReal: %s',
                                     *get_diacritization_preview(val_set, args.sliding_step, model, 100)
                                 )
                             ), LambdaCallback(on_epoch_end=write_epoch), TensorBoard()
                             ]
                  )
        logger.info('Training finished.')
    elif args.subcommand == 'diacritization':
        args.dropout_rate = 0
        args.calculate_der = False
        args.calculate_wer = False
        model, model_path = get_loaded_model(args)

        def read_lines_at_once(file, num_lines):
            for line in file:
                text = line
                for _, line in zip(range(1, num_lines), file):
                    text += line
                yield text

        logger.info('Diacritizing...')
        for text in read_lines_at_once(args.file, args.lines_at_once):
            text = diacritize_text(model, args, text)
            print(text, end='', file=args.out_file)
        args.out_file.close()
        logger.info('Done')
    elif args.subcommand == 'confusion-matrix':
        from multilevel_diacritizer.confusion_matrix import generate_confusion_matrix
        generate_confusion_matrix(args.predicted_file, args.test_file)
    elif args.subcommand == 'server':
        import os
        import sys
        logger.info('Running the diacritization server...')
        os.environ['FLASK_APP'] = f'{__file__}:create_server_app({sys.argv[2:]})'
        os.environ['FLASK_ENV'] = 'development'  # TODO: Remove the debugging lines.
        os.environ['FLASK_DEBUG'] = '1'
        os.system('flask run')
    else:
        main_parser.print_help()
