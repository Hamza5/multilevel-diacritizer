#!/usr/bin/env python3
from argparse import ArgumentParser
from logging import getLogger, basicConfig
from pathlib import Path
from constants import (DATASET_FILE_NAME, DEFAULT_DATA_DIR, DEFAULT_PARAMS_DIR, DEFAULT_TRAIN_STEPS, DEFAULT_BATCH_SIZE,
                       DEFAULT_EARLY_STOPPING_STEPS, DEFAULT_WINDOW_SIZE, DEFAULT_SLIDING_STEP, DEFAULT_MONITOR_METRIC,
                       DEFAULT_EMBEDDING_SIZE, DEFAULT_LSTM_SIZE, DEFAULT_DROPOUT_RATE)

basicConfig(level='INFO', format='%(asctime)s | %(name)s: %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
logger = getLogger(__name__)

if __name__ == '__main__':
    main_parser = ArgumentParser(description='Command-line text diacritics restoration tool.')
    subparsers = main_parser.add_subparsers(title='Commands', description='Available operations:',  dest='subcommand')
    data_parser = subparsers.add_parser('download-dataset', help='Get and extract the data of training and testing.')
    data_parser.add_argument('--data-dir', '-d', type=Path, default=DEFAULT_DATA_DIR,
                             help='Directory used for downloading and extracting the dataset.')
    data_parser.add_argument('--url', '-u', help='URL of the Tashkeela-processed dataset or any other dataset written'
                                                 ' in the same format.',
                             default='https://sourceforge.net/projects/tashkeela-processed/files/latest/download')
    train_parser = subparsers.add_parser('train', help='Train the model on a dataset.')
    train_parser.add_argument('--train-file', '-t', type=Path, required=True, action='append',
                              help='The file(s) containing the training data.')
    train_parser.add_argument('--val-file', '-v', type=Path, required=True, action='append',
                              help='The file(s) containing the testing data.')
    train_parser.add_argument('--params-dir', '-p', type=Path, default=DEFAULT_PARAMS_DIR,
                              help='Directory used to store the model parameters.')
    train_parser.add_argument('--epochs', '-e', type=int, default=DEFAULT_TRAIN_STEPS,
                              help='Maximum number of iterations before stopping the training process.')
    train_parser.add_argument('--batch-size', '-b', type=int, default=DEFAULT_BATCH_SIZE,
                              help='Maximum number of elements in a single batch.')
    train_parser.add_argument('--embedding-size', type=int, default=DEFAULT_EMBEDDING_SIZE,
                              help='The size of the embedding layer.')
    train_parser.add_argument('--lstm-size', type=int, default=DEFAULT_LSTM_SIZE,
                              help='The size of the lstm layers.')
    train_parser.add_argument('--dropout-rate', type=float, default=DEFAULT_DROPOUT_RATE,
                              help='The rate of the dropout.')
    train_parser.add_argument('--window-size', type=int, default=DEFAULT_WINDOW_SIZE,
                              help='The number of characters in a single instance of the data.')
    train_parser.add_argument('--sliding-step', type=int, default=DEFAULT_SLIDING_STEP,
                              help='The number of characters to skip to generate between the start of two consecutive'
                                   ' windows.')
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
    # test_parser = subparsers.add_parser('test', help='Test the model on a dataset.')
    # test_parser.add_argument('--data-dir', '-d', type=Path, default=DEFAULT_DATA_DIR,
    #                          help='Directory which contains vocabulary and data files.')
    # test_parser.add_argument('--params-dir', '-p', type=Path, default=DEFAULT_PARAMS_DIR,
    #                          help='Directory containing the model parameters.')
    # test_parser.add_argument('--batch-size', '-b', type=int, default=DEFAULT_BATCH_SIZE,
    #                          help='Maximum number of elements in a single batch.')
    # diacritization_parser = subparsers.add_parser('diacritize', help='Diacritize some text.')
    # diacritization_parser.add_argument('--text', default='', help='Undiacritized text.')
    # diacritization_parser.add_argument('--params-dir', '-p', type=Path, default=DEFAULT_PARAMS_DIR,
    #                                    help='Directory containing the model parameters.')
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
        import numpy as np
        from tensorflow.keras.optimizers import RMSprop
        from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
        from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN, LambdaCallback, EarlyStopping
        from multi_level_diacritization import MultiLevelDiacritizer

        model = MultiLevelDiacritizer(window_size=args.window_size, lstm_size=args.lstm_size,
                                      dropout_rate=args.dropout_rate, embedding_size=args.embedding_size,
                                      test_der=args.calculate_der, test_wer=args.calculate_wer)
        model.summary(positions=[.45, .6, .75, 1.])

        logger.info('Loading the training data...')
        train_set = MultiLevelDiacritizer.get_processed_window_dataset(
            [str(x) for x in args.train_file], args.batch_size, args.window_size, args.sliding_step
        )
        diacritics_factors = [np.max(x) / x for x in train_set['diacritics_count']]
        diacritics_factors = [x / np.sum(x) for x in diacritics_factors]

        logger.info('Loading the validation data...')
        val_set = MultiLevelDiacritizer.get_processed_window_dataset(
            [str(x) for x in args.val_file], args.batch_size, args.window_size, args.sliding_step
        )

        model.compile(RMSprop(0.001),
                      [SparseCategoricalCrossentropy(from_logits=True, name='primary_loss'),
                       SparseCategoricalCrossentropy(from_logits=True, name='secondary_loss'),
                       BinaryCrossentropy(from_logits=True, name='shadda_loss'),
                       BinaryCrossentropy(from_logits=True, name='sukoon_loss')])
        model_path = Path(
            f'params/{model.name}-E{args.embedding_size}L{args.lstm_size}W{args.window_size}S{args.sliding_step}.h5'
        )
        if model_path.exists():
            model.load_weights(str(model_path), by_name=True, skip_mismatch=True)

        last_epoch_path = Path(f'{args.params_dir}/last_epoch.txt')

        def write_epoch(epoch, logs):
            with last_epoch_path.open('w') as f:
                print(epoch, file=f)
                print(logs, file=f)

        def get_initial_epoch():
            if last_epoch_path.exists():
                with last_epoch_path.open() as f:
                    return int(f.readline())
            return 0

        model.fit(train_set['dataset'].repeat(), steps_per_epoch=train_set['size'], epochs=args.epochs,
                  initial_epoch=get_initial_epoch(),
                  # class_weight={output.name.split('/')[0]: dict(enumerate(diacritics_factors[i]))
                  #               for i, output in enumerate(model.outputs)},
                  validation_data=val_set['dataset'].repeat(), validation_steps=val_set['size'],
                  callbacks=[ModelCheckpoint(str(model_path), save_best_only=True, save_weights_only=True,
                                             monitor=args.monitor_metric), TerminateOnNaN(),
                             EarlyStopping(monitor=args.monitor_metric, patience=args.early_stopping_epochs, verbose=1),
                             LambdaCallback(
                                 on_epoch_end=lambda epoch, logs: logger.info(
                                     'Diacritization preview: %s',
                                     model.generate_sentence_from_batch(
                                         next(iter(
                                             val_set['dataset'].skip(randint(1, val_set['size'] - 1)).take(1)
                                         ))[0],
                                         args.sliding_step
                                     ).numpy().decode('UTF-8')
                                 )
                             ), LambdaCallback(on_epoch_end=write_epoch)
                             ]
                  )
        logger.info('Training finished.')
    # elif args.subcommand == 'test':
    #
    #     test(args.data_dir, args.params_dir, args.batch_size)
    # elif args.subcommand == 'diacritize':
    #     diacritization(args.text, args.params_dir)
