#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
from tf_functions import download_data, train, test, diacritization

DATA_DIR = Path('data/')
PARAMS_DIR = Path('params/')
BATCH_SIZE = 128
TRAIN_STEPS = 1000
EARLY_STOPPING_STEPS = 5

if __name__ == '__main__':
    main_parser = ArgumentParser(description='Command-line text diacritics restoration tool.')
    subparsers = main_parser.add_subparsers(title='Commands', description='Available operations:',  dest='subcommand')
    data_parser = subparsers.add_parser('download-data', help='Get and generate the data for training and testing.')
    data_parser.add_argument('--data-dir', '-t', type=Path, default=DATA_DIR,
                             help='Directory used for downloading and extracting the dataset.')
    data_parser.add_argument('--url', '-u', help='URL of the Tashkeela-processed dataset.',
                             default='https://sourceforge.net/projects/tashkeela-processed/files/latest/download')
    train_parser = subparsers.add_parser('train', help='Train the model on a dataset.')
    train_parser.add_argument('--data-dir', '-d', type=Path, default=DATA_DIR,
                              help='Directory which contains vocabulary and data files.')
    train_parser.add_argument('--params-dir', '-p', type=Path, default=PARAMS_DIR,
                              help='Directory used to store model parameters and training progress values.')
    train_parser.add_argument('--train-steps', '-s', type=int, default=TRAIN_STEPS,
                              help='Maximum number of steps before stopping the training.')
    train_parser.add_argument('--batch-size', '-b', type=int, default=BATCH_SIZE,
                              help='Maximum number of elements in a single batch.')
    train_parser.add_argument('--early-stopping-steps', '-e', type=int, default=EARLY_STOPPING_STEPS,
                              help='Number of training steps to wait before stopping the training if there is no'
                                   'improvement.')
    test_parser = subparsers.add_parser('test', help='Test the model on a dataset.')
    test_parser.add_argument('--data-dir', '-d', type=Path, default=DATA_DIR,
                             help='Directory which contains vocabulary and data files.')
    test_parser.add_argument('--params-dir', '-p', type=Path, default=PARAMS_DIR,
                             help='Directory containing the model parameters.')
    test_parser.add_argument('--batch-size', '-b', type=int, default=BATCH_SIZE,
                             help='Maximum number of elements in a single batch.')
    diacritization_parser = subparsers.add_parser('diacritize', help='Diacritize some text.')
    diacritization_parser.add_argument('--text', default='', help='Undiacritized text.')
    diacritization_parser.add_argument('--params-dir', '-p', type=Path, default=PARAMS_DIR,
                                       help='Directory containing the model parameters.')
    args = main_parser.parse_args()
    if args.subcommand == 'download-data':
        download_data(args.data_dir, args.url)
        print('Generated.')
    elif args.subcommand == 'train':
        train(args.data_dir, args.params_dir, args.train_steps, args.batch_size, args.early_stopping_steps)
        print('Trained.')
    elif args.subcommand == 'test':
        test(args.data_dir, args.params_dir, args.batch_size)
    elif args.subcommand == 'diacritize':
        diacritization(args.text, args.params_dir)
