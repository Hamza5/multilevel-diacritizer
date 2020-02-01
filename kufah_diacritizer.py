#!/usr/bin/python3
from argparse import ArgumentParser
from pathlib import Path
from models import download_data, train

DATA_DIR = Path('data/')

if __name__ == '__main__':
    main_parser = ArgumentParser(description='Command-line text diacritics restoration tool.')
    subparsers = main_parser.add_subparsers(title='Commands', description='Available operations:',  dest='subcommand')
    data_parser = subparsers.add_parser('download-data', help='Get and generate the data for training and testing.')
    data_parser.add_argument('--data-dir', '-t', type=Path, default=DATA_DIR,
                             help='Directory used for downloading and extracting the dataset.')
    data_parser.add_argument('--url', '-u', help='URL of the Tashkeela-processed dataset.',
                             default='https://sourceforge.net/projects/tashkeela-processed/files/latest/download')
    args = main_parser.parse_args()
    if args.subcommand == 'download-data':
        download_data(args.tmp_dir, args.url)
        print('Generated.')
    elif args.subcommand == 'train':
        train(args.data_dir, args.params_dir)
        print('Trained.')
    # elif args.subcommand == 'diacritize':
    #     predict(args.model, args.data_dir, args.output_dir, args.hyper_parameters_set,
    #             args.override_hyper_parameters_config, args.beam_size, args.input_file, args.output_file)
