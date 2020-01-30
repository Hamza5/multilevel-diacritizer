#!/usr/bin/python3
from argparse import ArgumentParser
from pathlib import Path
from models import download_data

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
        download_data(args.data_dir, args.url)
