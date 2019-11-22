#!/usr/bin/python3
from argparse import ArgumentParser

from dl_model import generate_data

if __name__ == '__main__':
    main_parser = ArgumentParser(description='Command-line text diacritics restoration tool.')
    subparsers = main_parser.add_subparsers(title='Commands', description='Available operations:', required=True,
                                            dest='subcommand')
    data_parser = subparsers.add_parser('generate-data', help='Get and generate the data for training and testing.')
    data_parser.add_argument('--tmp-dir', '-t', default='tmp/', help='Directory used for downloading and processing the'
                                                                     'dataset.')
    data_parser.add_argument('--data-dir', '-d', default='data/', help='Directory used for generating and storing the'
                                                                       'vocabulary file.')
    data_parser.add_argument('--download-url', '-u', default='', help='URL used to download the dataset.')
    args = main_parser.parse_args()
    if args.subcommand == 'generate-data':
        generate_data(args.tmp_dir, args.data_dir, args.download_url)
