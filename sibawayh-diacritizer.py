#!/usr/bin/python3
from argparse import ArgumentParser

from dl_model import generate_data, train

if __name__ == '__main__':
    main_parser = ArgumentParser(description='Command-line text diacritics restoration tool.')
    subparsers = main_parser.add_subparsers(title='Commands', description='Available operations:',  dest='subcommand')
    data_parser = subparsers.add_parser('generate-data', help='Get and generate the data for training and testing.')
    data_parser.add_argument('--tmp-dir', '-t', default='tmp/', help='Directory used for downloading and extracting the'
                                                                     'dataset.')
    data_parser.add_argument('--data-dir', '-d', default='data/', help='Directory used for generating and storing'
                                                                       'vocabulary and data files.')
    train_parser = subparsers.add_parser('train', help='Train the model on a generated dataset.')
    train_parser.add_argument('--data-dir', '-d', default='data/', help='Directory which contains vocabulary and data'
                                                                        'files.')
    train_parser.add_argument('--output-dir', '-o', default='output/', help='Directory used to store model parameters'
                                                                            'and training progress values.')
    train_parser.add_argument('--train-steps', '-s', type=int, default=1000, help='Maximum number of steps before'
                                                                                  'stopping the training.')
    train_parser.add_argument('--model', '-m', default='transformer', help='The model used for training.')
    train_parser.add_argument('--hyper-parameters-set', '-p', default='transformer_base', help='Default set of'
                                                                                               'hyper-parameters')
    train_parser.add_argument('--override-hyper-parameters-config', '-c', default='hparams.json',
                              help='JSON file containing the hyper-parameters values modifications.')
    train_parser.add_argument('--save-checkpoint-every-seconds', '-v', type=int, default=60*60,
                              help='Save the model\'s parameters every n seconds')
    train_parser.add_argument('--max-checkpoints', '-x', type=int, default=3, help='Leave at most n recent parameters'
                                                                                   'files saved on disk.')
    train_parser.add_argument('--early-stopping-steps', '-e', type=int, default=5,
                              help='Number of training steps to wait before stopping the training if there is no'
                                   'improvement.')
    args = main_parser.parse_args()
    if args.subcommand == 'generate-data':
        generate_data(args.tmp_dir, args.data_dir)
        print('Data generated.')
    elif args.subcommand == 'train':
        train(args.model, args.data_dir, args.output_dir, args.hyper_parameters_set,
              args.override_hyper_parameters_config, args.train_steps, args.save_checkpoint_every_seconds,
              args.max_checkpoints, args.early_stopping_steps)
        print('Trained.')
