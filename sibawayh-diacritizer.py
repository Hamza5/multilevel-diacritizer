#!/usr/bin/python3
from argparse import ArgumentParser

from dl_model import generate_data, train, predict

DATA_DIR = 'data/'
TMP_DIR = 'tmp/'
OUTPUT_DIR = 'output/'
MODEL = 'transformer'
HPARAMS_SET = 'transformer_base'
OVERRIDE_HPARAMS_FILE = 'hparams.json'
CHECKPOINT_EVERY_SECONDS = 60*15
MAX_CHECKPOINTS = 3
EARLY_STOPPING_STEPS = 5
TRAIN_STEPS = 10000
BEAM_SIZE = 3

if __name__ == '__main__':
    main_parser = ArgumentParser(description='Command-line text diacritics restoration tool.')
    subparsers = main_parser.add_subparsers(title='Commands', description='Available operations:',  dest='subcommand')
    data_parser = subparsers.add_parser('generate-data', help='Get and generate the data for training and testing.')
    data_parser.add_argument('--tmp-dir', '-t', default=TMP_DIR,
                             help='Directory used for downloading and extracting the dataset.')
    data_parser.add_argument('--data-dir', '-d', default=DATA_DIR,
                             help='Directory used for generating and storing vocabulary and data files.')
    train_parser = subparsers.add_parser('train', help='Train the model on a generated dataset.')
    train_parser.add_argument('--data-dir', '-d', default=DATA_DIR,
                              help='Directory which contains vocabulary and data files.')
    train_parser.add_argument('--output-dir', '-o', default=OUTPUT_DIR,
                              help='Directory used to store model parameters and training progress values.')
    train_parser.add_argument('--train-steps', '-s', type=int, default=TRAIN_STEPS,
                              help='Maximum number of steps before stopping the training.')
    train_parser.add_argument('--model', '-m', default=MODEL, help='The model used for training.')
    train_parser.add_argument('--hyper-parameters-set', '-p', default=HPARAMS_SET,
                              help='Default set of hyper-parameters.')
    train_parser.add_argument('--override-hyper-parameters-config', '-c', default=OVERRIDE_HPARAMS_FILE,
                              help='JSON file containing the hyper-parameters values modifications.')
    train_parser.add_argument('--save-checkpoint-every-seconds', '-v', type=int, default=CHECKPOINT_EVERY_SECONDS,
                              help='Save the model\'s parameters every n seconds.')
    train_parser.add_argument('--max-checkpoints', '-x', type=int, default=MAX_CHECKPOINTS,
                              help='Leave at most n recent parameters files saved on disk.')
    train_parser.add_argument('--early-stopping-steps', '-e', type=int, default=EARLY_STOPPING_STEPS,
                              help='Number of training steps to wait before stopping the training if there is no'
                                   'improvement.')
    prediction_parser = subparsers.add_parser('diacritize', help='Use a pretrained model to restore the diacritics of'
                                                                 'an Arabic text.')
    prediction_parser.add_argument('--data-dir', '-d', default=DATA_DIR,
                                   help='Directory which contains vocabulary and data files.')
    prediction_parser.add_argument('--output-dir', '-o', default=OUTPUT_DIR,
                                   help='Directory which contains model parameters.')
    prediction_parser.add_argument('--model', '-m', default=MODEL, help='The model used for training.')
    prediction_parser.add_argument('--hyper-parameters-set', '-p', default=HPARAMS_SET,
                                   help='Default set of hyper-parameters.')
    prediction_parser.add_argument('--override-hyper-parameters-config', '-c', default=OVERRIDE_HPARAMS_FILE,
                                   help='JSON file containing the hyper-parameters values modifications.')
    prediction_parser.add_argument('--beam-size', '-b', type=int, default=BEAM_SIZE,
                                   help='The number of paths used for the beam search.')
    prediction_parser.add_argument('--input-file', '-i', default=None,
                                   help='The input file containing the text to diacritize.')
    prediction_parser.add_argument('--output-file', '-t', default=None,
                                   help='The output file where to generate the diacritized text.')
    args = main_parser.parse_args()
    if args.subcommand == 'generate-data':
        generate_data(args.tmp_dir, args.data_dir)
        print('Data generated.')
    elif args.subcommand == 'train':
        train(args.model, args.data_dir, args.output_dir, args.hyper_parameters_set,
              args.override_hyper_parameters_config, args.train_steps, args.save_checkpoint_every_seconds,
              args.max_checkpoints, args.early_stopping_steps)
        print('Trained.')
    elif args.subcommand == 'diacritize':
        predict(args.model, args.data_dir, args.output_dir, args.hyper_parameters_set,
                args.override_hyper_parameters_config, args.beam_size, args.input_file, args.output_file)
