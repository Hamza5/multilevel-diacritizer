#!/usr/bin/env python3

"""
Script containing several functions to read and normalize the Tashkeela dataset and to convert its data to a standard
format.
"""

import re
import sys
import logging
import random
import os

from argparse import ArgumentParser
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import unquote
from zipfile import ZipFile

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%I:%M:%S%p')

# Hexadecimal values taken from https://www.unicode.org/charts/
D_NAMES = ['Fathatan', 'Dammatan', 'Kasratan', 'Fatha', 'Damma', 'Kasra', 'Shadda', 'Sukun']
NAME2DIACRITIC = dict((name, chr(code)) for name, code in zip(D_NAMES, range(0x064B, 0x0653)))
ARABIC_DIACRITICS = frozenset(NAME2DIACRITIC.values())
ARABIC_LETTERS = frozenset([chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)))])
EXTRA_SUKUN_REGEXP = re.compile(r'(?<=ال)' + NAME2DIACRITIC['Sukun'])
# YA_REGEXP = re.compile(r'ى(?=['+''.join(ARABIC_DIACRITICS)+r'])')
DIACRITIC_SHADDA_REGEXP = re.compile('(['+''.join(ARABIC_DIACRITICS)+'])('+NAME2DIACRITIC['Shadda']+')')
XML_TAG = r'(?:<.+>)+'
SENTENCE_SEPARATORS = ';,،؛.:؟!'
SPACES = ' \t'
DATETIME_REGEXP = re.compile(r'(?:\d+[-/:\s]+)+\d+')
NUMBER_REGEXP = re.compile(r'\d+(?:\.\d+)?')
ZERO_REGEXP = re.compile(r'\b0\b')
DOTS_NO_URL = r'(?<!\w)(['+SENTENCE_SEPARATORS+r']+)(?!\w)'
WORD_TOKENIZATION_REGEXP = re.compile('((?:[' + ''.join(ARABIC_LETTERS) + ']['+''.join(ARABIC_DIACRITICS)+r']*)+|\d+(?:\.\d+)?)')
SENTENCE_TOKENIZATION_REGEXP = re.compile(DOTS_NO_URL + '|' + XML_TAG)


def extract_diacritics(text):
    """
    Return the diacritics from the text while keeping their original positions including the Shadda marks.
    :param text: str, the diacritized text.
    :return: list, the diacritics. Positions with double diacritics have a tuple as elements.
    """
    assert isinstance(text, str)
    diacritics = []
    for i in range(1, len(text)):
        if text[i] in ARABIC_DIACRITICS:
            if text[i-1] == NAME2DIACRITIC['Shadda']:
                diacritics[-1] = (text[i-1], text[i])
            else:
                diacritics.append(text[i])
        elif text[i - 1] not in ARABIC_DIACRITICS:
            diacritics.append('')
    if text[-1] not in ARABIC_DIACRITICS:
        diacritics.append('')
    return diacritics


def fix_diacritics_errors(diacritized_text):
    """
    Fix and normalize some diacritization errors in the sentences.
    :param diacritized_text: the text containing the arabic letters with diacritics.
    :return: str, the fixed text.
    """
    assert isinstance(diacritized_text, str)
    # Remove the extra Sukun from ال
    diacritized_text = EXTRA_SUKUN_REGEXP.sub('', diacritized_text)
    # Fix misplaced Fathatan
    diacritized_text = diacritized_text.replace('اً', 'ًا')
    # Fix reversed Shadda-Diacritic
    diacritized_text = DIACRITIC_SHADDA_REGEXP.sub(r'\2\1', diacritized_text)
    # Fix ى that should be ي (disabled)
    # diacritized_text = YA_REGEXP.sub('ي', diacritized_text)
    # Remove the duplicated diacritics by leaving the second one only when there are two incompatible diacritics
    fixed_text = diacritized_text[0]
    for x in diacritized_text[1:]:
        if x in ARABIC_DIACRITICS and fixed_text[-1] in ARABIC_DIACRITICS:
            if fixed_text[-1] != NAME2DIACRITIC['Shadda'] or x == NAME2DIACRITIC['Shadda']:
                fixed_text = fixed_text[:-1]
        # Remove the diacritics that are without letters
        elif x in ARABIC_DIACRITICS and fixed_text[-1] not in ARABIC_LETTERS:
            continue
        fixed_text += x
    return fixed_text


def clean_text(text):
    """
    Remove the unwanted characters from the text.
    :param text: str, the unclean text.
    :return: str, the cleaned text.
    """
    assert isinstance(text, str)
    # Clean HTML garbage, tatweel, dates.
    return DATETIME_REGEXP.sub('', text.replace('ـ', '').replace('&quot;', ''))


def tokenize(sentence):
    """
    Tokenize a sentence into a list of words.
    :param sentence: str, the sentence to be tokenized.
    :return: list of str, list containing the words.
    """
    assert isinstance(sentence, str)
    return list(filter(lambda x: x != '' and x.isprintable(), re.split(WORD_TOKENIZATION_REGEXP, sentence)))


def filter_tokenized_sentence(sentence, min_words=2, min_word_diac_rate=0.8, min_word_diac_ratio=0.5):
    """
    Accept or void a sentence, and clean the tokens.
    :param sentence: the sentence to be filtered.
    :param min_words: minimum number of arabic words that must be left in the cleaned sentence in order to be accepted.
    :param min_word_diac_rate: rate of the diacritized words to the number of arabic words in the sentence.
    :param min_word_diac_ratio: ratio of the diacritized letters to the number of letters in the word.
    :return: list of str, the cleaned tokens or an empty list.
    """
    assert isinstance(sentence, list) and all(isinstance(w, str) for w in sentence)
    assert min_words >= 0
    assert min_word_diac_rate >= 0
    new_sentence = []
    if len(sentence) > 0:
        diac_word_count = 0
        arabic_word_count = 0
        for token in sentence:
            token = token.strip()
            if not token:
                continue
            word_chars = set(token)
            if word_chars & ARABIC_LETTERS != set():
                arabic_word_count += 1
                word_diacs = extract_diacritics(token)
                if len([x for x in word_diacs if x]) / len(word_diacs) >= min_word_diac_ratio:
                    diac_word_count += 1
            new_sentence.append(token)
        if arabic_word_count > 0 and arabic_word_count >= min_words:
            if diac_word_count / arabic_word_count >= min_word_diac_rate:
                return new_sentence
    return []


def read_text_file(file_path):
    """
    Reads a text file and returns a list of individual sentences.
    :param file_path: The path of the file.
    :return: list of str, each str is a sentence.
    """
    assert isinstance(file_path, str)
    sentences = []
    with open(file_path, 'rt', encoding='utf-8') as dataset_file:
        for line in dataset_file:
            line = clean_text(line.strip(SPACES+'\n'))
            if line == '':
                continue
            fragments = list(filter(lambda x: x != '',
                                    [x.strip(SPACES) for x in re.split(SENTENCE_TOKENIZATION_REGEXP, line)
                                     if x is not None]))
            if len(fragments) > 1:
                for f1, f2 in zip(fragments[:-1], fragments[1:]):
                    if set(f2).issubset(set(SENTENCE_SEPARATORS)):
                        sentences.append(f1+f2)
                    elif set(f1).issubset(set(SENTENCE_SEPARATORS)):
                        continue
                    else:
                        sentences.append(f1)
            else:
                sentences.extend(fragments)
    return sentences


def print_progress_bar(current, maximum):
    assert isinstance(current, int)
    assert isinstance(maximum, int)
    progress_text = '[{:50s}] {:d}/{:d} ({:0.2%})'.format('=' * int(current / maximum * 50), current, maximum,
                                                          current / maximum)
    sys.stderr.write('\r' + progress_text)
    sys.stderr.flush()


def preprocess(source, destination, min_words, ratio_diac_words, max_chars_count, ratio_diac_letters):
    with destination.open('w', encoding='UTF-8') as dest_file:
        if source.is_dir():
            sentences = []
            for (dirpath, dirnames, filenames) in os.walk(source):
                for file_name in filenames:
                    file_path = os.path.join(dirpath, file_name)
                    logging.info('Parsing {} ...'.format(file_path))
                    sentences.extend(read_text_file(str(file_path)))
        elif source.is_file():
            logging.info('Parsing {} ...'.format(source))
            sentences = read_text_file(str(source))
        else:
            logging.critical('{} is neither a file nor a directory!'.format(source))
            sys.exit(-2)
        logging.info('Preprocessing sentences...')
        filtered_sentences = set()
        for i, sf in enumerate(
                filter(lambda x: len(x) > 0,
                       map(lambda s: filter_tokenized_sentence(tokenize(fix_diacritics_errors(s)), min_words,
                                                               ratio_diac_words, ratio_diac_letters),
                           sentences)), 1):
            filtered_sentences.add(' '.join(sf))
            print_progress_bar(i, len(sentences))
        print(file=sys.stderr)
        logging.info('Generating file {} ...'.format(destination))
        for sf in filtered_sentences:
            print(sf[:max_chars_count].rstrip(), file=dest_file)
    logging.info('Pre-processing finished successfully.')


def partition(dataset_file, train_ratio, val_test_ratio, shuffle_every):

    def write_train_val_test_parts(train_path, val_path, test_path, sentences):
        train_size = round(train_ratio * len(sentences))
        val_size = round(val_test_ratio * (len(sentences) - train_size))
        random.shuffle(sentences)
        with train_path.open('a', encoding='UTF-8') as train_file:
            for s in sentences[:train_size]:
                train_file.write(s)
        with val_path.open('a', encoding='UTF-8') as val_file:
            for s in sentences[train_size:train_size + val_size]:
                val_file.write(s)
        with test_path.open('a', encoding='UTF-8') as test_file:
            for s in sentences[train_size + val_size:]:
                test_file.write(s)

    # Prepare files for train, validation and test
    train_path = dataset_file.with_name(dataset_file.stem + '_train.txt')
    val_path = dataset_file.with_name(dataset_file.stem + '_val.txt')
    test_path = dataset_file.with_name(dataset_file.stem + '_test.txt')
    train_path.open('w').close()
    val_path.open('w').close()
    test_path.open('w').close()
    logging.info('Generating sets from {} ...'.format(dataset_file))
    with dataset_file.open('r', encoding='UTF-8') as data_file:
        sentences = []
        for line in data_file:
            sentences.append(line)
            if len(sentences) % shuffle_every == 0:
                write_train_val_test_parts(train_path, val_path, test_path, sentences)
                logging.info('{} sentences written.'.format(len(sentences)))
                sentences.clear()
        write_train_val_test_parts(train_path, val_path, test_path, sentences)
        logging.info('{} sentences written'.format(len(sentences)))
    logging.info('Partitioning finished successfully.')


def download_extract_dataset(url):

    with urlopen(url) as dataset_source_file:
        headers = dataset_source_file.info()
        size = int(headers['Content-Length'])
        file_name = unquote(os.path.basename(dataset_source_file.geturl()))
        logging.info('Downloading {} ...'.format(file_name))
        with open(file_name, 'wb') as dataset_destination_file:
            bytes_read = 1
            counter = 0
            while bytes_read:
                bytes_read = dataset_source_file.read(1024)
                if bytes_read:
                    dataset_destination_file.write(bytes_read)
                    counter += len(bytes_read)
                print_progress_bar(counter, size)
        print(file=sys.stderr)
        logging.info('{} downloaded.'.format(file_name))
        dataset_zip_file = ZipFile(file_name)
        extract_dir = os.path.splitext(file_name)[0]
        logging.info('Extracting {} to {}/ ...'.format(file_name, extract_dir))
        dataset_zip_file.extractall(extract_dir)
        logging.info('Extraction finished.')


if __name__ == '__main__':
    args_parser = ArgumentParser(description='Command line tool for Tashkeela dataset processing.')
    subparsers = args_parser.add_subparsers(title='Sub-commands', description='Available operations',
                                            dest='subcommand', required=True)
    preprocessing_p = subparsers.add_parser('preprocess',
                                            help='Transform Arabic raw text files to a preprocessed dataset by'
                                                 'splitting sentences, dropping punctuation and noise, normalizing'
                                                 'spaces and numbers, then keeping only the highly diacritized'
                                                 'sentences.')
    preprocessing_p.add_argument('source', type=Path, help='Path of a raw text file or the root directory containing'
                                                           'the text files directly or in any subdirectory.')
    preprocessing_p.add_argument('destination', type=Path, help='Path of the generated text file after processing.')
    preprocessing_p.add_argument('--min-words', '-w', type=int, default=2,
                                 help='Minimum number of arabic words that must be left in the cleaned sentence in'
                                      'order to be accepted.')
    preprocessing_p.add_argument('--min-diac-words-ratio', '-d', type=float, default=1,
                                 help='Minimum rate of the diacritized words to the number of arabic words in the'
                                      'sentence.')
    preprocessing_p.add_argument('--min-diac-letters-ratio', '-l', type=float, default=0.5,
                                 help='Minimum ratio of the diacritized letters to the number of the letters in the'
                                      'word.')
    preprocessing_p.add_argument('--max-chars-count', '-c', type=int, default=2000,
                                 help='Maximum number of characters to keep in a long sentence.')
    partition_p = subparsers.add_parser('partition', help='Divide a dataset to train, validation and test fragments.')
    partition_p.add_argument('dataset_file', type=Path, help='The preprocessed dataset file.')
    partition_p.add_argument('--train-ratio', '-t', type=float, default=0.9, help='Ratio of data for training.')
    partition_p.add_argument('--val-test-ratio', '-v', type=float, default=0.5, help='Split ratio between validation'
                                                                                     'and test data.')
    partition_p.add_argument('--shuffle-every', '-s', type=int, default=100000,
                             help='Number of sentences to accumulate before shuffling.')
    download_p = subparsers.add_parser('download', help='Download the original Tashkeela dataset and extract it in a'
                                                        'new folder in the working directory.')
    download_p.add_argument('--url', '-u', default='https://sourceforge.net/projects/tashkeela/files/latest/download',
                            help='The URL of the original Tashkeela dataset.')
    args = args_parser.parse_args()
    if args.subcommand == 'download':
        download_extract_dataset(args.url)
    elif args.subcommand == 'preprocess':
        preprocess(args.source, args.destination, args.min_words, args.min_diac_words_ratio, args.max_chars_count,
                   args.min_diac_letters_ratio)
    elif args.subcommand == 'partition':
        partition(args.dataset_file, args.train_ratio, args.val_test_ratio, args.shuffle_every)
