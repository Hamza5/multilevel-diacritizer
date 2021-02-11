import re
from pathlib import Path

import tensorflow as tf

PRIMARY_DIACRITICS = sorted('َُِ')
SECONDARY_DIACRITICS = sorted('ًٌٍ')
SHADDA = 'ّ'
SUKOON = 'ْ'
ARABIC_LETTERS = frozenset([chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)))])
ARABIC_PATTERN = re.compile(r'[%s]' % ''.join(ARABIC_LETTERS))
DIACRITICS = frozenset(PRIMARY_DIACRITICS + SECONDARY_DIACRITICS + [SHADDA, SUKOON])
DIACRITICS_PATTERN = re.compile(r'[%s]' % ''.join(DIACRITICS))
DIGIT = '0'
DIGIT_PATTERN = re.compile(r'\d')
SENTENCE_SEPARATORS = ';,،؛.:؟!'
SENTENCE_TOKENIZATION_REGEXP = re.compile(r'([%s](?!\w)|\n)' % SENTENCE_SEPARATORS)

DATASET_FILE_NAME = 'Tashkeela-processed.zip'
DEFAULT_WINDOW_SIZE = 30
DEFAULT_SLIDING_STEP = DEFAULT_WINDOW_SIZE // 3
DEFAULT_EMBEDDING_SIZE = 64
DEFAULT_LSTM_SIZE = 64
DEFAULT_DROPOUT_RATE = 0.3
DEFAULT_DATA_DIR = Path('data/')
DEFAULT_PARAMS_DIR = Path('params/')
DEFAULT_BATCH_SIZE = 1024
DEFAULT_TRAIN_STEPS = 100
DEFAULT_EARLY_STOPPING_STEPS = 4
DEFAULT_MONITOR_METRIC = 'val_loss'
DEFAULT_DIACRITIZATION_LINES_COUNT = 100


CHARS = sorted(ARABIC_LETTERS.union({DIGIT, ' '}))

ENCODE_LETTERS_TABLE = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(tf.constant(CHARS), tf.range(1, len(CHARS)+1)), 0
)
DECODE_LETTERS_TABLE = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(tf.range(1, len(CHARS)+1), tf.constant(CHARS)), '|'
)
ENCODE_BINARY_TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.constant(['']), tf.constant([0])), 1
)
ENCODE_PRIMARY_TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.constant(PRIMARY_DIACRITICS + [SHADDA + x for x in PRIMARY_DIACRITICS]),
                                        tf.tile(tf.range(1, len(PRIMARY_DIACRITICS)+1), [2])), 0
)
ENCODE_SECONDARY_TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.constant(SECONDARY_DIACRITICS + [SHADDA + x for x in SECONDARY_DIACRITICS]),
                                        tf.tile(tf.range(1, len(SECONDARY_DIACRITICS)+1), [2])), 0
)
DECODE_SHADDA_TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.constant([0]), tf.constant([''])), SHADDA
)
DECODE_SUKOON_TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.constant([0]), tf.constant([''])), SUKOON
)
DECODE_PRIMARY_TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.range(1, 4), tf.constant(PRIMARY_DIACRITICS)), ''
)
DECODE_SECONDARY_TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.range(1, 4), tf.constant(SECONDARY_DIACRITICS)), ''
)
