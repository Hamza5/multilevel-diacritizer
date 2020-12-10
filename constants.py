import re

import tensorflow as tf


PRIMARY_DIACRITICS = sorted('َُِ')
SECONDARY_DIACRITICS = sorted('ًٌٍ')
SHADDA = 'ّ'
SUKOON = 'ْ'
ARABIC_LETTERS = frozenset([chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)))])
DIACRITICS = frozenset(PRIMARY_DIACRITICS + SECONDARY_DIACRITICS + [SHADDA, SUKOON])
NUMBER = '0'
NUMBER_PATTERN = re.compile(r'\d+(?:\.\d+)?')


DEFAULT_WINDOW_SIZE = 32
DEFAULT_SLIDING_STEP = 5
DEFAULT_EMBEDDING_SIZE = 64
DEFAULT_LSTM_SIZE = 128
DEFAULT_DROPOUT_RATE = 0.15


CHARS = sorted(ARABIC_LETTERS.union({NUMBER, ' '}))

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
