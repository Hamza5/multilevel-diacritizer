import tensorflow as tf

from multilevel_diacritizer.constants import CHARS, SHORT_VOWELS, DOUBLE_CASE_ENDINGS, SHADDA, SUKOON


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
    tf.lookup.KeyValueTensorInitializer(tf.constant(SHORT_VOWELS + [SHADDA + x for x in SHORT_VOWELS]),
                                        tf.tile(tf.range(1, len(SHORT_VOWELS) + 1), [2])), 0
)
ENCODE_SECONDARY_TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.constant(DOUBLE_CASE_ENDINGS + [SHADDA + x for x in DOUBLE_CASE_ENDINGS]),
                                        tf.tile(tf.range(1, len(DOUBLE_CASE_ENDINGS) + 1), [2])), 0
)
DECODE_SHADDA_TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.constant([0]), tf.constant([''])), SHADDA
)
DECODE_SUKOON_TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.constant([0]), tf.constant([''])), SUKOON
)
DECODE_PRIMARY_TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.range(1, 4), tf.constant(SHORT_VOWELS)), ''
)
DECODE_SECONDARY_TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.range(1, 4), tf.constant(DOUBLE_CASE_ENDINGS)), ''
)
