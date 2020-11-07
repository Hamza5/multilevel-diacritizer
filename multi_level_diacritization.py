import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

from processing import (NUMBER, NUMBER_PATTERN, DIACRITICS, PRIMARY_DIACRITICS, SECONDARY_DIACRITICS, SHADDA, SUKOON,
                        ARABIC_LETTERS)
from losses import WeightedBinaryCrossEntropy, WeightedSparseCategoricalCrossEntropy

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
        tf.lookup.KeyValueTensorInitializer(tf.range(1, len(CHARS)+1), tf.constant(CHARS)), ''
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


class MultiLevelDiacritizer(Model):

    def __init__(self, window_size=DEFAULT_WINDOW_SIZE, lstm_size=DEFAULT_LSTM_SIZE, dropout_rate=DEFAULT_DROPOUT_RATE,
                 embedding_size=DEFAULT_EMBEDDING_SIZE, name=None, **kwargs):
        inputs = Input(shape=(window_size,), name='input')
        embedding = Embedding(len(CHARS) + 1, embedding_size, name='embedding')(inputs)

        initial_layer = Bidirectional(LSTM(lstm_size, dropout=dropout_rate, return_sequences=True),
                                      name='initial_layer')(embedding)
        primary_diacritics_layer = Bidirectional(LSTM(lstm_size, dropout=dropout_rate, return_sequences=True),
                                                 name='primary_diacritics_layer')(initial_layer)
        primary_diacritics_output = Dense(4, name='primary_diacritics_output')(primary_diacritics_layer)

        secondary_diacritics_layer = Bidirectional(LSTM(lstm_size, dropout=dropout_rate, return_sequences=True),
                                                   name='secondary_diacritics_layer')(primary_diacritics_layer)
        secondary_diacritics_output = Dense(4, name='secondary_diacritics_output')(secondary_diacritics_layer)

        shadda_layer = Bidirectional(LSTM(lstm_size, dropout=dropout_rate, return_sequences=True),
                                     name='shadda_layer')(secondary_diacritics_layer)
        shadda_output = Dense(1, name='shadda_output')(shadda_layer)

        sukoon_layer = Bidirectional(LSTM(lstm_size, dropout=dropout_rate, return_sequences=True),
                                     name='sukoon_layer')(shadda_layer)
        sukoon_output = Dense(1, name='sukoon_output')(sukoon_layer)

        super(MultiLevelDiacritizer, self).__init__(
            inputs=inputs,
            outputs=[primary_diacritics_output, secondary_diacritics_output, shadda_output, sukoon_output],
            name=name or self.__class__.__name__,
            **kwargs
        )

    @classmethod
    def normalize_entities(cls, text):
        return tf.strings.strip(tf.strings.regex_replace(tf.strings.regex_replace(tf.strings.regex_replace(
            tf.strings.regex_replace(text, NUMBER_PATTERN.pattern, NUMBER),
            r'([^\p{Arabic}\p{P}\d\s' + ''.join(DIACRITICS) + '])+', ''),
            r'\p{P}+', ''), r'\s{2,}', ' '))

    @classmethod
    def separate_diacritics(cls, diacritized_text):
        letter_diacritic = tf.strings.split(
            tf.strings.regex_replace(diacritized_text, r'(.[' + ''.join(DIACRITICS) + ']*)', r'\1&'), '&'
        )
        decoded = tf.strings.unicode_decode(letter_diacritic, 'UTF-8')
        letters, diacritics = decoded[:-1, :1], decoded[:-1, 1:]
        return [tf.strings.unicode_encode(x, 'UTF-8') for x in (letters, diacritics)]

    @classmethod
    def filter_diacritics(cls, diacritics_sequence, filtered_diacritics):
        filtered_diacritics = tf.reshape(filtered_diacritics, (-1, 1))
        diacritics_positions = tf.reduce_any(diacritics_sequence == filtered_diacritics, axis=0)
        return tf.where(diacritics_positions, diacritics_sequence, '')

    @classmethod
    def clean_and_encode_sentence(cls, sentence):
        text = cls.normalize_entities(sentence)
        letters, diacritics = cls.separate_diacritics(text)
        padding = [[0, 1]]
        letters, diacritics = tf.pad(letters, padding), tf.pad(diacritics, padding)
        shadda_diacritics = cls.filter_diacritics(diacritics,
                                                  [SHADDA + x for x in (PRIMARY_DIACRITICS + SECONDARY_DIACRITICS)] +
                                                  [SHADDA])
        sukoon_diacritics = cls.filter_diacritics(diacritics, [SUKOON])
        primary_diacritics = cls.filter_diacritics(diacritics,
                                                   PRIMARY_DIACRITICS + [SHADDA + x for x in PRIMARY_DIACRITICS])
        secondary_diacritics = cls.filter_diacritics(diacritics,
                                                     SECONDARY_DIACRITICS + [SHADDA + x for x in SECONDARY_DIACRITICS])
        encoded_letters = ENCODE_LETTERS_TABLE.lookup(letters)
        encoded_shadda_diacritics = ENCODE_BINARY_TABLE.lookup(shadda_diacritics)
        encoded_sukoon_diacritics = ENCODE_BINARY_TABLE.lookup(sukoon_diacritics)
        encoded_primary_diacritics = ENCODE_PRIMARY_TABLE.lookup(primary_diacritics)
        encoded_secondary_diacritics = ENCODE_SECONDARY_TABLE.lookup(secondary_diacritics)
        return (encoded_letters,
                (encoded_primary_diacritics, encoded_secondary_diacritics, encoded_shadda_diacritics,
                 encoded_sukoon_diacritics))

    @classmethod
    def get_processed_window_dataset(cls, file_paths, batch_size, window_size, sliding_step):
        dataset = tf.data.TextLineDataset(file_paths).map(cls.clean_and_encode_sentence, tf.data.experimental.AUTOTUNE)
        zip_data = lambda x, y: tf.data.Dataset.zip((x, y))
        dataset = dataset.unbatch().window(window_size, sliding_step, drop_remainder=True) \
            .flat_map(zip_data).batch(window_size, drop_remainder=True).batch(batch_size)

        def count_diacritics(diacritics_count, new_element):
            _, (primary_diacritics, secondary_diacritics, shadda_diacritics, sukoon_diacritics) = new_element
            primary_count, secondary_count, shadda_count, sukoon_count = diacritics_count
            primary_count += tf.reduce_sum(
                tf.cast(tf.reshape(tf.range(4), (1, -1)) == tf.reshape(primary_diacritics, (-1, 1)), tf.int32),
                axis=0)
            secondary_count += tf.reduce_sum(
                tf.cast(tf.reshape(tf.range(4), (1, -1)) == tf.reshape(secondary_diacritics, (-1, 1)), tf.int32),
                axis=0)
            shadda_count += tf.reduce_sum(
                tf.cast(tf.reshape(tf.range(2), (1, -1)) == tf.reshape(shadda_diacritics, (-1, 1)), tf.int32),
                axis=0)
            sukoon_count += tf.reduce_sum(
                tf.cast(tf.reshape(tf.range(2), (1, -1)) == tf.reshape(sukoon_diacritics, (-1, 1)), tf.int32),
                axis=0)
            return primary_count, secondary_count, shadda_count, sukoon_count

        diacritics_count = dataset.reduce((tf.zeros(4, tf.int32), tf.zeros(4, tf.int32), tf.zeros(1, tf.int32),
                                           tf.zeros(1, tf.int32)), count_diacritics)
        diacritics_count = [x.numpy() for x in diacritics_count]
        size = dataset.reduce(0, lambda old, new: old + 1).numpy()
        return dataset.prefetch(tf.data.experimental.AUTOTUNE), size, diacritics_count

    @classmethod
    def combine_diacritics(cls, primary_diacritics, secondary_diacritics, shadda_diacritics, sukoon_diacritics):
        main_diacritics = tf.where(primary_diacritics != '', primary_diacritics,
                                   tf.where(secondary_diacritics != '', secondary_diacritics, sukoon_diacritics))
        return tf.strings.reduce_join((shadda_diacritics, main_diacritics), axis=0)

    @classmethod
    def decode_encoded_sentence(cls, encoded_letters, encoded_diacritics):
        (encoded_primary_diacritics, encoded_secondary_diacritics,
         encoded_shadda_diacritics, encoded_sukoon_diacritics) = encoded_diacritics
        letters = DECODE_LETTERS_TABLE.lookup(encoded_letters)
        primary_diacritics = DECODE_PRIMARY_TABLE.lookup(encoded_primary_diacritics)
        secondary_diacritics = DECODE_SECONDARY_TABLE.lookup(encoded_secondary_diacritics)
        shadda_diacritics = DECODE_SHADDA_TABLE.lookup(encoded_shadda_diacritics)
        sukoon_diacritics = DECODE_SUKOON_TABLE.lookup(encoded_sukoon_diacritics)
        diacritics = cls.combine_diacritics(primary_diacritics, secondary_diacritics, shadda_diacritics,
                                            sukoon_diacritics)
        return tf.strings.reduce_join(tf.strings.reduce_join((letters, diacritics), axis=0))


if __name__ == '__main__':
    import os.path
    import numpy as np
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN

    model = MultiLevelDiacritizer()
    model.summary()
    train_set, train_steps, diacritics_count = MultiLevelDiacritizer.get_processed_window_dataset(
        ['data/ATB3_train.txt'], 1024, DEFAULT_WINDOW_SIZE, DEFAULT_SLIDING_STEP
    )
    diacritics_factors = [np.max(x) / x for x in diacritics_count]
    diacritics_factors = [x / np.sum(x) for x in diacritics_factors]
    val_set, val_steps, _ = MultiLevelDiacritizer.get_processed_window_dataset(
        ['data/ATB3_val.txt'], 1024, DEFAULT_WINDOW_SIZE, DEFAULT_SLIDING_STEP
    )
    epochs = 200

    model.compile(RMSprop(0.001),
                  [WeightedSparseCategoricalCrossEntropy(from_logits=True, name='primary_loss',
                                                         class_weights=diacritics_factors[0]),
                   WeightedSparseCategoricalCrossEntropy(from_logits=True, name='secondary_loss',
                                                         class_weights=diacritics_factors[1]),
                   WeightedBinaryCrossEntropy(from_logits=True, name='shadda_loss',
                                              class_weights=diacritics_factors[2]),
                   WeightedBinaryCrossEntropy(from_logits=True, name='sukoon_loss',
                                              class_weights=diacritics_factors[3])],
                  ['accuracy']
                  )
    model_path = f'params/{model.name}.h5'
    if os.path.exists(model_path):
        model.load_weights(model_path)
    model.fit(train_set.repeat(epochs), steps_per_epoch=train_steps, epochs=epochs,
              validation_data=val_set.repeat(epochs), validation_steps=val_steps,
              callbacks=[ModelCheckpoint(model_path, save_best_only=True), TerminateOnNaN()])
