from pathlib import Path

import tensorflow as tf
from transformers import TFXLNetModel, XLNetConfig

from processing import DIACRITICS, NUMBER, NUMBER_PATTERN, ARABIC_LETTERS

DATASET_FILE_NAME = 'Tashkeela-processed.zip'
SEQUENCE_LENGTH = 100  # TODO: Need to change this to a more accurate value.


@tf.function
def tf_separate_diacritics(d_text: tf.string):
    letter_diacritic = tf.strings.split(
        tf.strings.regex_replace(d_text, r'(.[' + ''.join(DIACRITICS) + ']*)', r'\1&'), '&'
    )
    decoded = tf.strings.unicode_decode(letter_diacritic, 'UTF-8')
    letters, diacritics = decoded[:-1, :1], decoded[:-1, 1:]
    return [tf.strings.unicode_encode(x, 'UTF-8') for x in (letters, diacritics)]


@tf.function
def tf_merge_diacritics(letters: tf.string, diacritics: tf.string):
    merged_letters_diacritics = tf.strings.reduce_join((letters, diacritics), 0)
    return tf.strings.reduce_join(merged_letters_diacritics, 0)


@tf.function
def tf_normalize_entities(text: tf.string):
    return tf.strings.strip(tf.strings.regex_replace(tf.strings.regex_replace(tf.strings.regex_replace(
        tf.strings.regex_replace(text, NUMBER_PATTERN.pattern, NUMBER),
        r'([^\p{Arabic}\p{P}\s'+''.join(DIACRITICS)+'])+', ''),
        r'\p{P}+', ''), r'\s{2,}', ' '))


CHARS = sorted(ARABIC_LETTERS.union({NUMBER, ' '}))
DIACS = sorted(DIACRITICS - {'ّ'}) + sorted('ّ'+x for x in (DIACRITICS - {'ّ'}))
LETTERS_TABLE = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(tf.constant(CHARS), tf.range(1, len(CHARS)+1)), len(CHARS)+1
)
DIACRITICS_TABLE = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(tf.constant(DIACS), tf.range(1, len(DIACS)+1)), 0
)


@tf.function
def tf_encode(letters: tf.string, diacritics: tf.string):
    return LETTERS_TABLE.lookup(letters), DIACRITICS_TABLE.lookup(diacritics)


@tf.function
def tf_pad_with_attention_mask(letters: tf.int32, diacritics: tf.int32):
    letters = tf.pad(letters, [[0, SEQUENCE_LENGTH]], constant_values=0)[:SEQUENCE_LENGTH]
    diacritics = tf.pad(diacritics, [[0, SEQUENCE_LENGTH]], constant_values=0)[:SEQUENCE_LENGTH]
    return (letters, tf.cast(letters > 0, tf.int32)), diacritics


@tf.function
def tf_data_processing(text: tf.string):
    text = tf_normalize_entities(text)
    letters, diacritics = tf_separate_diacritics(text)
    encoded_letters, encoded_diacritics = tf_encode(letters, diacritics)
    return tf_pad_with_attention_mask(encoded_letters, encoded_diacritics)


@tf.function
def no_padding_loss(y_true, y_pred):
    return tf.keras.metrics.sparse_categorical_crossentropy(y_true[y_true > 0], y_pred[y_true > 0])


@tf.function
def no_padding_accuracy(y_true, y_pred):
    return tf.keras.metrics.sparse_categorical_accuracy(y_true[y_true > 0], y_pred[y_true > 0])


class XLNetDiacritizationModel(TFXLNetModel):

    def __init__(self, config, *inputs, **kwargs):
        super(XLNetDiacritizationModel, self).__init__(config, *inputs, **kwargs)
        self.classifier = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(DIACS)+1, activation='softmax'),
                                                          input_shape=(SEQUENCE_LENGTH, self.config.d_model))

    def call(self, inputs, **kwargs):
        return self.classifier(super(XLNetDiacritizationModel, self).call(inputs, **kwargs)[0])


def download_data(data_dir, download_url):
    assert isinstance(data_dir, Path)
    assert isinstance(download_url, str)
    data_dir = data_dir.expanduser()
    dataset_file_path = data_dir.joinpath(DATASET_FILE_NAME)
    tf.keras.utils.get_file(str(dataset_file_path.absolute()), download_url, cache_dir=str(data_dir.absolute()),
                            cache_subdir=str(data_dir.absolute()), extract=True)


def train(data_dir, params_dir, epochs, batch_size, early_stop):
    assert isinstance(data_dir, Path)
    assert isinstance(params_dir, Path)
    assert isinstance(epochs, int)
    assert isinstance(batch_size, int)
    assert isinstance(early_stop, int)
    train_file_paths = [str(data_dir.joinpath(p)) for p in data_dir.glob('*train*.txt')]
    val_file_paths = [str(data_dir.joinpath(p)) for p in data_dir.glob('*val*.txt')]
    train_dataset = tf.data.TextLineDataset(train_file_paths).repeat()\
        .map(tf_data_processing, tf.data.experimental.AUTOTUNE).batch(batch_size, True).prefetch(1)
    val_dataset = tf.data.TextLineDataset(val_file_paths).repeat()\
        .map(tf_data_processing, tf.data.experimental.AUTOTUNE).batch(batch_size, True).prefetch(1)
    config = XLNetConfig.from_pretrained('xlnet-base-cased', cache_dir=str(params_dir.absolute()),
                                         vocab_size=len(CHARS)+1)
    xlnet = XLNetDiacritizationModel(config)
    xlnet.compile(tf.keras.optimizers.RMSprop(), no_padding_loss, [no_padding_accuracy])
    train_steps = tf.data.TextLineDataset(train_file_paths).batch(batch_size)\
        .reduce(0, lambda old, new: old + 1).numpy()
    val_steps = tf.data.TextLineDataset(val_file_paths).batch(batch_size)\
        .reduce(0, lambda old, new: old + 1).numpy()
    last_iteration = 0
    weight_files = sorted(params_dir.glob(xlnet.name+'-*.h5'))
    if len(weight_files) > 0:
        last_weights_file = str(params_dir.joinpath(weight_files[-1]).absolute())
        last_iteration = int(last_weights_file.split('-')[-1].split('.')[0])
        xlnet.load_weights(last_weights_file)
    xlnet.fit(train_dataset, steps_per_epoch=train_steps, epochs=epochs, validation_data=val_dataset,
              validation_steps=val_steps, initial_epoch=last_iteration,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=early_stop, verbose=1, restore_best_weights=True),
                         tf.keras.callbacks.ModelCheckpoint(
                             str(params_dir.joinpath(xlnet.name+'-{epoch:04d}.h5').absolute()), save_best_only=True,
                             save_weights_only=True
                         ),
                         tf.keras.callbacks.TerminateOnNaN(), tf.keras.callbacks.TensorBoard(str(data_dir.absolute()))]
              )
