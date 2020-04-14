from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, Bidirectional
# tf.config.experimental_run_functions_eagerly(True)

from processing import DIACRITICS, NUMBER, NUMBER_PATTERN, SEPARATED_SUFFIXES, SEPARATED_PREFIXES, MIN_STEM_LEN,\
    HAMZAT_PATTERN, ORDINARY_ARABIC_LETTERS_PATTERN

DATASET_FILE_NAME = 'Tashkeela-processed.zip'
SEQUENCE_LENGTH = 512  # TODO: Need to change this to a more accurate value.
OPTIMIZER = tf.keras.optimizers.RMSprop()
TF_CHAR_ENCODING = 'UTF8_CHAR'
MONITOR_VALUE = 'loss'


@tf.function
def tf_max_length_pairs(pairs: tf.Tensor):
    lengths = tf.strings.length(tf.strings.reduce_join(pairs, -1), TF_CHAR_ENCODING)
    return pairs[tf.argmax(lengths)]


@tf.function
def tf_separate_affixes(u_word: tf.string):

    def regex_match(x):
        return tf.strings.regex_full_match(x[0], x[1])

    prefixes = tf.constant(sorted(SEPARATED_PREFIXES.union([''])))
    suffixes = tf.constant(sorted(SEPARATED_SUFFIXES.union([''])))
    prefixes_patterns = tf.strings.join([tf.constant('^'), prefixes, tf.constant('.+')])
    suffixes_patterns = tf.strings.join([tf.constant('.+'), suffixes, tf.constant('$')])
    possible_prefixes = tf.map_fn(regex_match,
                                  tf.stack([tf.broadcast_to(u_word, prefixes_patterns.shape), prefixes_patterns], -1),
                                  tf.bool)
    possible_suffixes = tf.map_fn(regex_match,
                                  tf.stack([tf.broadcast_to(u_word, suffixes_patterns.shape), suffixes_patterns], -1),
                                  tf.bool)
    accepted_affixes = tf.TensorArray(tf.string, size=1, dynamic_size=True, clear_after_read=True, element_shape=(2,))
    accepted_affixes.write(accepted_affixes.size(), tf.constant(['', '']))  # Words with length less than the threshold.
    for i in tf.where(possible_prefixes)[:, 0]:
        for j in tf.where(possible_suffixes)[:, 0]:
            stem_len = tf.strings.length(u_word, TF_CHAR_ENCODING) - (tf.strings.length(prefixes[i], TF_CHAR_ENCODING) +
                                                                      tf.strings.length(suffixes[j], TF_CHAR_ENCODING))
            if stem_len >= MIN_STEM_LEN:
                accepted_affixes = accepted_affixes.write(accepted_affixes.size(),
                                                          tf.stack((prefixes[i], suffixes[j]), -1))
    accepted_affixes = accepted_affixes.stack()
    prefix_suffix = tf_max_length_pairs(accepted_affixes)
    p_len = tf.strings.length(prefix_suffix[0], TF_CHAR_ENCODING)
    s_len = tf.strings.length(prefix_suffix[1], TF_CHAR_ENCODING)
    w_len = tf.strings.length(u_word, TF_CHAR_ENCODING)
    return tf.stack([prefix_suffix[0], tf.strings.substr(u_word, p_len, w_len - (p_len + s_len), TF_CHAR_ENCODING),
                     prefix_suffix[1]])


@tf.function
def tf_word_to_pattern(word: tf.string):
    letters, diacritics = tf_separate_diacritics(word)
    u_word = tf.strings.reduce_join(letters)
    pre_st_suf = tf_separate_affixes(u_word)
    prefix, stem, suffix = pre_st_suf[0], pre_st_suf[1], pre_st_suf[2]
    stem = tf.strings.regex_replace(stem, 'ى', 'ا')
    stem = tf.strings.regex_replace(stem, HAMZAT_PATTERN.pattern, 'ء')
    stem = tf.strings.regex_replace(stem, ORDINARY_ARABIC_LETTERS_PATTERN.pattern, 'ح')
    return tf_merge_diacritics(tf.strings.unicode_split(tf.strings.join([prefix, stem, suffix]), 'UTF-8'), diacritics)


@tf.function
def tf_convert_to_pattern(d_text: tf.string):
    parts = tf.strings.split(tf.strings.regex_replace(d_text, r'(\w+|\p{P}|\s)', r'\1&'), '&')
    return tf.strings.reduce_join(tf.map_fn(tf_word_to_pattern, parts))


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


CHARS = sorted({'ح', 'ء', 'ة', NUMBER, ' '}.union(''.join(SEPARATED_PREFIXES.union(SEPARATED_SUFFIXES))))
DIACS = sorted(DIACRITICS.difference({'ّ'}).union({''}).union('ّ'+x for x in DIACRITICS.difference({'ّ'})))
LETTERS_TABLE = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(tf.constant(CHARS), tf.range(1, len(CHARS)+1)), 0
)
DIACRITICS_TABLE = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(tf.constant(DIACS), tf.range(1, len(DIACS)+1)), 0
)


@tf.function
def tf_encode(letters: tf.string, diacritics: tf.string):
    return LETTERS_TABLE.lookup(letters), DIACRITICS_TABLE.lookup(diacritics)


@tf.function
def tf_pad(letters: tf.int32, diacritics: tf.int32):
    letters = tf.pad(letters, [[0, SEQUENCE_LENGTH]], constant_values=0)[:SEQUENCE_LENGTH]
    diacritics = tf.pad(diacritics, [[0, SEQUENCE_LENGTH]], constant_values=0)[:SEQUENCE_LENGTH]
    return letters, diacritics


@tf.function
def tf_data_processing(text: tf.string):
    text = tf_convert_to_pattern(tf_normalize_entities(text))
    letters, diacritics = tf_separate_diacritics(text)
    encoded_letters, encoded_diacritics = tf_encode(letters, diacritics)
    return tf_pad(encoded_letters, encoded_diacritics)


@tf.function
def no_padding_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.cast(tf.argmax(y_pred, axis=-1), tf.float32))[tf.greater(y_true, 0)], tf.float32))


def download_data(data_dir, download_url):
    assert isinstance(data_dir, Path)
    assert isinstance(download_url, str)
    data_dir = data_dir.expanduser()
    dataset_file_path = data_dir.joinpath(DATASET_FILE_NAME)
    tf.keras.utils.get_file(str(dataset_file_path.absolute()), download_url, cache_dir=str(data_dir.absolute()),
                            cache_subdir=str(data_dir.absolute()), extract=True)


def get_model(params_dir):
    assert isinstance(params_dir, Path)
    model = Sequential([
        Embedding(len(CHARS)+1, 128, input_length=SEQUENCE_LENGTH),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.1)),
        Bidirectional(LSTM(32, return_sequences=True, dropout=0.1)),
        TimeDistributed(Dense(len(DIACS)+1))
    ], name='BLSTM128-BLSTM32')
    model.compile(OPTIMIZER, tf.keras.losses.SparseCategoricalCrossentropy(True), [no_padding_accuracy])
    last_iteration = 0
    weight_files = sorted([x.name for x in params_dir.glob(model.name + '-*.h5')])
    if len(weight_files) > 0:
        last_weights_file = str(params_dir.joinpath(weight_files[-1]).absolute())
        last_iteration = int(last_weights_file.split('-')[-1].split('.')[0])
        model.load_weights(last_weights_file)
    return model, last_iteration


def train(data_dir, params_dir, epochs, batch_size, early_stop):
    assert isinstance(data_dir, Path)
    assert isinstance(params_dir, Path)
    assert isinstance(epochs, int)
    assert isinstance(batch_size, int)
    assert isinstance(early_stop, int)
    train_file_paths = [str(p.absolute()) for p in data_dir.glob('*train*.txt')]
    val_file_paths = [str(p.absolute()) for p in data_dir.glob('*val*.txt')]
    train_dataset = tf.data.TextLineDataset(train_file_paths).repeat()\
        .map(tf_data_processing, tf.data.experimental.AUTOTUNE).batch(batch_size, True)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = tf.data.TextLineDataset(val_file_paths).repeat()\
        .map(tf_data_processing, tf.data.experimental.AUTOTUNE).batch(batch_size, True)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    train_steps = tf.data.TextLineDataset(train_file_paths).batch(batch_size)\
        .reduce(0, lambda old, new: old + 1).numpy()
    val_steps = tf.data.TextLineDataset(val_file_paths).batch(batch_size)\
        .reduce(0, lambda old, new: old + 1).numpy()
    model, last_iteration = get_model(params_dir)
    model.fit(train_dataset, steps_per_epoch=train_steps, epochs=epochs, validation_data=val_dataset,
              validation_steps=val_steps, initial_epoch=last_iteration,  # TODO: Think about the classes and their weights.
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=early_stop, verbose=1, restore_best_weights=True,
                                                          monitor=MONITOR_VALUE),
                         tf.keras.callbacks.ModelCheckpoint(
                             str(params_dir.joinpath(model.name+'-{epoch:03d}.h5').absolute()), save_best_only=True,
                             save_weights_only=True, monitor=MONITOR_VALUE
                         ),
                         tf.keras.callbacks.TerminateOnNaN(), tf.keras.callbacks.TensorBoard(str(data_dir.absolute()))]
              )


def test(data_dir, params_dir, batch_size):
    assert isinstance(data_dir, Path)
    assert isinstance(params_dir, Path)
    test_file_paths = [str(data_dir.joinpath(p)) for p in data_dir.glob('*test*.txt')]
    test_dataset = tf.data.TextLineDataset(test_file_paths).repeat()\
        .map(tf_data_processing, tf.data.experimental.AUTOTUNE).batch(batch_size, True).prefetch(1)
    test_steps = tf.data.TextLineDataset(test_file_paths).batch(batch_size)\
        .reduce(0, lambda old, new: old + 1).numpy()
    model, last_iteration = get_model(params_dir)
    print(model.evaluate(test_dataset, steps=test_steps))
