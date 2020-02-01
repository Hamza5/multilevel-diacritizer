from pathlib import Path

import tensorflow as tf

from processing import DIACRITICS, NUMBER, FOREIGN, UNKNOWN, NUMBER_PATTERN

DATASET_FILE_NAME = 'Tashkeela-processed.zip'


@tf.function
def tf_separate_diacritics(d_text: tf.string):
    letter_diacritic = tf.strings.split(
        tf.strings.regex_replace(d_text, r'(.[' + ''.join(DIACRITICS) + ']*)', r'\1&'), '&'
    )
    decoded = tf.strings.unicode_decode(letter_diacritic, 'UTF-8')
    letters, diacritics = decoded[:, :1], decoded[:, 1:]
    return [tf.strings.unicode_encode(x, 'UTF-8') for x in (letters, diacritics)]


@tf.function
def tf_merge_diacritics(letters: tf.string, diacritics: tf.string):
    merged_letters_diacritics = tf.strings.reduce_join((letters, diacritics), 0)
    return tf.strings.reduce_join(merged_letters_diacritics, 0)


@tf.function
def tf_normalize_entities(text: tf.string):
    return tf.strings.strip(tf.strings.regex_replace(tf.strings.regex_replace(tf.strings.regex_replace(
        tf.strings.regex_replace(text, NUMBER_PATTERN.pattern, '0'),
        r'([^\p{Arabic}\p{P}\s'+''.join(DIACRITICS)+'])+', FOREIGN),
        r'\p{P}+', ''), r'\s{2,}', ' '))


def download_data(data_dir, download_url):
    assert isinstance(data_dir, Path)
    assert isinstance(download_url, str)
    data_dir = data_dir.expanduser()
    dataset_file_path = data_dir.joinpath(DATASET_FILE_NAME)
    tf.keras.utils.get_file(str(dataset_file_path.absolute()), download_url, cache_dir=str(data_dir.absolute()),
                            cache_subdir=str(data_dir.absolute()), extract=True)


def train(data_dir, params_dir):
    assert isinstance(data_dir, Path)
    assert isinstance(params_dir, Path)
    # train_file_paths = [str(data_dir.joinpath(p)) for p in data_dir.glob('*train*.txt')]
    val_file_paths = [str(data_dir.joinpath(p)) for p in data_dir.glob('*val*.txt')]
    # test_file_paths = [str(data_dir.joinpath(p)) for p in data_dir.glob('*test*.txt')]
    val_dataset = tf.data.TextLineDataset(val_file_paths).map(tf_normalize_entities).map(tf_separate_diacritics).map(tf_merge_diacritics)
    it = iter(val_dataset)
    for i in range(5):
        s = next(it)
        print(s.numpy().decode())
