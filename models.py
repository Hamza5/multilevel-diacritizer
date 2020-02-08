from pathlib import Path

import tensorflow as tf
from transformers import TFXLNetModel, XLNetConfig

from processing import DIACRITICS, NUMBER, NUMBER_PATTERN, ARABIC_LETTERS

DATASET_FILE_NAME = 'Tashkeela-processed.zip'
SEQUENCE_LENGTH = 200  # TODO: Need to change this to a more accurate value.


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


# def generate_vocabulary_file(vocabulary_file_path: str, dataset: tf.data.Dataset):
#     tokens = {NUMBER, FOREIGN}
#     for w in dataset:
#         tks = w.numpy().decode('UTF-8').split()
#         tokens.update(tks)
#     with tf.io.gfile.GFile(vocabulary_file_path, 'w') as vocabulary_file:
#         for token in sorted(tokens):
#             print(token, file=vocabulary_file)
#
#
# def get_vocabulary_table(vocabulary_file_path: str):
#     vocabulary_table = tf.lookup.StaticVocabularyTable(
#         tf.lookup.TextFileInitializer(vocabulary_file_path, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE, tf.int64,
#                                       tf.lookup.TextFileIndex.LINE_NUMBER),
#         1)
#     return vocabulary_table


CHARS = sorted(ARABIC_LETTERS.union({NUMBER, ' '}))
DIACS = sorted(DIACRITICS - {'ّ'}) + sorted('ّ'+x for x in (DIACRITICS - {'ّ'}))
LETTERS_TABLE = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(tf.constant(CHARS), tf.range(1, len(CHARS)+1)), len(CHARS)+1
)
DIACRITICS_TABLE = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(tf.constant(DIACS), tf.range(1, len(DIACS)+1)), len(DIACS)+1
)


@tf.function
def tf_encode(letters: tf.string, diacritics: tf.string):
    return LETTERS_TABLE.lookup(letters), DIACRITICS_TABLE.lookup(diacritics)


@tf.function
def tf_pad_with_attention_mask(letters: tf.string, diacritics: tf.string):
    letters = tf.pad(letters, [[0, SEQUENCE_LENGTH]], constant_values=0)[:SEQUENCE_LENGTH]
    diacritics = tf.pad(diacritics, [[0, SEQUENCE_LENGTH]], constant_values=0)[:SEQUENCE_LENGTH]
    return (letters, letters > 0), diacritics


class XLNetDiacritizer(TFXLNetModel):

    def __init__(self, config, *inputs, **kwargs):
        super(XLNetDiacritizer, self).__init__(config, *inputs, **kwargs)
        self.classifier = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(DIACS) + 1, activation='softmax'),
                                                          input_shape=(SEQUENCE_LENGTH, self.config.d_model))

    def call(self, inputs, **kwargs):
        return self.classifier(super(XLNetDiacritizer, self).call(inputs, **kwargs)[0])


def download_data(data_dir, download_url):
    assert isinstance(data_dir, Path)
    assert isinstance(download_url, str)
    data_dir = data_dir.expanduser()
    dataset_file_path = data_dir.joinpath(DATASET_FILE_NAME)
    tf.keras.utils.get_file(str(dataset_file_path.absolute()), download_url, cache_dir=str(data_dir.absolute()),
                            cache_subdir=str(data_dir.absolute()), extract=True)


def train(data_dir, params_dir, epochs):
    assert isinstance(data_dir, Path)
    assert isinstance(params_dir, Path)
    assert isinstance(epochs, int)
    train_file_paths = [str(data_dir.joinpath(p)) for p in data_dir.glob('*train*.txt')]
    val_file_paths = [str(data_dir.joinpath(p)) for p in data_dir.glob('*val*.txt')]
    train_dataset = tf.data.TextLineDataset(train_file_paths).map(tf_normalize_entities).map(tf_separate_diacritics).map(tf_encode)
    val_dataset = tf.data.TextLineDataset(val_file_paths).map(tf_normalize_entities).map(tf_separate_diacritics).map(tf_encode)
    config = XLNetConfig.from_pretrained('xlnet-base-cased', cache_dir=str(params_dir.absolute()),
                                         vocab_size=len(CHARS))
    model = XLNetDiacritizer(config)
    print(model)
    first = lambda x, y: x
    example_inputs = val_dataset.skip(5).take(3).map(tf_pad_with_attention_mask).map(first).batch(3)
    # print(next(iter(example_inputs)))
    outputs = model.predict(example_inputs)
    print(outputs)
    print(outputs.shape)
