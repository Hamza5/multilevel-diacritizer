from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

from tf_functions import CHARS
from losses import WeightedBinaryCrossEntropy, WeightedSparseCategoricalCrossEntropy

DEFAULT_WINDOW_SIZE = 32
DEFAULT_SLIDING_STEP = 5
DEFAULT_EMBEDDING_SIZE = 64
DEFAULT_LSTM_SIZE = 128
DEFAULT_DROPOUT_RATE = 0.1


class MultiLevelDiacritizer(Model):

    def __init__(self, window_size=DEFAULT_WINDOW_SIZE, lstm_size=DEFAULT_LSTM_SIZE, dropout_rate=DEFAULT_DROPOUT_RATE,
                 embedding_size=DEFAULT_EMBEDDING_SIZE, name=None, **kwargs):
        inputs = Input(shape=(window_size,), name='input')
        embedding = Embedding(len(CHARS) + 1, embedding_size, name='embedding')(inputs)

        primary_diacritics_layer = Bidirectional(LSTM(lstm_size, dropout=dropout_rate, return_sequences=True),
                                                 name='primary_diacritics_layer')(embedding)
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


if __name__ == '__main__':
    import numpy as np
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN
    from tf_functions import get_processed_window_dataset

    train_set, train_steps, diacritics_count = get_processed_window_dataset(['data/ATB3_train.txt'], 1024,
                                                                            DEFAULT_WINDOW_SIZE, DEFAULT_SLIDING_STEP)
    diacritics_factors = [np.max(x) / x for x in diacritics_count]
    diacritics_factors = [x / np.sum(x) for x in diacritics_factors]
    val_set, val_steps, _ = get_processed_window_dataset(['data/ATB3_val.txt'], 1024, DEFAULT_WINDOW_SIZE,
                                                         DEFAULT_SLIDING_STEP)
    epochs = 1000

    model = MultiLevelDiacritizer()
    model.summary()
    # for x, y in train_set:
    #     p_pri, p_sec, p_sh, p_su = model(x)
    #     y_pri, y_sec, y_sh, y_su = y
    #     print(tf.shape(p_pri), tf.shape(y_pri))
    #     print(tf.shape(p_sec), tf.shape(y_sec))
    #     print(tf.shape(p_sh), tf.shape(y_sh))
    #     print(tf.shape(p_su), tf.shape(y_su))
    #     break
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
    model.fit(train_set.repeat(epochs), steps_per_epoch=train_steps, epochs=epochs,
              validation_data=val_set.repeat(epochs), validation_steps=val_steps,
              callbacks=[ModelCheckpoint(f'params/{model.name}.h5', save_best_only=True), TerminateOnNaN()])
