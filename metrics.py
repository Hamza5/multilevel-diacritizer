import tensorflow as tf

from constants import NUMBER, ENCODE_LETTERS_TABLE


class ErrorRate(tf.keras.metrics.Metric):

    def __init__(self, function, name=None, **kwargs):
        super(ErrorRate, self).__init__(name=name, **kwargs)
        self.function = function
        self.value_sum = self.add_weight(f'{name or __class__.__name__}_sum', initializer=tf.zeros_initializer())
        self.count = self.add_weight('count', initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, x):
        y_pred = tf.argmax(y_pred, axis=2)
        y_true, y_pred, x = tf.cast(y_true, tf.int32), tf.cast(y_pred, tf.int32), tf.cast(x, tf.int32)
        batch_value = tf.map_fn(self.function, (y_true, y_pred, x), fn_output_signature=tf.float32)
        self.value_sum.assign_add(tf.reduce_mean(batch_value))
        self.count.assign_add(1)

    def result(self):
        return 1 - (self.value_sum / self.count)


class DiacritizationErrorRate(ErrorRate):

    @staticmethod
    def char_acc(y_true__y_pred__x):
        y_true, y_pred, x = y_true__y_pred__x
        letters_positions = tf.reduce_all(
            x != tf.reshape(tf.concat(([0], ENCODE_LETTERS_TABLE.lookup(tf.constant([' ', NUMBER]))), axis=0), (-1, 1)),
            axis=0)
        return tf.reduce_mean(tf.cast(y_true[letters_positions] == y_pred[letters_positions], tf.float32))

    def __init__(self, name='DER', **kwargs):
        super(DiacritizationErrorRate, self).__init__(self.char_acc, name=name, **kwargs)


class WordErrorRate(ErrorRate):

    @staticmethod
    def word_acc(y_true__y_pred__x):

        initial = (tf.constant(True), tf.constant(True), tf.constant(0.0), tf.constant(0.0))

        def count_correct_words(last_state, y_true__y_pred__x):
            y_true, y_pred, x = y_true__y_pred__x
            last_was_letter, comparison, words_count, correct_count = last_state
            if x > ENCODE_LETTERS_TABLE.lookup(tf.constant(NUMBER)):
                comparison = tf.logical_and(comparison, y_true == y_pred)
                last_was_letter = tf.logical_or(last_was_letter, True)
            elif last_was_letter:
                correct_count = correct_count + tf.cast(comparison, tf.float32)
                words_count = words_count + 1.0
                comparison = tf.logical_or(comparison, True)
                last_was_letter = tf.logical_and(last_was_letter, False)
            return last_was_letter, comparison, words_count, correct_count

        _, _, words_count, correct_count = tf.foldl(count_correct_words, y_true__y_pred__x, initial)
        return correct_count / words_count

    def __init__(self, name='WER', **kwargs):
        super(WordErrorRate, self).__init__(self.word_acc, name=name, **kwargs)
