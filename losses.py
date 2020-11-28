import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy


class BinaryCrossentropyWithInputs(BinaryCrossentropy):

    def call(self, y_true, y_pred):
        y_pred = y_pred[:, tf.shape(y_true)[1]:]
        return super(BinaryCrossentropyWithInputs, self).call(y_true, y_pred)


class SparseCategoricalCrossentropyWithInputs(SparseCategoricalCrossentropy):

    def call(self, y_true, y_pred):
        y_pred = y_pred[:, tf.shape(y_true)[1]:]
        return super(SparseCategoricalCrossentropyWithInputs, self).call(y_true, y_pred)


# class WeightedBinaryCrossentropy(BinaryCrossentropy):
#
#     def __init__(self, class_weights, **kwargs):
#         super(WeightedBinaryCrossentropy, self).__init__(**kwargs)
#         self.weights = tf.convert_to_tensor(class_weights)
#
#     def __call__(self, y_true, y_pred, sample_weight=None):
#         sample_weight = tf.where(y_true == 0, self.weights[0], self.weights[1])
#         return super(WeightedBinaryCrossentropy, self).__call__(y_true, y_pred, sample_weight)
#
#
# class WeightedSparseCategoricalCrossentropy(SparseCategoricalCrossentropy):
#
#     def __init__(self, class_weights, **kwargs):
#         super(WeightedSparseCategoricalCrossentropy, self).__init__(**kwargs)
#         self.weights = tf.convert_to_tensor(class_weights)
#
#     def __call__(self, y_true, y_pred, sample_weight=None):
#         sample_weight = tf.gather(self.weights, tf.cast(y_true, tf.int32))
#         return super(WeightedSparseCategoricalCrossentropy, self).__call__(y_true, y_pred, sample_weight)
