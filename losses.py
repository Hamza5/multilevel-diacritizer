import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy


class WeightedBinaryCrossEntropy(BinaryCrossentropy):

    def __init__(self, class_weights, **kwargs):
        super(WeightedBinaryCrossEntropy, self).__init__(**kwargs)
        self.weights = tf.convert_to_tensor(class_weights)

    def __call__(self, y_true, y_pred, sample_weight=None):
        sample_weight = tf.where(y_true == 0, self.weights[0], self.weights[1])
        return super(WeightedBinaryCrossEntropy, self).__call__(y_true, y_pred, sample_weight)


class WeightedSparseCategoricalCrossEntropy(SparseCategoricalCrossentropy):

    def __init__(self, class_weights, **kwargs):
        super(WeightedSparseCategoricalCrossEntropy, self).__init__(**kwargs)
        self.weights = tf.convert_to_tensor(class_weights)

    def __call__(self, y_true, y_pred, sample_weight=None):
        sample_weight = tf.gather(self.weights, y_true)
        return super(WeightedSparseCategoricalCrossEntropy, self).__call__(y_true, y_pred, sample_weight)
