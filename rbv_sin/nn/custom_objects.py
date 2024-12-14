
import tensorflow as tf

class MaskIoU(tf.metrics.Mean):
    """
    Computes the intersection over union (IoU) metric for two masks.
    This class is usable in tensorflow model metric list.
    """

    def __init__(self, target_shape, name="mean", dtype=None) -> None:
        """
        Initialises the parameters for the metric.

        Arguments:
        - 'target_shape' - The shape of the masks which are compared.
        - 'name' - Name of the metric.
        - 'dtype' - tf.metrics.Mean argument dtype.
        """
        super().__init__(name, dtype)
        self.target_shape = target_shape
        self.pixel_count = self.target_shape[0] * self.target_shape[1]

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the state of the metric by computing iou from two masks.
        """
        y_true_mask = tf.reshape(tf.math.round(y_true) == 1, [-1, self.pixel_count])
        y_pred_mask = tf.reshape(tf.math.round(y_pred) == 1, [-1, self.pixel_count])

        intersection_mask = tf.math.logical_and(y_true_mask, y_pred_mask)
        union_mask = tf.math.logical_or(y_true_mask, y_pred_mask)

        intersection = tf.reduce_sum(tf.cast(intersection_mask, tf.float32), axis=1)
        union = tf.reduce_sum(tf.cast(union_mask, tf.float32), axis=1)

        iou = tf.where(union == 0, 1., intersection / union)
        return super().update_state(iou, sample_weight)
    
    def get_config(self):
        """
        Returns a dictionary describing the parameters of this metric.
        """
        base_config = super(MaskIoU, self).get_config()
        base_config["target_shape"] = self.target_shape
        return base_config

    @classmethod
    def from_config(cls, config):
        """
        Constructs this metric from the given dictionary.
        """
        return cls(**config)
