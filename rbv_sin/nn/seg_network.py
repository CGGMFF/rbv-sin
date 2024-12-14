
import tensorflow as tf

from rbv_sin.nn.sin_factory import SINBasicUNET

@tf.keras.utils.register_keras_serializable(package="rsin_package") 
class VesselSegNetwork(tf.keras.Model):
    """Custom vessel segmentation network utilising the SIN architecture."""

    def __init__(self, base_input_shape, base_filters, *args, batch_norm=False, trainable=True, **kwargs):
        """
        Initialises and defines the vessel segmentation network.

        Arguments:
        - 'base_input_shape' - The input shape of the network.
        - 'base_filters' - The numebr of filters in the first layer.
        - 'batch_norm' - Whether the convolutions should be defined with batch normalisation.
        """
        super().__init__(*args, trainable=trainable, **kwargs)
        self.base_input_shape = base_input_shape
        self.base_filters = base_filters
        self.batch_norm = batch_norm
        self._define()

    def _define(self):
        """Defines the model with SIN architecture extended by an output convolution."""
        self.common_unet = SINBasicUNET(self.base_input_shape, self.base_filters, batch_norm=self.batch_norm, trainable=self.trainable)
        self.out_conv = tf.keras.layers.Conv2D(1, 1, 1, "valid", activation=tf.nn.sigmoid, dilation_rate=(1, 1), trainable=self.trainable)

    def call(self, input_tensor, training=False):
        x = self.common_unet(input_tensor, training=training)
        x = self.out_conv(x, training=training)
        return x
