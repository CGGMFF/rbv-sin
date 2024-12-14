
import tensorflow as tf

from rbv_sin.nn.sin_factory import SINBasicUNET

@tf.keras.utils.register_keras_serializable(package="rsin_package") 
class VesselInpGenerator(tf.keras.Model):
    """Custom vessel inpainting network utilising the SIN architecture."""

    def __init__(self, base_input_shape, base_filters, *args, batch_norm=False, trainable=True, **kwargs):
        """
        Initialises and defines the vessel inpainting network.

        Arguments:
        - 'base_input_shape' - The input shape of the inpainting network.
        - 'base_filters' - The numebr of filters in the first convolution layer.
        """
        super().__init__(*args, trainable=trainable, **kwargs)
        self.base_input_shape = base_input_shape
        self.base_filters = base_filters
        self.batch_norm = batch_norm
        self._define()

    def _define(self):
        """Defines the inpainting network as the SIN network extended by an output convolution."""
        self.common_unet = SINBasicUNET(self.base_input_shape, self.base_filters, batch_norm=self.batch_norm, trainable=self.trainable, activation=tf.nn.relu)
        self.out_conv = tf.keras.layers.Conv2D(3, 1, 1, "valid", activation=None, dilation_rate=(1, 1), trainable=self.trainable)

    def call(self, input_tensor, training=False):
        x = self.common_unet(input_tensor, training=training)
        x = self.out_conv(x, training=training)
        return x
