
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="rsin_package") 
class Conv2DWithBN(tf.keras.layers.Layer):
    """Custom convolution layer utilising batch normalisation."""

    def __init__(self, filters, kernel_size, strides, padding, activation, dilation_rate=(1, 1), use_bias=False, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        """Defines a normal convolution layer followed by batch normalisation."""
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, activation=None, dilation_rate=dilation_rate, use_bias=use_bias)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation(activation)

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor, training=training)
        x = self.batchnorm(x, training=training)
        x = self.activation(x, training=training)
        return x
    
@tf.keras.utils.register_keras_serializable(package="rsin_package")
class Conv2DTransposeWithBN(tf.keras.layers.Layer):
    """Custom transposed convolution layer utilising batch normalisation."""

    def __init__(self, filters, kernel_size, strides, padding, activation, dilation_rate=(1, 1), use_bias=False, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        """Defines a transposed convolution layer followed by batch normalisation."""
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.conv = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding, activation=None, dilation_rate=dilation_rate, use_bias=use_bias)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation(activation)

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor, training=training)
        x = self.batchnorm(x, training=training)
        x = self.activation(x, training=training)
        return x
    
@tf.keras.utils.register_keras_serializable(package="rsin_package")
class Conv2DOptBN(tf.keras.layers.Layer):
    """Custom convolution layer implementing optional batch normalisation."""

    def __init__(self, filters, kernel_size, strides, padding, activation, dilation_rate=(1, 1), use_bias=False, batch_norm=False, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        """Defines a traditional convolution layer that is optionally followed by batch noralisation if 'batch_norm' is true."""
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        if batch_norm:
            self.conv = Conv2DWithBN(filters, kernel_size, strides, padding, activation, dilation_rate=dilation_rate, use_bias=use_bias, trainable=self.trainable)
        else:
            self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, activation=activation, dilation_rate=dilation_rate, trainable=self.trainable)

    def call(self, input_tensor, training=False):
        return self.conv(input_tensor, training=training)
    
@tf.keras.utils.register_keras_serializable(package="rsin_package")
class Conv2DTransposeOptBN(tf.keras.layers.Layer):
    """Custom transposed convolution layer implementing optional batch normalisation."""

    def __init__(self, filters, kernel_size, strides, padding, activation, dilation_rate=(1, 1), use_bias=False, batch_norm=False, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        """Defines a traditional transposed convolution that is followed by batch normalisation if 'batch_norm' is true."""
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        if batch_norm:
            self.conv = Conv2DTransposeWithBN(filters, kernel_size, strides, padding, activation, dilation_rate=dilation_rate, use_bias=use_bias, trainable=self.trainable)
        else:
            self.conv = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding, activation=activation, dilation_rate=dilation_rate, trainable=self.trainable)

    def call(self, input_tensor, training=False):
        return self.conv(input_tensor, training=training)

@tf.keras.utils.register_keras_serializable(package="rsin_package")
class SINSelfAttention(tf.keras.layers.Layer):
    """Class defining the SIN attention layer."""

    def __init__(self, ca_shape, filters, inner_filters, kernel_size, strides, padding, activation, batch_norm=False, trainable=True, **kwargs):
        """
        Initialises and defines the SIN multi-head positional self-attention layer.

        Arguments:
        - 'ca_shape' - Attention input shape.
        - 'filters' - The number of output filters.
        - 'inner_filters' - The number of filters in the key, query and value convolutions.
        - 'kernel_size' - The size of the convolution kernels in the attention.
        - 'strides' - The stride of the convolutions in the attention - this should be 1.
        - 'padding' - Padding of the convolutions.
        - 'activation' - Activation applied to convolutions outside of the query and key.
        - 'batch_norm' - Whether the convolutions should be followed by batch normalisation.
        """
        super().__init__(trainable, **kwargs)
        self.ca_shape = ca_shape
        self.conv_key = Conv2DOptBN(inner_filters, kernel_size, strides, padding, tf.nn.elu, batch_norm=batch_norm, trainable=trainable)
        self.conv_query = Conv2DOptBN(inner_filters, kernel_size, strides, padding, tf.nn.elu, batch_norm=batch_norm, trainable=trainable)
        self.conv_value = Conv2DOptBN(inner_filters, kernel_size, strides, padding, activation, batch_norm=batch_norm, trainable=trainable)
        self.flattened_shape = [ca_shape[0] * ca_shape[1], inner_filters]

        self.key_reshape = tf.keras.layers.Reshape(self.flattened_shape)
        self.query_reshape = tf.keras.layers.Reshape(self.flattened_shape)
        self.value_reshape = tf.keras.layers.Reshape(self.flattened_shape)
        self.mul_query_key = tf.keras.layers.Dot(axes=(2, 2))
        # =======================
        self.mul_matrix_ident = tf.keras.layers.Multiply()
        # =======================
        self.div_matrix = tf.keras.layers.Lambda(lambda x: x / tf.sqrt(float(self.flattened_shape[1])))
        self.softmax = tf.keras.layers.Softmax()
        self.mul_matrix_value = tf.keras.layers.Dot(axes=(2, 1))
        self.result_reshape = tf.keras.layers.Reshape([ca_shape[0], ca_shape[1], inner_filters])
        
        self.conv_result = Conv2DOptBN(filters, kernel_size, strides, padding, activation, batch_norm=batch_norm, trainable=trainable)

    def call(self, input_tensor, training=False):
        x_key = self.conv_key(input_tensor, training=training)
        x_query = self.conv_query(input_tensor, training=training)
        x_value = self.conv_value(input_tensor, training=training)

        x_key = self.key_reshape(x_key, training=training)
        x_query = self.query_reshape(x_query, training=training)
        x_value = self.value_reshape(x_value, training=training)
        x = self.mul_query_key([x_query, x_key], training=training)
        # ===========================
        # Multiply the diagonal elements with zero.
        one_minus_ident = 1.0 - tf.eye(self.flattened_shape[0], self.flattened_shape[0], [input_tensor.shape[0]])
        x = self.mul_matrix_ident([x, one_minus_ident], training=training)
        # ===========================
        x = self.div_matrix(x, training=training)
        x = self.softmax(x, training=training)
        x = self.mul_matrix_value([x, x_value], training=training)
        x = self.result_reshape(x, training=training)
        x = self.conv_result(x, training=training)
        return x

@tf.keras.utils.register_keras_serializable(package="rsin_package")
class SINBasicUNET(tf.keras.Model):
    """Custom definition of the SIN common architecture."""

    def __init__(self, base_input_shape, base_filters, *args, batch_norm=False, trainable=True, activation=tf.nn.relu, **kwargs):
        """
        Initialises and defines the architecture.

        Arguments:
        - 'base_input_shape' - The input shape for the network.
        - 'base_filters' - The number of filters at the first layer.
        """
        super().__init__(*args, trainable=trainable, **kwargs)
        self.base_input_shape = base_input_shape
        self.base_filters = base_filters
        self.batch_norm = batch_norm
        self.activation = activation
        self._define()

    def _define(self):
        """Defines the architecture."""
        # Downscaling layer definitions.
        self.input_conv_1 = Conv2DOptBN(1 * self.base_filters, 3, 1, "same", self.activation, batch_norm=self.batch_norm, dilation_rate=(1, 1))
        self.down_shape_1 = [self.base_input_shape[0] // 2, self.base_input_shape[1] // 2] + list(self.base_input_shape[2:])
        self.down_conv_1 = Conv2DOptBN(2 * self.base_filters, 3, 2, "same", self.activation, batch_norm=self.batch_norm, dilation_rate=(1, 1))
        self.down_shape_2 = [self.down_shape_1[0] // 2, self.down_shape_1[1] // 2] + list(self.base_input_shape[2:])
        self.down_conv_2 = Conv2DOptBN(4 * self.base_filters, 3, 2, "same", self.activation, batch_norm=self.batch_norm, dilation_rate=(1, 1))
        self.down_shape_3 = [self.down_shape_2[0] // 2, self.down_shape_2[1] // 2] + list(self.base_input_shape[2:])
        self.down_conv_3 = Conv2DOptBN(8 * self.base_filters, 3, 2, "same", self.activation, batch_norm=self.batch_norm, dilation_rate=(1, 1))
        self.down_shape_4 = [self.down_shape_3[0] // 2, self.down_shape_3[1] // 2] + list(self.base_input_shape[2:])
        self.down_conv_4 = Conv2DOptBN(16 * self.base_filters, 3, 2, "same", self.activation, batch_norm=self.batch_norm, dilation_rate=(1, 1))

        # Attention layer definitions.
        self.ca_down_2 = SINSelfAttention(self.down_shape_2, 4 * self.base_filters, 4 * self.base_filters, 1, 1, "same", self.activation, batch_norm=self.batch_norm, trainable=self.trainable)
        self.ca_down_2_concat = tf.keras.layers.Add()
        self.ca_down_2_conv = Conv2DOptBN(4 * self.base_filters, 3, 1, "same", self.activation, batch_norm=self.batch_norm, dilation_rate=(1, 1))
        self.ca_down_3 = SINSelfAttention(self.down_shape_3, 8 * self.base_filters, 8 * self.base_filters, 1, 1, "same", self.activation, batch_norm=self.batch_norm, trainable=self.trainable)
        self.ca_down_3_concat = tf.keras.layers.Add()
        self.ca_down_3_conv = Conv2DOptBN(8 * self.base_filters, 3, 1, "same", self.activation, batch_norm=self.batch_norm, dilation_rate=(1, 1))
        self.ca_down_4 = SINSelfAttention(self.down_shape_4, 16 * self.base_filters, 16 * self.base_filters, 1, 1, "same", self.activation, batch_norm=self.batch_norm, trainable=self.trainable)
        self.ca_down_4_concat = tf.keras.layers.Add()
        self.ca_down_4_conv = Conv2DOptBN(16 * self.base_filters, 3, 1, "same", self.activation, batch_norm=self.batch_norm, dilation_rate=(1, 1))

        # Straight pass layer definitions.
        self.straight_conv_1_1 = Conv2DOptBN(2 * self.base_filters, 3, 1, "same", self.activation, batch_norm=self.batch_norm, dilation_rate=(2, 2))
        self.straight_conv_2_1 = Conv2DOptBN(4 * self.base_filters, 3, 1, "same", self.activation, batch_norm=self.batch_norm, dilation_rate=(2, 2))
        self.straight_concat_2 = tf.keras.layers.Add()
        self.straight_conv_3_1 = Conv2DOptBN(8 * self.base_filters, 3, 1, "same", self.activation, batch_norm=self.batch_norm, dilation_rate=(2, 2))
        self.straight_concat_3 = tf.keras.layers.Add()
        self.straight_conv_4_1 = Conv2DOptBN(16 * self.base_filters, 3, 1, "same", self.activation, batch_norm=self.batch_norm, dilation_rate=(2, 2))
        self.straight_concat_4 = tf.keras.layers.Add()

        # Upscaling layer definitions.
        self.up_conv_1 = Conv2DTransposeOptBN(8 * self.base_filters, 3, 2, "same", self.activation, batch_norm=self.batch_norm, dilation_rate=(1, 1))
        self.up_concat_ca_1 = tf.keras.layers.Add()
        self.up_conv_ca_1 = Conv2DTransposeOptBN(8 * self.base_filters, 3, 1, "same", self.activation, batch_norm=self.batch_norm, dilation_rate=(1, 1))
        self.up_conv_2 = Conv2DTransposeOptBN(4 * self.base_filters, 3, 2, "same", self.activation, batch_norm=self.batch_norm, dilation_rate=(1, 1))
        self.up_concat_ca_2 = tf.keras.layers.Add()
        self.up_conv_ca_2 = Conv2DTransposeOptBN(4 * self.base_filters, 3, 1, "same", self.activation, batch_norm=self.batch_norm, dilation_rate=(1, 1))
        self.up_conv_3 = Conv2DTransposeOptBN(2 * self.base_filters, 3, 2, "same", self.activation, batch_norm=self.batch_norm, dilation_rate=(1, 1))
        self.up_concat_3 = tf.keras.layers.Add()
        self.up_conv_4 = Conv2DTransposeOptBN(1 * self.base_filters, 3, 2, "same", self.activation, batch_norm=self.batch_norm, dilation_rate=(1, 1))
        self.up_concat_4 = tf.keras.layers.Add()

        self.out_conv = Conv2DOptBN(1 * self.base_filters, 3, 1, "same", self.activation, batch_norm=self.batch_norm, dilation_rate=(1, 1))

    def call(self, input_tensor, training=False):
        # Downscaling convolutions
        x_input = self.input_conv_1(input_tensor, training=training)
        x_down_1 = self.down_conv_1(x_input, training=training)
        x_down_2 = self.down_conv_2(x_down_1, training=training)
        x_down_3 = self.down_conv_3(x_down_2, training=training)
        x_down_4 = self.down_conv_4(x_down_3, training=training)

        # =========================================================================================
        # Attention

        x_ca_2 = self.ca_down_2(x_down_2, training=training)
        x_ca_2 = self.ca_down_2_concat([x_ca_2, x_down_2], training=training)
        x_ca_2 = self.ca_down_2_conv(x_ca_2, training=training)

        x_ca_3 = self.ca_down_3(x_down_3, training=training)
        x_ca_3 = self.ca_down_3_concat([x_ca_3, x_down_3], training=training)
        x_ca_3 = self.ca_down_3_conv(x_ca_3, training=training)

        x_ca_4 = self.ca_down_4(x_down_4, training=training)
        x_ca_4 = self.ca_down_4_concat([x_ca_4, x_down_4], training=training)
        x_ca_4 = self.ca_down_4_conv(x_ca_4, training=training)

        # ========================================================================================
        # Straight pass

        x_down_1 = self.straight_conv_1_1(x_down_1, training=training)

        x_down_2 = self.straight_conv_2_1(x_down_2, training=training)
        x_ca_2 = self.straight_concat_2([x_down_2, x_ca_2], training=training)

        x_down_3 = self.straight_conv_3_1(x_down_3, training=training)
        x_ca_3 = self.straight_concat_3([x_down_3, x_ca_3], training=training)

        x_down_4 = self.straight_conv_4_1(x_down_4, training=training)
        x_ca_4 = self.straight_concat_4([x_down_4, x_ca_4], training=training)

        # ========================================================================================
        # Up convolutions

        x_up_1 = self.up_conv_1(x_ca_4, training=training)
        x_up_1 = self.up_concat_ca_1([x_up_1, x_ca_3], training=training)
        x_up_1 = self.up_conv_ca_1(x_up_1, training=training)

        x_up_2 = self.up_conv_2(x_up_1, training=training)
        x_up_2 = self.up_concat_ca_2([x_up_2, x_ca_2], training=training)
        x_up_2 = self.up_conv_ca_2(x_up_2, training=training)

        x_up_3 = self.up_conv_3(x_up_2, training=training)
        x_up_3 = self.up_concat_3([x_down_1, x_up_3], training=training)

        x_up_4 = self.up_conv_4(x_up_3, training=training)
        x_up_4 = self.up_concat_4([x_input, x_up_4], training=training)

        x_out = self.out_conv(x_up_4, training=training)
        return x_out

@tf.keras.utils.register_keras_serializable(package="rsin_package")
class SINVGG19(tf.keras.Model):
    """Custom model of VGG19 with multiple outputs intended for perceptual loss computation."""

    def __init__(self, base_input_shape, *args, trainable=False, **kwargs):
        """
        Defines the VGG19 with imagenet weights using tensorflow.

        Arguments:
        - 'base_input_shape' - Input shape for VGG19.
        """
        super().__init__(*args, trainable=trainable, **kwargs)
        self.base_input_shape = base_input_shape
        self._define()

    def _define(self):
        """Definition of the VGG 19 model with specific outputs."""
        self.vgg19 = tf.keras.applications.vgg19.VGG19(include_top=False, weights="imagenet", input_shape=self.base_input_shape)
        custom_outputs = [self.vgg19.get_layer("block1_conv2").output, self.vgg19.get_layer("block2_conv2").output, self.vgg19.get_layer("block3_conv4").output, self.vgg19.get_layer("block4_conv4").output]
        self.extraction_model = tf.keras.Model(inputs=self.vgg19.inputs, outputs=custom_outputs)

    def call(self, input_tensor, training=False):
        # Computes the custom VGG 19 outputs with the requried preprocessing.
        x = tf.keras.applications.vgg19.preprocess_input(input_tensor)
        x_out_1, x_out_2, x_out_3, x_out_4 = self.extraction_model(x, training=training)
        return x_out_1, x_out_2, x_out_3, x_out_4
