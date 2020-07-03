import tensorflow as tf

class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, activation):
        """ filters     : int
            kernel_size : int
            strides     : int
        """
        super(ConvLayer, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.conv = tf.keras.layers.SeparableConv1D(self.filters,
                                                    self.kernel_size,
                                                    self.strides, self.padding)
    def call(self, x, training):
        """ x : (B, T, F) """

        # TODO Check batch norm params and train/valid as well as padding
        x = tf.layers.keras.BatchNormalization(self.conv(x))
        if activation:
            x = tf.nn.swish(x)
        return x


class SELayer(tf.keras.layers.Layer):
    def __init__(self, num_units):
        """ num_units : [int] """
        super(SELayer, self).__init__()
        self.num_units = num_units
        self.fc_layers = []

        for i in range(len(num_units)):
            self.fc_layers.append(tf.keras.layers.Dense(num_units[i]))

    def call(self, x):
        """ x : (B, T, F) """

        x_orig = x
        shape = tf.shape(x)

        for i in range(len(num_units)):
            x = tf.nn.swish(fc_layers[i](x))

        x = tf.reshape(tf.nn.sigmoid(x), shape)
        return x * x_orig


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, se_units, filters,
                 kernel_size, strides, padding="same", residual=True):
        super(ConvBlock, self).__init__()
        self.num_layers = num_layers
        self.se_layer = SELayer(se_units) if se_units else None
        self.residual = ConvLayer(filters, kernel_size,
                                  strides, padding, activation=False) if residual else None

        self.conv_layers = []
        strides = [strides] + [1] * (num_layers - 1)
        for i, stride in enumerate(strides):
            self.conv_layers.append(ConvLayer(filters, kernel_size, stride, padding))

    def call(self, x, training):
        x_orig = x
        for i in range(num_layers):
            x = self.conv_layers(x, training)

        if self.residual is None and self.se_layer is None:
            return x
        if self.se_layer is not None:
            x = self.se_layer(x)
        if self.residual is not None:
            x = x + self.residual(x_orig)

        return tf.nn.swish(x)


class AudioEncoder(tf.keras.layers):
    """ Audio encoder in RNN-Transducer architecture """
    pass


class LabelEncoder(tf.keras.layers):
    """ Label encoder in RNN-Transducer architecture """
    pass


class JointNetwork(tf.keras.layers):
    """ Network combining output of audio and label encoders """
    pass
