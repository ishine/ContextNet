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
    def call(self, x, x_len, training):
        """ x : (B, T, F) """

        # TODO Check batch norm params and train/valid as well as padding
        x = tf.layers.keras.BatchNormalization(self.conv(x))
        if activation:
            x = tf.nn.swish(x)
        return x, x_len


class SELayer(tf.keras.layers.Layer):
    def __init__(self, num_units):
        """ num_units : [int] """
        super(SELayer, self).__init__()
        self.num_units = num_units
        self.fc_layers = []

        for i in range(len(num_units)):
            self.fc_layers.append(tf.keras.layers.Dense(num_units[i]))

    def call(self, x, x_len):
        """ x : (B, T, F) """

        mask = tf.sequence_mask(x_len)
        x = x * mask
        x_orig = x

        x = tf.reduce_sum(x) / x_len
        for i in range(len(num_units)):
            x = tf.nn.swish(fc_layers[i](x))

        x = tf.expand_dims(tf.nn.sigmoid(x), 1)
        return x * x_orig


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, se_units, filters,
                 kernel_size, strides=1, padding="same", residual=True):
        """ num_layers : int
            se_units   : [int]
            filters    : int
            kernel_size: int
            strides    : int
            padding    : str
            residual   : boolean
        """
        super(ConvBlock, self).__init__()
        self.num_layers = num_layers
        self.se_layer = SELayer(se_units) if se_units else None
        self.residual = ConvLayer(filters, kernel_size,
                                  strides, padding, activation=False) if residual else None

        self.conv_layers = []
        strides = [strides] + [1] * (num_layers - 1)
        for i, stride in enumerate(strides):
            self.conv_layers.append(ConvLayer(filters, kernel_size, stride, padding))

    def call(self, x, x_len, training):
        x_orig = x
        x_len_orig = x_len
        for i in range(num_layers):
            x, x_len = self.conv_layers(x, x_len, training)

        if self.residual is None and self.se_layer is None:
            return x
        if self.se_layer is not None:
            x = self.se_layer(x, x_len)
        if self.residual is not None:
            x, _ = x + self.residual(x_orig, x_len_orig)

        return tf.nn.swish(x), x_len


class AudioEncoder(tf.keras.layers.Layer):
    """ Transcription network in RNN-Transducer architecture """
    def __init__(self, create_conv_blocks):
        """ create_conv_blocks: function which returns [ConvBlock] """
        super(AudioEncoder, self).__init__()
        self.network = create_conv_blocks()

    def call(self, x, x_len, training):
        for conv_block in self.network:
            x = conv_block(x, x_len, training)
        return x


class LabelEncoder(tf.keras.layers.Layer):
    """ Prediction network in RNN-Transducer architecture """
    def __init__(self, num_layers, num_units, out_dim):
        """ num_layers : int
            num_units  : int
            out_dim    : int
        """
        super(LabelEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.out_dim = out_dim

        self.network, self.projection = [], []
        for i in range(num_layers):
            self.network.append(tf.keras.layers.LSTM(num_units, return_sequences=True))
            self.projection.append(tf.keras.layers.Dense(out_dim))

    def call(self, y, y_len):
        for i in range(self.num_layers):
            mask = tf.sequence_mask(y_len)
            y = self.projection(self.network(y, mask))
        return y


class ContextNet(tf.keras.Model):
    """ Network combining output of transcription and prediction networks """
    def __init__(self, num_units, num_vocab, create_conv_blocks,
                 num_lstms, lstm_units, out_dim):
        super(ContextNet, self).__init__()
        self.num_units = num_units
        self.num_vocab = num_vocab
        self.audio_encoder = AudioEncoder(create_conv_blocks)
        self.label_encoder = LabelEncoder(num_lstms, lstm_units, out_dim)

        self.projection = tf.keras.layers.Dense(num_units)
        self.output = tf.keras.layers.Dense(num_vocab + 1)

    def call(self, x, y, x_len, y_len, training):
        """ x : (B, T, F)
            y : (B, U, F')
            x_len : (B,)
            y_len : (B,)
        """
        x = audio_encoder(x, x_len, training)
        y = label_encoder(y, y_len)

        x = tf.expand_dims(x, 2)
        y = tf.expand_dims(y, 1)

        z = tf.nn.tanh(self.projection(x + y))
        return self.output(z)
