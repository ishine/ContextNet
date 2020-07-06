import tensorflow as tf


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, activation=True):
        """ filters     : int
            kernel_size : int
            strides     : int
        """
        assert padding == "same", "padding = '%s' not implemented?" % padding
        super(ConvLayer, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation

        if strides == 1:
            self.conv = tf.keras.layers.SeparableConv1D(self.filters,
                                                        self.kernel_size,
                                                        self.strides, self.padding)
        else:
            self.conv = tf.keras.layers.SeparableConv1D(self.filters,
                                                        self.kernel_size,
                                                        self.strides, "valid")
        # Batch normalization variables
        # TODO Bias variables in previous conv layer is redundant
        self.beta = tf.Variable(tf.zeros(self.filters), trainable=True)
        self.gamma = tf.Variable(tf.ones(self.filters), trainable=True)

        self.population_mean = tf.Variable(tf.zeros(self.filters), trainable=False)
        self.population_variance = tf.Variable(tf.ones(self.filters), trainable=False)

    # https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    def _batch_norm(self, x, x_len, momentum=0.99, epsilon=0.001, training=None):
        if training:
            num_timesteps = tf.cast(tf.reduce_sum(x_len), dtype=tf.float32)
            batch_size = tf.shape(x)[0]
            x_reshape = tf.reshape(x, [-1, self.filters])

            # TODO Any gradient issue here? Need to check paper
            batch_mean = tf.math.reduce_sum(x_reshape, axis=0) / num_timesteps
            batch_variance = tf.math.squared_difference(x_reshape, batch_mean)
            batch_variance = batch_variance * tf.reshape(tf.sequence_mask(x_len, dtype=tf.float32), [-1, 1])
            batch_variance = tf.math.reduce_sum(batch_variance, axis=0) / num_timesteps

            # Update population statistics
            self.population_mean = self.population_mean * momentum + batch_mean * (1 - momentum)
            self.population_variance = self.population_variance * momentum + batch_variance * (1 - momentum)
            x = tf.nn.batch_normalization(x, batch_mean, batch_variance,
                                          self.beta, self.gamma, epsilon)
        else:
            # TODO Can population mean and variance be issue for initial validation steps?
            x = tf.nn.batch_normalization(x, population_mean, population_variance,
                                          self.beta, self.gamma, epsilon)
        return x * tf.expand_dims(tf.sequence_mask(x_len, dtype=tf.float32), -1)

    def _convolution(self, x, x_len):
        """ SeparableConv1D for padded input and "same" padding
            Not verified / tested for "valid" padding
        """
        if self.strides > 1:
            final_timesteps = tf.cast(tf.math.ceil(x_len / self.strides), dtype="int32")
            required_length = self.strides * (final_timesteps - 1) + self.kernel_size
            num_padding = required_length - x_len
            left_padding = num_padding // 2
            right_padding = num_padding - left_padding
            max_left_padding = self.kernel_size // 2
            max_right_padding = self.kernel_size - max_left_padding

            # Zero padding
            batch_size = tf.shape(x)[0]
            feat_dim = tf.shape(x)[-1]
            x_max_padded = tf.concat([tf.zeros([batch_size, max_left_padding, feat_dim]),
                                     x,
                                     tf.zeros([batch_size, max_right_padding, feat_dim])], 1)
            max_required_length = tf.math.reduce_max(required_length)
            start_timesteps = max_left_padding - left_padding

            # Work around for x = x_max_padded[:, start_timesteps:, :]
            idx = tf.expand_dims(tf.range(max_required_length), 0) + tf.expand_dims(start_timesteps, 1)
            batch_id = tf.tile(tf.reshape(tf.range(batch_size), [batch_size, 1, 1]), [1, max_required_length, 1])
            idx = tf.concat((batch_id, tf.reshape(idx, [batch_size, -1, 1])), -1)
            x = tf.gather_nd(x_max_padded, idx)

        x_len = tf.cast(tf.math.ceil(x_len / self.strides), dtype="int32")
        mask = tf.expand_dims(tf.sequence_mask(x_len, dtype=tf.float32), -1)
        x = mask * self.conv(x)
        return x, x_len

    def call(self, x, x_len, training=None):
        """ x : (B, T, F) """
        x, x_len = self._convolution(x, x_len)
        x = self._batch_norm(x, x_len, training=training)
        if self.activation:
            x = tf.nn.swish(x)
        return x, x_len


class SELayer(tf.keras.layers.Layer):
    def __init__(self, num_units):
        """ num_units : [int] """
        super(SELayer, self).__init__()
        self.num_units = num_units
        self.fc_layers = []

        for num_unit in num_units:
            self.fc_layers.append(tf.keras.layers.Dense(num_unit))

    def call(self, x, x_len):
        """ x : (B, T, F) """

        x_orig = x

        # TODO Assumption that previous layers output masked sequence
        x = tf.reduce_sum(x, axis=1) / tf.expand_dims(tf.cast(x_len, tf.float32), 1)
        for i in range(len(self.num_units)):
            x = tf.nn.swish(self.fc_layers[i](x))

        x = tf.expand_dims(tf.nn.sigmoid(x), 1)
        return x * x_orig


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, se_units, num_layers, filters,
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
        for stride in strides:
            self.conv_layers.append(ConvLayer(filters, kernel_size, stride, padding))

    def call(self, x, x_len, training=None):
        x_orig = x
        x_len_orig = x_len
        for conv_layer in self.conv_layers:
            x, x_len = conv_layer(x, x_len, training=training)

        if self.residual is None and self.se_layer is None:
            return x
        if self.se_layer is not None:
            x = self.se_layer(x, x_len)
        if self.residual is not None:
            x = x + self.residual(x_orig, x_len_orig)[0]

        return tf.nn.swish(x), x_len


class AudioEncoder(tf.keras.layers.Layer):
    """ Transcription network in RNN-Transducer architecture """
    def __init__(self, create_conv_blocks):
        """ create_conv_blocks: function which returns [ConvBlock] """
        super(AudioEncoder, self).__init__()
        self.network = create_conv_blocks()

    def call(self, x, x_len, training=None):
        for conv_block in self.network:
            x, x_len = conv_block(x, x_len, training=training)
        return x, x_len


class LabelEncoder(tf.keras.layers.Layer):
    """ Prediction network in RNN-Transducer architecture """
    def __init__(self, num_layers, num_units, out_dim, num_vocab, embed_dim):
        """ num_layers : int
            num_units  : int
            out_dim    : int
        """
        super(LabelEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.out_dim = out_dim

        self.embedding = tf.keras.layers.Embedding(num_vocab, embed_dim)
        self.network, self.projection = [], []
        for i in range(num_layers):
            self.network.append(tf.keras.layers.LSTM(num_units, return_sequences=True))
            self.projection.append(tf.keras.layers.Dense(out_dim))

    def call(self, y, y_len):
        y = self.embedding(y)
        for i in range(self.num_layers):
            mask = tf.sequence_mask(y_len)
            y = self.projection[i](self.network[i](y, mask=mask))
        return y


class ContextNet(tf.keras.Model):
    """ Network combining output of transcription and prediction networks """
    def __init__(self, num_units, num_vocab, create_conv_blocks,
                 num_lstms, lstm_units, out_dim):
        super(ContextNet, self).__init__()
        self.num_units = num_units
        self.num_vocab = num_vocab
        self.audio_encoder = AudioEncoder(create_conv_blocks)
        self.label_encoder = LabelEncoder(num_lstms, lstm_units,
                                          out_dim, num_vocab, num_units)

        self.projection = tf.keras.layers.Dense(num_units)
        self.output_layer = tf.keras.layers.Dense(num_vocab + 1)

    def call(self, x, y, x_len, y_len, training=None):
        """ x : (B, T, F)
            y : (B, U, F')
            x_len : (B,)
            y_len : (B,)
        """
        x, x_len = audio_encoder(x, x_len, training=training)
        y = label_encoder(y, y_len)

        x = tf.expand_dims(x, 2)
        y = tf.expand_dims(y, 1)

        z = tf.nn.tanh(self.projection(x + y))
        return self.output_layer(z), x_len, y_len
