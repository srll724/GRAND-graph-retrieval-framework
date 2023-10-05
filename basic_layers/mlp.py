import tensorflow as tf

KL = tf.keras.layers
KA = tf.keras.activations


class MLP(KL.Layer):
    def __init__(self,
                 output_sizes,
                 w_init=None,
                 b_init=None,
                 use_bias=True,
                 activation="relu",
                 dropout_rate=None,
                 activate_final=False,
                 name=None):
        if not use_bias and b_init is not None:
            raise ValueError("When with_bias=False, b_init must not be set.")

        super(MLP, self).__init__(name=name)
        self.output_sizes = output_sizes
        self.w_init = w_init or "glorot_uniform"
        self.b_init = b_init or "zeros"
        self.use_bias = use_bias
        self.activation = KA.get(activation)
        self.dropout_rate = dropout_rate
        self.activate_final = activate_final

        self.layers = []
        for i, size in enumerate(self.output_sizes):
            self.layers.append(
                KL.Dense(
                    units=size,
                    activation=None,
                    kernel_initializer=self.w_init,
                    bias_initializer=self.b_init,
                    use_bias=self.use_bias,
                    name=f"dense_{i}"))

    def call(self, inputs, training):
        use_dropout = self.dropout_rate not in (None, 0)

        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs = layer(outputs)
            if (i < len(self.output_sizes) - 1) or self.activate_final:
                if training and use_dropout:
                    outputs = tf.nn.dropout(outputs, rate=self.dropout_rate)
                outputs = self.activation(outputs)
        return outputs
