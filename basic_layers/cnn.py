import tensorflow as tf

KL = tf.keras.layers


class CNN(KL.Layer):
    def __init__(self,
                 filter_sizes,
                 kernel_sizes,
                 pool_sizes,
                 padding="SAME",
                 pool_type="max",
                 activation="relu",
                 use_bias=True,
                 activate_final=True,
                 name=None):
        super(CNN, self).__init__(name=name)
        assert len(filter_sizes) == len(kernel_sizes) == len(pool_sizes)

        self.filter_sizes = filter_sizes
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        self.padding = padding
        self.pool_type = pool_type
        self.activation = activation
        self.use_bias = use_bias
        self.activate_final = activate_final

        self.layers = []
        if self.pool_type == "max":
            pool_class = KL.MaxPooling2D
        else:
            raise ValueError
        for i, (fs, ks, ps) in enumerate(
                zip(self.filter_sizes, self.kernel_sizes, self.pool_sizes)):
            if i < len(self.filter_sizes) - 1 or self.activate_final:
                act = self.activation
            else:
                act = None
            conv_layer = KL.Conv2D(
                filters=fs,
                kernel_size=ks,
                activation=act,
                padding=self.padding,
                use_bias=self.use_bias,
                name=f"Conv2D_{i}")
            pool_layer = pool_class(
                pool_size=ps,
                padding=self.padding,
                name=f"Pool2D_{i}")
            self.layers.append((conv_layer, pool_layer))

    def call(self, inputs):
        outputs = inputs
        for conv_layer, pool_layer in self.layers:
            outputs = conv_layer(outputs)
            outputs = pool_layer(outputs)
        return outputs
