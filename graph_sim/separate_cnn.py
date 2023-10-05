import tensorflow as tf

from basic_layers import CNN

KL = tf.keras.layers


class SeparateCNN(KL.Layer):
    def __init__(self,
                 n_splits,
                 filter_sizes,
                 kernel_sizes,
                 pool_sizes,
                 padding="SAME",
                 pool_type="max",
                 activation="relu",
                 use_bias=True,
                 activate_final=True,
                 name=None):
        super(SeparateCNN, self).__init__(name=name)

        self.n_splits = n_splits
        self.filter_sizes = filter_sizes
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        self.padding = padding
        self.pool_type = pool_type
        self.activation = activation
        self.use_bias = use_bias
        self.activate_final = activate_final

        self.layers = []
        for i in range(self.n_splits):
            self.layers.append(CNN(
                filter_sizes=self.filter_sizes,
                kernel_sizes=self.kernel_sizes,
                pool_sizes=self.pool_sizes,
                padding=self.padding,
                pool_type=self.pool_type,
                activation=self.activation,
                use_bias=self.use_bias,
                activate_final=self.activate_final,
                name=f"CNN_{i}"))

    def call(self, inputs):
        splits = tf.split(inputs, num_or_size_splits=self.n_splits, axis=-1)
        outputs = []
        for i, layer in enumerate(self.layers):
            outputs.append(layer(splits[i]))
        outputs = tf.concat(outputs, axis=-1)
        outputs = tf.reshape(outputs, [outputs.shape[0], -1])
        return outputs


if __name__ == "__main__":
    from dataset import Dataset
    from basic_layers import GCN
    from graph_sim.pairwise_node_similarity import PairwiseNodeSimilarity

    ds = Dataset("aids700", 32, 0.25, "bfs",
                 "onehot", "type", "regression", 123)
    train_data = ds.get_data("training")
    graph_data = train_data[0]

    gcn = GCN([128, 64, 32])

    node_states, all_node_states = gcn(graph_data, training=True)

    pnm = PairwiseNodeSimilarity(54)

    n_graphs = ds.n_graphs_per_batch_train

    results = []
    for i in range(len(gcn.output_sizes)):
        sim_mat, _ = pnm(all_node_states[i], graph_data.graph_idx, n_graphs)
        results.append(sim_mat)
    sim_mat = tf.stack(results, axis=-1)

    cnn = SeparateCNN(
        len(gcn.output_sizes),
        [16, 32, 64, 128, 128],
        [6, 6, 5, 5, 5],
        [2, 2, 2, 3, 3])
    o = cnn(sim_mat)
