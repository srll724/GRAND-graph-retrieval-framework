import tensorflow as tf

KL = tf.keras.layers


class PairwiseNodeSimilarity(KL.Layer):
    def __init__(self, fixed_size, padding=True, norm=False, name=None):
        super(PairwiseNodeSimilarity, self).__init__(name=name)

        self.fixed_size = fixed_size
        self.padding = padding
        self.norm = norm

    def call(self, features, graph_idx, n_graphs, resize=True):
        partitions = tf.dynamic_partition(features, graph_idx, n_graphs)

        resized_results = []
        unresized_results = []
        with tf.name_scope("graph_pair_similarity"):
            for i in range(0, n_graphs, 2):
                with tf.name_scope(f"graph_pair_{i // 2}"):
                    x = partitions[i]
                    y = partitions[i + 1]
                    this_sim = self.compute_pairwise_similarity(
                        x, y, resize=resize)
                    if resize:
                        resized_results.append(this_sim[0])
                        unresized_results.append(this_sim[1])
                    else:
                        unresized_results.append(this_sim)
        if resize:
            with tf.name_scope("stack_similarity"):
                outputs = tf.stack(resized_results, axis=0)
            return outputs, unresized_results
        else:
            return unresized_results

    def compute_pairwise_similarity(self, x, y, resize):
        if self.padding:
            with tf.name_scope("padding"):
                x_shape, y_shape = tf.shape(x), tf.shape(y)
                max_dim = tf.maximum(x_shape[0], y_shape[0])
                padded_x = tf.pad(x, [[0, max_dim - x_shape[0]], [0, 0]])
                padded_y = tf.pad(y, [[0, max_dim - y_shape[0]], [0, 0]])
        else:
            padded_x, padded_y = x, y
        if self.norm:
            assert len(padded_x.shape) == 2
            depth = tf.cast(tf.shape(padded_x)[1], tf.float32)
            padded_x *= depth ** -0.5
        sim_mat = tf.matmul(padded_x, padded_y, transpose_b=True)
        # sim_mat = tf.matmul(x, y, transpose_b=True)

        if resize:
            with tf.name_scope("resize"):
                # resized_sim_mat = tf.compat.v1.image.resize_images(
                #     tf.expand_dims(sim_mat, axis=2),
                #     [self.fixed_size, self.fixed_size],
                #     align_corners=True)
                resized_sim_mat = tf.image.resize(
                    tf.expand_dims(sim_mat, axis=2),
                    [self.fixed_size, self.fixed_size],
                    method="bilinear")
                resized_sim_mat = tf.squeeze(resized_sim_mat, axis=2)
            return resized_sim_mat, sim_mat
        else:
            return sim_mat


if __name__ == "__main__":
    from dataset import Dataset
    from basic_layers.gcn import GCN

    ds = Dataset("aids700", "regression", "bfs",
                 "onehot", "type", 0.25, 32, 1000, 123)
    train_data = ds.get_data("training")
    graph_data = train_data[0]

    gcn = GCN([128, 64, 32], graph_pooling=True)

    node_states, all_node_states = gcn(graph_data, training=True)

    import ipdb; ipdb.set_trace()

    pnm = PairwiseNodeSimilarity(54)

    n_graphs = ds.n_graphs_per_batch_train

    results = []
    for i in range(len(gcn.output_sizes)):
        sim_mat_list, _ = pnm(all_node_states[i], graph_data.graph_idx, n_graphs)
        results.append(sim_mat_list)

    @tf.function
    def test1():
        sim_mats = []
        for i in range(n_graphs // 2):
            temp = []
            for j in range(len(gcn.output_sizes)):
                temp.append(results[j][i])
            sim_mats.append(tf.stack(temp, axis=-1))
        sim_mat = tf.stack(sim_mats, axis=0)
        return sim_mat

    @tf.function
    def test2():
        new_results = [tf.stack(ts, axis=0) for ts in results]
        new_sim_mat = tf.stack(new_results, axis=-1)
        return new_sim_mat

    r1 = test1()
    r2 = test2()
    print(tf.reduce_all(tf.equal(r1, r2)))
    print(tf.reduce_all(tf.equal(r1, test1.python_function())))
    print(tf.reduce_all(tf.equal(r2, test2.python_function())))
