import tensorflow as tf

from gmn.graph_prop_layer import GraphPropLayer


class GraphPropMatchingLayer(GraphPropLayer):
    def __init__(self, *args, **kwargs):
        assert "matching_similarity_type" in kwargs
        self.similarity_type = kwargs["matching_similarity_type"]
        del kwargs["matching_similarity_type"]

        GraphPropLayer.__init__(self, *args, **kwargs)

    def call(self, node_states, from_idx, to_idx,
             graph_idx, n_graphs, training,
             node_features=None, edge_features=None):
        aggregated_messages = self.compute_aggregated_messages(
            node_states, from_idx, to_idx, training=training,
            edge_features=edge_features)

        cross_graph_attention = batch_block_pair_attention(
            node_states, graph_idx, n_graphs, self.similarity_type)
        attention_messages = node_states - cross_graph_attention

        updated_node_states = self.compute_node_update(
            node_states, [aggregated_messages, attention_messages],
            training=training, node_features=node_features)

        return updated_node_states


def batch_block_pair_attention(data, block_idx, n_blocks, similarity):
    similarity_fn = PAIRWISE_SIMILARITY_FUNCTION[similarity]

    partitions = tf.dynamic_partition(data, block_idx, n_blocks)

    results = []
    for i in range(0, n_blocks, 2):
        x = partitions[i]
        y = partitions[i + 1]
        attention_x, attention_y = compute_cross_attention(x, y, similarity_fn)
        results.append(attention_x)
        results.append(attention_y)
    results = tf.concat(results, axis=0)

    return results


# it is too slow to use TensorArray for dynamic unrolling
def _batch_block_pair_attention(data, block_idx, n_blocks, similarity):
    similarity_fn = PAIRWISE_SIMILARITY_FUNCTION[similarity]

    results = tf.TensorArray(tf.float32, size=n_blocks)
    for i in range(0, n_blocks, 2):
        mask_x = tf.equal(block_idx, i)
        mask_y = tf.equal(block_idx, i + 1)

        x = tf.boolean_mask(data, mask_x)
        y = tf.boolean_mask(data, mask_y)

        attention_x, attention_y = compute_cross_attention(x, y, similarity_fn)

        results = results.write(i, attention_x)
        results = results.write(i + 1, attention_y)

    results = results.concat()
    return results


def compute_cross_attention(x, y, similarity_fn):
    simi = similarity_fn(x, y)

    simi_x = tf.nn.softmax(simi, axis=1)
    simi_y = tf.nn.softmax(simi, axis=0)

    attention_x = tf.matmul(simi_x, y)
    attention_y = tf.matmul(simi_y, x, transpose_a=True)

    return attention_x, attention_y


def pairwise_dot_product_similarity(x, y):
    return tf.matmul(x, y, transpose_b=True)


PAIRWISE_SIMILARITY_FUNCTION = {
    "dotproduct": pairwise_dot_product_similarity
}
