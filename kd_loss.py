import tensorflow as tf

from graph_sim.pairwise_node_similarity import PairwiseNodeSimilarity


class KDLoss:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.pns = PairwiseNodeSimilarity(None, padding=False, norm=False)

    def __call__(self, outputs, graph_data, n_graphs, kd_type, mode):
        assert kd_type in ["score", "node", "subgraph"]
        assert mode in ["t->s", "s->t"]

        if kd_type == "score":
            if mode == "t->s":
                sim_true, sim_pred = outputs["score_t"], outputs["score_s"]
            elif mode == "s->t":
                sim_true, sim_pred = outputs["score_s"], outputs["score_t"]
            sim_true = tf.reshape(sim_true, [-1, 1 + self.FLAGS.n_neg_graphs])
            sim_pred = tf.reshape(sim_pred, [-1, 1 + self.FLAGS.n_neg_graphs])
            kd_loss = self.compute_kld(sim_true, sim_pred, "score")

        elif kd_type == "node":
            if mode == "t->s":
                node_states_true = outputs["node_states_t"]
                node_states_pred = outputs["node_states_s"]
            elif mode == "s->t":
                node_states_true = outputs["node_states_s"]
                node_states_pred = outputs["node_states_t"]
            if self.FLAGS.neighbor_enhanced:
                def _func(node_states):
                    to_states = tf.gather(node_states, graph_data.hatA_to_idx)
                    weighted_to_states = tf.reshape(
                        graph_data.hatA_weights, [-1, 1]) * to_states
                    new_states = tf.math.unsorted_segment_sum(
                        weighted_to_states, graph_data.hatA_from_idx,
                        tf.shape(node_states)[0])
                    return new_states
                enhanced_node_states_true = _func(node_states_true)
                enhanced_node_states_pred = _func(node_states_pred)
                node_sim_true = self.pns(
                    enhanced_node_states_true, graph_data.graph_idx,
                    n_graphs, resize=False)
                node_sim_pred = self.pns(
                    enhanced_node_states_pred, graph_data.graph_idx,
                    n_graphs, resize=False)
            else:
                node_sim_true = self.pns(
                    node_states_true, graph_data.graph_idx,
                    n_graphs, resize=False)
                node_sim_pred = self.pns(
                    node_states_pred, graph_data.graph_idx,
                    n_graphs, resize=False)

            # if self.FLAGS.topk_node:
            #     top_node_sim_true, top_node_sim_pred = [], []
            #     for sim_true, sim_pred in zip(node_sim_true, node_sim_pred):
            #         k = tf.minimum(tf.shape(sim_true)[1], 10)
            #         top_sim_true, top_indices = tf.math.top_k(sim_true, k=k)
            #         top_sim_pred = tf.gather(
            #             sim_pred, top_indices, batch_dims=1)
            #         top_node_sim_true.append(top_sim_true)
            #         top_node_sim_pred.append(top_sim_pred)
            #     kld_list = [
            #         self.compute_kld(y_true, y_pred, "node")
            #         for y_true, y_pred in zip(
            #             top_node_sim_true, top_node_sim_pred)]
            # else:
            #     kld_list = [
            #         self.compute_kld(y_true, y_pred, "node")
            #         for y_true, y_pred in zip(node_sim_true, node_sim_pred)]
            # kd_loss = tf.reduce_sum(kld_list)

            padded_node_sim_true, padded_node_sim_pred = self.pad_sim_list(
                node_sim_true, node_sim_pred)
            kd_loss = self.compute_kld(
                padded_node_sim_true, padded_node_sim_pred, "node")

        elif kd_type == "subgraph":
            if mode == "t->s":
                node_states_true = outputs["node_states_t"]
                node_states_pred = outputs["node_states_s"]
            elif mode == "s->t":
                node_states_true = outputs["node_states_s"]
                node_states_pred = outputs["node_states_t"]
            if self.FLAGS.segment == "mean":
                segment_op = tf.math.unsorted_segment_mean
            elif self.FLAGS.segment == "sum":
                segment_op = tf.math.unsorted_segment_sum
            if self.FLAGS.n_subgraphs == -1:
                subgraph_sim_true = embedding_to_subgraph_similarity_new(
                    node_states_true,
                    graph_data.subgraph_idx, graph_data.n_subgraphs,
                    graph_data.subgraph_to_graph_idx, n_graphs, segment_op)
                subgraph_sim_pred = embedding_to_subgraph_similarity_new(
                    node_states_pred,
                    graph_data.subgraph_idx, graph_data.n_subgraphs,
                    graph_data.subgraph_to_graph_idx, n_graphs, segment_op)
                # kld_list = [
                #     self.compute_kld(y_true, y_pred, "subgraph")
                #     for y_true, y_pred in zip(
                #         subgraph_sim_true, subgraph_sim_pred)]
                # kd_loss = tf.reduce_sum(kld_list)
                padded_subgraph_sim_true, padded_subgraph_sim_pred \
                    = self.pad_sim_list(subgraph_sim_true, subgraph_sim_pred)
                kd_loss = self.compute_kld(
                    padded_subgraph_sim_true,
                    padded_subgraph_sim_pred, "subgraph")
            else:
                subgraph_sim_true = embedding_to_subgraph_similarity(
                    node_states_true,
                    graph_data.subgraph_idx, graph_data.n_subgraphs,
                    self.FLAGS.n_subgraphs, segment_op)
                subgraph_sim_pred = embedding_to_subgraph_similarity(
                    node_states_pred,
                    graph_data.subgraph_idx, graph_data.n_subgraphs,
                    self.FLAGS.n_subgraphs, segment_op)
                kd_loss = self.compute_kld(
                    subgraph_sim_true, subgraph_sim_pred, "subgraph")

        return kd_loss

    def pad_sim_list(self, sim_list_true, sim_list_pred):
        lengths = tf.convert_to_tensor(
            [tf.shape(x)[1] for x in sim_list_true])
        max_len = tf.reduce_max(lengths)
        padded_true_list = [
            tf.pad(x,
                   paddings=[[0, 0], [0, max_len - lengths[i]]],
                   constant_values=-1e9)
            for i, x in enumerate(sim_list_true)]
        padded_pred_list = [
            tf.pad(x,
                   paddings=[[0, 0], [0, max_len - lengths[i]]],
                   constant_values=-1e9)
            for i, x in enumerate(sim_list_pred)]
        padded_true = tf.concat(padded_true_list, axis=0)
        padded_pred = tf.concat(padded_pred_list, axis=0)
        return padded_true, padded_pred

    def compute_kld(self, y_true, y_pred, kd_type):
        if kd_type == "score":
            T_true = self.FLAGS.kd_score_T
            T_pred = self.FLAGS.kd_score_T
            reduce_op = tf.reduce_sum
        elif kd_type == "node":
            if self.FLAGS.kd_node_mode == 1:
                T_true = self.FLAGS.kd_node_T
                T_pred = self.FLAGS.kd_node_T
            elif self.FLAGS.kd_node_mode == 2:
                T_true = self.FLAGS.kd_node_T
                T_pred = 1.0
            reduce_op = tf.reduce_sum
        elif kd_type == "subgraph":
            if self.FLAGS.kd_subgraph_mode == 1:
                T_true = self.FLAGS.kd_subgraph_T
                T_pred = self.FLAGS.kd_subgraph_T
            elif self.FLAGS.kd_subgraph_mode == 2:
                T_true = self.FLAGS.kd_subgraph_T
                T_pred = 1.0
            reduce_op = tf.reduce_sum
        prob_true = tf.nn.softmax(y_true / T_true, axis=-1)
        prob_pred = tf.nn.softmax(y_pred / T_pred, axis=-1)
        kld = reduce_op(tf.keras.losses.KLD(prob_true, prob_pred))
        return kld


def embedding_to_subgraph_similarity(embeddings, subgraph_idx, n_subgraphs,
                                     k, segment_op):
    # subgraph_embeddings = tf.math.unsorted_segment_sum(
    subgraph_embeddings = segment_op(
        embeddings, subgraph_idx, n_subgraphs)
    embed_dim = tf.shape(subgraph_embeddings)[1]
    subembed_x, subembed_y = tf.split(
        tf.reshape(subgraph_embeddings, [-1, 2 * k * embed_dim]),
        num_or_size_splits=2, axis=1)
    subembed_x = tf.reshape(subembed_x, [-1, k, embed_dim])
    subembed_y = tf.reshape(subembed_y, [-1, k, embed_dim])
    subgraph_sim = tf.matmul(subembed_x, subembed_y, transpose_b=True)
    return subgraph_sim


def embedding_to_subgraph_similarity_new(embeddings,
                                         subgraph_idx, n_subgraphs,
                                         subgraph_to_graph_idx, n_graphs,
                                         segment_op):
    subgraph_embeddings = segment_op(
        embeddings, subgraph_idx, n_subgraphs)
    partitions = tf.dynamic_partition(
        subgraph_embeddings, subgraph_to_graph_idx, n_graphs)
    subgraph_sim = []
    for i in range(0, n_graphs, 2):
        x = partitions[i]
        y = partitions[i + 1]
        sim = tf.matmul(x, y, transpose_b=True)
        subgraph_sim.append(sim)
    return subgraph_sim
