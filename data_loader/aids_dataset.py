import os.path as osp
import glob
import pickle
import numpy as np
import networkx as nx
import tensorflow as tf

from data_loader.dataset import Dataset
from utils.common_utils import sorted_nicely
from utils.node_ordering import node_ordering as node_ordering_func


class AIDSDataset(Dataset):
    def init(self):
        self.data_name = "aids700"
        self.data_dir = osp.join(
            osp.dirname(osp.abspath(__file__)), "../data", self.data_name)
        self.val_ratio = 0.25
        self.node_feat_type = "onehot"
        self.node_feat_name = "type"
        self.batched_data_file_suffix = (
            f"{self.node_feat_type}_{self.node_feat_name}_{self.n_subgraphs}")
        self.rng_preprocess = np.random.RandomState(66666)

    def load_data(self):
        orig_train_files = sorted_nicely(
            glob.glob(f"{self.data_dir}/train/*.gexf"))
        orig_test_files = sorted_nicely(
            glob.glob(f"{self.data_dir}/test/*.gexf"))

        def _relabeling_func(g):
            if self.node_ordering is None:
                assert False
                new_g = nx.convert_node_labels_to_integers(g)
            elif self.node_ordering == "bfs":
                order, mapping = node_ordering_func(
                    g, "bfs", self.node_feat_name, [])
                node_list = list(g.nodes)
                mapping = {node_list[j]: i for i, j in enumerate(order)}
                new_g = nx.relabel_nodes(g, mapping)
            else:
                raise ValueError
            return new_g

        def _read_graph(file):
            gid = int(osp.basename(file).split(".")[0])
            g = nx.read_gexf(file)
            g.graph["gid"] = gid
            g = _relabeling_func(g)
            assert nx.is_connected(g)
            return g
        orig_train_graphs = [_read_graph(f) for f in orig_train_files]
        orig_test_graphs = [_read_graph(f) for f in orig_test_files]

        n_train_graphs = int(len(orig_train_graphs)
                             * (1 - self.val_ratio))
        self.train_graphs = orig_train_graphs[:n_train_graphs]
        self.val_graphs = orig_train_graphs[n_train_graphs:]
        self.test_graphs = orig_test_graphs

        self.n_train_graphs = len(self.train_graphs)
        self.n_val_graphs = len(self.val_graphs)
        self.n_test_graphs = len(self.test_graphs)

        self.train_gid_to_index = {
            g.graph["gid"]: index for index, g in enumerate(self.train_graphs)}
        self.val_gid_to_index = {
            g.graph["gid"]: index for index, g in enumerate(self.val_graphs)}
        self.test_gid_to_index = {
            g.graph["gid"]: index for index, g in enumerate(self.test_graphs)}

        ged_file = glob.glob(f"{self.data_dir}/{self.data_name}"
                             f"*ged_astar_gidpair_dist_map.pickle")[0]
        with open(ged_file, "rb") as f:
            self.gidpair_to_ged = pickle.load(f)

        self.train_pos_pairs = []
        labels = []
        for g1 in self.train_graphs:
            this_label = []
            for g2 in self.train_graphs:
                this_label.append(self.gidpair_to_ged[(g1.graph["gid"], g2.graph["gid"])])
            labels.append(this_label)
        labels = np.array(labels)
        for i in range(labels.shape[0]):
            labels[i, i] = 100
        labels = np.less_equal(labels, labels.min(axis=1, keepdims=True))
        for i, j in zip(*labels.nonzero()):
            self.train_pos_pairs.append((self.train_graphs[i].graph["gid"], self.train_graphs[j].graph["gid"]))
        self.rng_preprocess.shuffle(self.train_pos_pairs)
        # self.val_pos_pairs = []
        # labels = []
        # for g1 in self.val_graphs:
        #     this_label = []
        #     for g2 in self.train_graphs + self.val_graphs:
        #         this_label.append(self.gidpair_to_ged[(g1.graph["gid"], g2.graph["gid"])])
        #     labels.append(this_label)
        # labels = np.array(labels)
        # labels[labels == 0] = 100
        # labels = np.less_equal(labels, labels.min(axis=1, keepdims=True))
        # for i, j in zip(*labels.nonzero()):
        #     tmp_j = j % self.n_train_graphs
        #     self.val_pos_pairs.append((self.val_graphs[i].graph["gid"], self.))
        # self.test_pos_pairs = []

    def sample_once(self, rng):
        index = rng.choice(len(self.train_pos_pairs))
        query_gid, pos_gid = self.train_pos_pairs[index]
        query = self.gid_to_g(query_gid)
        pos = self.gid_to_g(pos_gid)
        k_neg = []
        while True:
            neg = self.train_graphs[rng.choice(len(self.train_graphs))]
            neg_gid = neg.graph["gid"]
            if self.gidpair_to_ged[(query_gid, neg_gid)] > self.gidpair_to_ged[(query_gid, pos_gid)]:
                k_neg.append(neg)
            if len(k_neg) == self.n_neg_graphs:
                break
        return query, pos, k_neg

    def _sample_once(self, rng):
        from_graphs, to_graphs = self.train_graphs, self.train_graphs
        while True:
            n_from, n_to = len(from_graphs), len(to_graphs)
            gidx_anchor = rng.choice(n_from)
            gidx_others = rng.choice(
                n_to, size=1 + self.n_neg_graphs, replace=False)
            g_anchor = from_graphs[gidx_anchor]
            g_others = [to_graphs[gidx] for gidx in gidx_others]
            sim = [
                -1.0 * self.gidpair_to_ged[
                    (g_anchor.graph["gid"], g.graph["gid"])]
                for g in g_others]
            sorted_sim, sorted_g_others = list(
                zip(*sorted(zip(sim, g_others), key=lambda x: x[0])))
            g_pos, gs_neg = sorted_g_others[-1], sorted_g_others[:-1]
            if sorted_sim[-1] > sorted_sim[-2]:
                # print(sorted_sim[-1], sorted_sim[-2])
                break
        return g_anchor, g_pos, gs_neg

    def prepare_mbp(self, mode):
        if mode == "val":
            return self.prepare_val_mbp()
        elif mode == "test":
            return self.prepare_test_mbp()

    def prepare_ebp(self, mode):
        if mode == "val":
            return self.prepare_val_ebp()
        elif mode == "test":
            return self.prepare_test_ebp()

    def handle_ebp_predictions(self, preds, i, mode):
        return preds
        if mode == "val":
            return self.handle_val_predictions(preds, i)
        elif mode == "test":
            return self.handle_test_predictions(preds, i)

    def prepare_val_mbp(self):
        query_graphs = self.val_graphs
        candidate_graphs = self.train_graphs
        graph_pairs = []
        labels = []
        for qg in query_graphs:
            this_label = []
            for cg in candidate_graphs:
                graph_pairs.append((qg, cg))
                this_label.append(
                    self.gidpair_to_ged[(qg.graph["gid"], cg.graph["gid"])])
            labels.append(this_label)
        labels = np.array(labels)
        # labels[labels == 0] = 100
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        labels = tf.cast(
            tf.less_equal(
                labels, tf.reduce_min(labels, axis=1, keepdims=True)),
            dtype=tf.float32)
        return graph_pairs, labels

    def prepare_val_ebp(self):
        query_graphs = self.val_graphs
        candidate_graphs = self.train_graphs
        _, labels = self.prepare_val_mbp()
        assert labels.shape[0] == len(query_graphs) and labels.shape[1] == len(candidate_graphs)
        return query_graphs, candidate_graphs, labels

    def handle_val_predictions(self, preds, i):
        predictions = preds.numpy()
        offset = len(self.train_graphs)
        sizes = [x.shape[0] for x in self.val_ebp_labels]
        offset_sizes = np.cumsum(sizes) + offset
        for idx in range(predictions.shape[0]):
            predictions[idx, offset_sizes[i] - sizes[i] + idx] = -9999.0
        predictions = tf.convert_to_tensor(predictions)
        return predictions

    def prepare_test_ebp(self):
        query_graphs = self.test_graphs
        candidate_graphs = (
            self.train_graphs + self.val_graphs)
        _, labels = self.prepare_test_mbp()
        assert len(query_graphs) == labels.shape[0] and len(candidate_graphs) == labels.shape[1]
        return query_graphs, candidate_graphs, labels

    def prepare_test_mbp(self):
        query_graphs = self.test_graphs
        candidate_graphs = self.train_graphs + self.val_graphs
        graph_pairs = []
        labels = []
        for qg in query_graphs:
            this_label = []
            for cg in candidate_graphs:
                graph_pairs.append((qg, cg))
                this_label.append(
                    self.gidpair_to_ged[(qg.graph["gid"], cg.graph["gid"])])
            labels.append(this_label)
        labels = np.array(labels)
        # labels[labels == 0] = 100
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        labels = tf.cast(
            tf.less_equal(
                labels, tf.reduce_min(labels, axis=1, keepdims=True)),
            dtype=tf.float32)
        return graph_pairs, labels

    def handle_test_predictions(self, preds, i):
        predictions = preds.numpy()
        offset = len(self.train_graphs) + len(self.val_graphs)
        sizes = [x.shape[0] for x in self.test_ebp_labels]
        offset_sizes = np.cumsum(sizes) + offset
        for idx in range(predictions.shape[0]):
            predictions[idx, offset_sizes[i] - sizes[i] + idx] = - 9999.0
        predictions = tf.convert_to_tensor(predictions)
        return predictions

    def __prepare_val_test(self, mode):
        assert mode in ["val", "test"]
        if mode == "val":
            query_graphs = self.val_graphs
            candidate_graphs = self.train_graphs
        elif mode == "test":
            query_graphs = self.test_graphs
            candidate_graphs = self.train_graphs + self.val_graphs

        graph_pairs = []
        labels = []
        for qg in query_graphs:
            this_label = []
            for cg in candidate_graphs:
                graph_pairs.append((qg, cg))
                this_label.append(
                    self.gidpair_to_ged[(qg.graph["gid"], cg.graph["gid"])])
            labels.append(this_label)
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        labels = tf.cast(
            tf.less_equal(
                labels, tf.reduce_min(labels, axis=1, keepdims=True)),
            dtype=tf.float32)
        # setattr(self, f"{mode}_pairs", graph_pairs)
        # setattr(self, f"{mode}_labels", labels)
        return graph_pairs, labels

    def gid_to_g(self, gid):
        if gid in self.train_gid_to_index:
            index = self.train_gid_to_index[gid]
            g = self.train_graphs[index]
        elif gid in self.val_gid_to_index:
            index = self.val_gid_to_index[gid]
            g = self.val_graphs[index]
        elif gid in self.test_gid_to_index:
            index = self.test_gid_to_index[gid]
            g = self.test_graphs[index]
        else:
            raise ValueError
        assert gid == g.graph["gid"]
        return g
