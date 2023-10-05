import json
import pickle
import itertools
import os.path as osp
import networkx as nx
import numpy as np
import scipy as sp
import tensorflow as tf
from collections import defaultdict

from data_loader.dataset import Dataset
from utils.common_utils import string_to_numpy, chunked


class AlchemyDataset(Dataset):
    def init(self):
        self.data_name = "alchemy"
        self.data_dir = osp.join(
            osp.dirname(osp.abspath(__file__)), "../data", self.data_name)
        self.node_feat_type = "ready"
        self.node_feat_name = None
        self.node_ordering = None
        self.eval_size = 2000
        self.eval_n_neg = 99
        self.batched_data_file_suffix = (
            f"{self.node_feat_type}_{self.node_feat_name}_"
            f"{self.eval_size}_{self.eval_n_neg}")
        assert not hasattr(self, "rng_preprocess")
        self.rng_preprocess = np.random.RandomState(66666)

    def load_data(self):
        offset = 0
        for mode in ["train", "val", "test"]:
            graph_list = []
            with open(osp.join(
                    self.data_dir, f"{mode}_graphs.json"), "r") as f:
                for i, line in enumerate(f):
                    g_data = json.loads(line)
                    g = nx.node_link_graph(g_data)
                    g.graph["gid"] = offset + i
                    g.graph["label"] = np.array(g.graph["label"])
                    graph_list.append(g)
            setattr(self, f"{mode}_graphs", graph_list)
            offset += len(graph_list)

        filepath = osp.join(self.data_dir, "train_pos_pairs.pkl")
        if osp.isfile(filepath):
            with open(filepath, "rb") as f:
                train_pos_pairs = pickle.load(f)
        else:
            train_pos_pairs = set()
            train_graph_labels = np.stack(
                [g.graph["label"] for g in self.train_graphs],
                axis=0).astype(np.float32)
            offset = 0
            for chunk in chunked(train_graph_labels, 10000):
                this_labels = np.stack(chunk, axis=0)
                sim = -1.0 * sp.spatial.distance.cdist(
                    this_labels, train_graph_labels).astype(np.float32)
                for i in range(sim.shape[0]):
                    sim[i, i + offset] = -9999.0
                mask_1 = sim >= -0.5
                mask_2 = sim >= np.max(sim, axis=1, keepdims=True)
                for mask in [mask_1, mask_2]:
                    indices_1, indices_2 = np.nonzero(mask)
                    indices_1 += offset
                    train_pos_pairs.update(zip(indices_1, indices_2))
                offset += sim.shape[0]
            train_pos_pairs = list(train_pos_pairs)
            with open(filepath, "wb") as f:
                pickle.dump(train_pos_pairs, f)
        self.rng_preprocess.shuffle(train_pos_pairs)
        self.train_pos_pairs = train_pos_pairs
        self.rel_graphs = defaultdict(list)
        for i, j in self.train_pos_pairs:
            self.rel_graphs[i].append(j)
        self.rel_graphs = dict(self.rel_graphs)

    def sample_once(self, rng):
        index = rng.choice(len(self.train_pos_pairs))
        query_idx, pos_idx = self.train_pos_pairs[index]
        query = self.train_graphs[query_idx]
        pos = self.train_graphs[pos_idx]
        k_neg = []
        while True:
            neg_idx = rng.choice(len(self.train_graphs))
            if (neg_idx != query_idx) \
                    and (neg_idx not in self.rel_graphs[query_idx]):
                neg = self.train_graphs[neg_idx]
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
            label_anchor = g_anchor.graph["label"].reshape((1, -1))
            labels_others = np.stack(
                [g.graph["label"] for g in g_others], axis=0)
            sim = -1.0 * sp.spatial.distance.cdist(
                label_anchor, labels_others).flatten()
            sorted_sim, sorted_g_others = list(
                zip(*sorted(zip(sim, g_others), key=lambda x: x[0])))
            g_pos, gs_neg = sorted_g_others[-1], sorted_g_others[:-1]
            if sorted_sim[-1] > sorted_sim[-2]:
                break
        return g_anchor, g_pos, gs_neg

    def prepare_val(self):
        rnd_perm = self.rng_preprocess.permutation(len(self.val_graphs))
        query_graphs = [self.val_graphs[i] for i in rnd_perm[:self.eval_size]]
        graph_pairs = []
        labels = []
        for qg in query_graphs:
            candidate_graph_indices = self.rng_preprocess.choice(
                len(self.train_graphs),
                size=1 + self.eval_n_neg, replace=False)
            candidate_graphs = [
                self.train_graphs[i] for i in candidate_graph_indices]
            qg_label = qg.graph["label"].reshape(1, -1)
            cg_labels = np.stack(
                [cg.graph["label"] for cg in candidate_graphs], axis=0)
            sim = -1.0 * sp.spatial.distance.cdist(
                qg_label, cg_labels).flatten()
            labels.append(sim)
            graph_pairs.extend([(qg, cg) for cg in candidate_graphs])
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        labels = tf.cast(
            tf.greater_equal(
                labels, tf.reduce_max(labels, axis=1, keepdims=True)),
            tf.float32)
        return graph_pairs, labels

    def prepare_test(self):
        query_graphs = self.test_graphs
        candidate_graphs = (
            self.train_graphs + self.val_graphs + self.test_graphs)
        qg_labels = np.stack(
            [qg.graph["label"] for qg in query_graphs],
            axis=0).astype(np.float32)
        cg_labels = np.stack(
            [cg.graph["label"] for cg in candidate_graphs],
            axis=0).astype(np.float32)
        sim = -1.0 * sp.spatial.distance.cdist(
            qg_labels, cg_labels).astype(np.float32)
        offset = len(self.train_graphs) + len(self.val_graphs)
        for i in range(len(query_graphs)):
            sim[i, i + offset] = - 9999.0
        mask_1 = sim >= -0.5
        mask_2 = sim >= np.max(sim, axis=1, keepdims=True)
        indices = set()
        for mask in [mask_1, mask_2]:
            indices.update(zip(*np.nonzero(mask)))
        indices = list(indices)
        values = np.ones((len(indices),), dtype=np.float32)
        shape = [len(query_graphs), len(candidate_graphs)]
        with tf.device("/device:cpu:0"):
            labels = tf.sparse.reorder(
                tf.sparse.SparseTensor(indices, values, shape))
            labels = tf.sparse.to_dense(labels)
        return query_graphs, candidate_graphs, labels

    def handle_test_predictions(self, preds, i):
        predictions = preds.numpy()
        offset = len(self.train_graphs) + len(self.val_graphs)
        sizes = [x.shape[0] for x in self.test_labels]
        offset_sizes = np.cumsum(sizes) + offset
        for idx in range(predictions.shape[0]):
            predictions[idx, offset_sizes[i] - sizes[i] + idx] = - 9999.0
        predictions = tf.convert_to_tensor(predictions)
        return predictions

    def construct_hatA_graphs(self):
        for g in itertools.chain(
                self.train_graphs, self.val_graphs, self.test_graphs):
            g.graph["hatA_edges"] = string_to_numpy(g.graph["hatA_edges"])

    def partition_graphs(self):
        assert self.n_subgraphs in [2, 3]
        for g in itertools.chain(
                self.train_graphs, self.val_graphs, self.test_graphs):
            g.graph["partition"] = g.graph["partition"][str(self.n_subgraphs)]
