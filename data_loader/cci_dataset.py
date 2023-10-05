import json
import os.path as osp
import itertools
import numpy as np
import networkx as nx
import tensorflow as tf

from data_loader.dataset import Dataset
from utils.common_utils import string_to_numpy


class CCIDataset(Dataset):
    def init(self):
        self.data_name = "cci"
        self.data_dir = osp.join(
            osp.dirname(osp.abspath(__file__)), "../data", self.data_name)
        self.threshold = 800
        # self.node_feat_type = "sparse"
        # self.node_feat_name = "h"
        self.node_feat_type = "onehot"
        self.node_feat_name = "type"
        self.node_ordering = None
        self.eval_size = 2000
        self.eval_n_neg = 99
        self.batched_data_file_suffix = (
            f"{self.threshold}_{self.node_feat_type}_{self.node_feat_name}_"
            f"{self.eval_size}_{self.eval_n_neg}")
        assert not hasattr(self, "rng_preprocess")
        self.rng_preprocess = np.random.RandomState(66666)

    def load_data(self):
        graphs = {}
        with open(osp.join(
                self.data_dir, f"graphs_{self.threshold}.json"), "r") as f:
            for line in f:
                g_data = json.loads(line)
                g = nx.node_link_graph(g_data)
                # g.graph["node_feat_type"] = "sparse"
                graphs[g.graph["gid"]] = g
        with open(osp.join(
                self.data_dir, f"scores_{self.threshold}.json"), "r") as f:
            scores = json.load(f)
        for k, v in scores.items():
            scores[k] = {int(kk): vv for kk, vv in v.items()}
        scores = {int(k): v for k, v in scores.items()}
        for k, v in scores.items():
            if k in v:
                v.pop(k)
        n_nodes = np.array([g.number_of_nodes() for g in graphs.values()])
        max_n_nodes = np.mean(n_nodes) + 3 * np.std(n_nodes)
        # max_n_nodes = 100
        # max_n_nodes = n_nodes.max()
        min_n_nodes = 10

        graphs = {gid: g for gid, g in graphs.items()
                  if min_n_nodes <= g.number_of_nodes() <= max_n_nodes}
        for gid_i in list(scores.keys()):
            if gid_i not in graphs:
                # assert False
                this_scores = scores.pop(gid_i)
                for gid_j in this_scores.keys():
                    tmp = scores[gid_j].pop(gid_i)
                    assert tmp == this_scores[gid_j]
        gid_to_delete = [k for k, v in scores.items() if len(v) == 0]
        for gid in gid_to_delete:
            scores.pop(gid)
            graphs.pop(gid)
        # assert len(gid_to_delete) == 0

        self.graphs = list(graphs.values())
        self.scores = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)
                       for k, v in scores.items()}

        tmp = 0
        for _ in range(1):
            rtns = self.split_data()
            if len(rtns[-3]) > tmp:
                tmp = len(rtns[-3])
                best_rtns = rtns
        self.train_graphs, self.val_graphs, self.test_graphs, \
            self.train_gid_to_index, self.val_gid_to_index, \
            self.test_gid_to_index, self.rel_graphs, self.train_pos_pairs, \
            self.val_pos_pairs, self.test_pos_pairs = best_rtns
        self.rng_preprocess.shuffle(self.train_pos_pairs)
        self.rng_preprocess.shuffle(self.val_pos_pairs)
        self.rng_preprocess.shuffle(self.test_pos_pairs)

    def split_data(self):
        frac_array = np.array([0.9, 0.1, 0.1])
        n_graphs = len(self.graphs)
        lengths = (n_graphs * frac_array).astype(np.int32)
        # extra = n_graphs - np.sum(lengths)
        # lengths[1] += np.floor(extra / 2)
        # lengths[2] += np.ceil(extra / 2)
        lengths[0] -= 1000
        lengths[1] = 1000
        lengths[2] = n_graphs - lengths[0] - lengths[1]
        assert n_graphs == np.sum(lengths)
        self.rng_preprocess.shuffle(self.graphs)
        train_slice, val_slice, test_slice = [
            slice(offset - length, offset)
            for offset, length in zip(np.cumsum(lengths), lengths)]

        train_graphs = self.graphs[train_slice]
        val_graphs = self.graphs[val_slice]
        test_graphs = self.graphs[test_slice]

        train_gid_to_index = {
            g.graph["gid"]: index for index, g in enumerate(train_graphs)}
        val_gid_to_index = {
            g.graph["gid"]: index for index, g in enumerate(val_graphs)}
        test_gid_to_index = {
            g.graph["gid"]: index for index, g in enumerate(test_graphs)}

        rel_graphs = {
            k: [x[0] for x in v] for k, v in self.scores.items()}

        train_pos_pairs = []
        for gid_i, value in rel_graphs.items():
            if gid_i in train_gid_to_index:
                for gid_j in value:
                    if gid_j in train_gid_to_index:
                        train_pos_pairs.append((gid_i, gid_j))
        val_pos_pairs = []
        for gid_i, value in rel_graphs.items():
            if gid_i in val_gid_to_index:
                for gid_j in value:
                    if gid_j in train_gid_to_index:
                        val_pos_pairs.append((gid_i, gid_j))
        test_pos_pairs = []
        for gid_i, value in rel_graphs.items():
            if gid_i in test_gid_to_index:
                for gid_j in value:
                    if gid_j not in test_gid_to_index:
                        test_pos_pairs.append((gid_i, gid_j))
        return (train_graphs, val_graphs, test_graphs,
                train_gid_to_index, val_gid_to_index, test_gid_to_index,
                rel_graphs,
                train_pos_pairs, val_pos_pairs, test_pos_pairs)

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
        if mode == "val":
            return self.handle_val_predictions(preds, i)
        elif mode == "test":
            return self.handle_test_predictions(preds, i)

    def prepare_val_mbp(self):
        mode = "val"
        if mode == "val":
            pos_pairs = getattr(self, f"{mode}_pos_pairs")
            neg_graphs = self.train_graphs
        elif mode == "test":
            pos_pairs = getattr(self, f"{mode}_pos_pairs")
            neg_graphs = self.train_graphs + self.val_graphs
        graph_pairs = []
        labels = []
        pos_pairs = pos_pairs[:self.eval_size]
        for query_gid, pos_gid in pos_pairs:
            this_label = []
            query, pos = self.gid_to_g(query_gid), self.gid_to_g(pos_gid)
            graph_pairs.append((query, pos))
            this_label.append(1.0)
            for _ in range(self.eval_n_neg):
                while True:
                    neg_index = self.rng_preprocess.choice(len(neg_graphs))
                    neg = neg_graphs[neg_index]
                    neg_gid = neg.graph["gid"]
                    if (neg_gid != query_gid) \
                            and (neg_gid not in self.rel_graphs[query_gid]):
                        graph_pairs.append((query, neg))
                        this_label.append(0.0)
                        break
            labels.append(this_label)
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        return graph_pairs, labels

    def prepare_val_ebp(self):
        # query_graphs = self.val_graphs[:self.eval_size]
        query_graphs = self.val_graphs
        candidate_graphs = self.train_graphs + self.val_graphs
        indices, values = [], []
        for gid_i, gids in self.rel_graphs.items():
            if gid_i not in self.val_gid_to_index:
                continue
            index_i = self.val_gid_to_index[gid_i]
            # if index_i >= self.eval_size:
            #     continue
            for gid_j in gids:
                a = self.train_gid_to_index.get(gid_j)
                b = self.val_gid_to_index.get(gid_j)
                c = self.test_gid_to_index.get(gid_j)
                count = 0
                if a is not None:
                    index_j = a
                    count += 1
                if b is not None:
                    index_j = b + len(self.train_graphs)
                    count += 1
                if c is not None:
                    continue
                    index_j = c + len(self.train_graphs) + len(self.val_graphs)
                    count += 1
                assert count == 1
                indices.append((index_i, index_j))
                values.append(1.0)
        shape = [len(query_graphs), len(candidate_graphs)]
        labels = tf.sparse.reorder(
            tf.sparse.SparseTensor(indices, values, shape))
        labels = tf.sparse.to_dense(labels)
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
            self.train_graphs + self.val_graphs + self.test_graphs)
        indices, values = [], []
        for gid_i, gids in self.rel_graphs.items():
            if gid_i not in self.test_gid_to_index:
                continue
            index_i = self.test_gid_to_index[gid_i]
            for gid_j in gids:
                a = self.train_gid_to_index.get(gid_j)
                b = self.val_gid_to_index.get(gid_j)
                c = self.test_gid_to_index.get(gid_j)
                count = 0
                if a is not None:
                    index_j = a
                    count += 1
                if b is not None:
                    index_j = b + len(self.train_graphs)
                    count += 1
                if c is not None:
                    index_j = c + len(self.train_graphs) + len(self.val_graphs)
                    count += 1
                assert count == 1
                indices.append((index_i, index_j))
                values.append(1.0)
        shape = [len(query_graphs), len(candidate_graphs)]
        labels = tf.sparse.reorder(
            tf.sparse.SparseTensor(indices, values, shape))
        labels = tf.sparse.to_dense(labels)
        return query_graphs, candidate_graphs, labels

    def prepare_test_mbp(self):
        query_graphs = self.test_graphs  # [:1000]
        candidate_graphs = (
            self.train_graphs + self.val_graphs + self.test_graphs)
        # graph_pairs = []
        # for gq in query_graphs:
        #     for gc in candidate_graphs:
        #         graph_pairs.append((gq, gc))
        indices, values = [], []
        for gid_i, gids in self.rel_graphs.items():
            if gid_i not in self.test_gid_to_index:
                continue
            index_i = self.test_gid_to_index[gid_i]
            # if index_i >= 1000:
            #     continue
            for gid_j in gids:
                a = self.train_gid_to_index.get(gid_j)
                b = self.val_gid_to_index.get(gid_j)
                c = self.test_gid_to_index.get(gid_j)
                count = 0
                if a is not None:
                    index_j = a
                    count += 1
                if b is not None:
                    index_j = b + len(self.train_graphs)
                    count += 1
                if c is not None:
                    index_j = c + len(self.train_graphs) + len(self.val_graphs)
                    count += 1
                assert count == 1
                indices.append((index_i, index_j))
                values.append(1.0)
        shape = [len(query_graphs), len(candidate_graphs)]
        labels = tf.sparse.reorder(
            tf.sparse.SparseTensor(indices, values, shape))
        labels = tf.sparse.to_dense(labels)
        # return graph_pairs, labels
        return query_graphs, candidate_graphs, labels

    def handle_test_predictions(self, preds, i):
        predictions = preds.numpy()
        offset = len(self.train_graphs) + len(self.val_graphs)
        sizes = [x.shape[0] for x in self.test_ebp_labels]
        offset_sizes = np.cumsum(sizes) + offset
        for idx in range(predictions.shape[0]):
            predictions[idx, offset_sizes[i] - sizes[i] + idx] = - 9999.0
        predictions = tf.convert_to_tensor(predictions)
        return predictions

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

    def sample_once(self, rng):
        index = rng.choice(len(self.train_pos_pairs))
        query_gid, pos_gid = self.train_pos_pairs[index]
        query = self.gid_to_g(query_gid)
        pos = self.gid_to_g(pos_gid)
        k_neg = []
        while True:
            neg = self.train_graphs[rng.choice(len(self.train_graphs))]
            neg_gid = neg.graph["gid"]
            if (neg_gid != query_gid) \
                    and (neg_gid not in self.rel_graphs[query_gid]):
                k_neg.append(neg)
            if len(k_neg) == self.n_neg_graphs:
                break
        return query, pos, k_neg

    def construct_hatA_graphs(self):
        for g in itertools.chain(
                self.train_graphs, self.val_graphs, self.test_graphs):
            g.graph["hatA_edges"] = string_to_numpy(g.graph["hatA_edges"])

    def partition_graphs(self):
        assert self.n_subgraphs in [2, 3, -1]
        for g in itertools.chain(
                self.train_graphs, self.val_graphs, self.test_graphs):
            g.graph["partition"] = g.graph["partition"][str(self.n_subgraphs)]
