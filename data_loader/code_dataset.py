import os.path as osp
import glob
import json
import ray
import pickle
import itertools
import numpy as np
import networkx as nx
import tensorflow as tf
from collections import defaultdict

from data_loader.dataset import Dataset
from utils.common_utils import sorted_nicely
from utils.gcn_utils import preprocess_features
from utils.graph_utils import parallel_partition, QueueActor


class CodeDataset(Dataset):
    def init(self):
        self.data_name = "code"
        self.data_dir = osp.join(
            osp.abspath(osp.dirname(__file__)), "../data", self.data_name)
        self.data_files = sorted_nicely(
            glob.glob(osp.join(self.data_dir, "openssl*json")))
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
        graphs = {}
        labels = defaultdict(list)
        for file in self.data_files:
            with open(file, "r") as f:
                for line in f:
                    d = json.loads(line.strip())
                    g = self.to_nx_graph(d)
                    if g:
                        gid = len(graphs)
                        g.graph["gid"] = gid
                        graphs[gid] = g
                        labels[g.graph["fname"]].append(gid)
        labels_to_delete = [k for k, v in labels.items() if len(v) < 2]
        for la in labels_to_delete:
            for gid in labels[la]:
                graphs.pop(gid)
            labels.pop(la)

        self.graphs = list(graphs.values())
        self.labels = dict(labels)
        self.split_data()

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

        self.train_graphs = self.graphs[train_slice]
        self.val_graphs = self.graphs[val_slice]
        self.test_graphs = self.graphs[test_slice]

        self.train_gid_to_index = {
            g.graph["gid"]: index for index, g in enumerate(self.train_graphs)}
        self.val_gid_to_index = {
            g.graph["gid"]: index for index, g in enumerate(self.val_graphs)}
        self.test_gid_to_index = {
            g.graph["gid"]: index for index, g in enumerate(self.test_graphs)}

        self.train_pos_pairs = []
        for g in self.train_graphs:
            gid = g.graph["gid"]
            for ggid in self.labels[g.graph["fname"]]:
                if ggid != gid and ggid in self.train_gid_to_index:
                    self.train_pos_pairs.append((gid, ggid))
        self.rng_preprocess.shuffle(self.train_pos_pairs)
        self.val_pos_pairs = []
        for g in self.val_graphs:
            gid = g.graph["gid"]
            for ggid in self.labels[g.graph["fname"]]:
                if ggid != gid and ggid not in self.test_gid_to_index:
                    self.val_pos_pairs.append((gid, ggid))
        self.rng_preprocess.shuffle(self.val_pos_pairs)
        self.test_pos_pairs = []
        for g in self.test_graphs:
            gid = g.graph["gid"]
            for ggid in self.labels[g.graph["fname"]]:
                if ggid != gid:
                    self.test_pos_pairs.append((gid, ggid))
        self.rng_preprocess.shuffle(self.test_pos_pairs)

    def sample_once(self, rng):
        index = rng.choice(len(self.train_pos_pairs))
        query_gid, pos_gid = self.train_pos_pairs[index]
        query = self.gid_to_g(query_gid)
        pos = self.gid_to_g(pos_gid)
        k_neg = []
        while True:
            neg = self.train_graphs[rng.choice(len(self.train_graphs))]
            neg_gid = neg.graph["gid"]
            if neg_gid not in self.labels[query.graph["fname"]]:
                k_neg.append(neg)
            if len(k_neg) == self.n_neg_graphs:
                break
        return query, pos, k_neg

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
                    if neg_gid not in self.labels[query.graph["fname"]]:
                        graph_pairs.append((query, neg))
                        this_label.append(0.0)
                        break
            labels.append(this_label)
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        return graph_pairs, labels

    def prepare_val_ebp(self):
        query_graphs = self.val_graphs
        candidate_graphs = self.train_graphs + self.val_graphs
        indices, values = [], []
        for gid_i, gid_j in self.val_pos_pairs:
            index_i = self.val_gid_to_index[gid_i]
            a = self.train_gid_to_index.get(gid_j)
            b = self.val_gid_to_index.get(gid_j)
            c = self.test_gid_to_index.get(gid_j)
            assert c is None
            count = 0
            if a is not None:
                index_j = a
                count += 1
            if b is not None:
                index_j = b + len(self.train_graphs)
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
        for gid_i, gid_j in self.test_pos_pairs:
            index_i = self.test_gid_to_index[gid_i]
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
        return self.prepare_test_ebp()

    def handle_test_predictions(self, preds, i):
        predictions = preds.numpy()
        offset = len(self.train_graphs) + len(self.val_graphs)
        sizes = [x.shape[0] for x in self.test_ebp_labels]
        offset_sizes = np.cumsum(sizes) + offset
        for idx in range(predictions.shape[0]):
            predictions[idx, offset_sizes[i] - sizes[i] + idx] = - 9999.0
        predictions = tf.convert_to_tensor(predictions)
        return predictions

    def legacy_prepare_test(self):
        query_graphs = self.test_graphs
        candidate_graphs = (
            self.train_graphs + self.val_graphs + self.test_graphs)
        offset = len(self.train_graphs) + len(self.val_graphs)
        indices, values = [], []
        for gid1, gid2 in self.test_pos_pairs:
            index_1 = self.test_gid_to_index[gid1]
            index_2 = self.test_gid_to_index[gid2] + offset
            indices.append((index_1, index_2))
            values.append(1.0)
        shape = [len(query_graphs), len(candidate_graphs)]
        labels = tf.SparseTensor(indices, values, shape)
        labels = tf.sparse.to_dense(tf.sparse.reorder(labels))
        return query_graphs, candidate_graphs, labels

    def legacy_handle_test_predictions(self, preds):
        predictions = preds.numpy()
        offset = len(self.train_graphs) + len(self.val_graphs)
        for i in range(len(self.test_graphs)):
            predictions[i, i + offset] = - 9999.0
        predictions = tf.convert_to_tensor(predictions)
        return predictions

    def construct_hatA_graphs(self):
        filepath = osp.join(self.data_dir, f"hatA_edges.pkl")
        if osp.isfile(filepath):
            with open(filepath, "rb") as f:
                hatA_edges = pickle.load(f)
            for g in itertools.chain(self.train_graphs, self.val_graphs,
                                     self.test_graphs):
                g.graph["hatA_edges"] = hatA_edges[g.graph["gid"]]
        else:
            Dataset.construct_hatA_graphs(self)
            to_pickle = {}
            for g in itertools.chain(self.train_graphs, self.val_graphs,
                                     self.test_graphs):
                to_pickle[g.graph["gid"]] = g.graph["hatA_edges"]
            with open(filepath, "wb") as f:
                pickle.dump(to_pickle, f)

    def partition_graphs(self):
        filepath = osp.join(
            self.data_dir, f"partitions_{self.n_subgraphs}.json")
        if osp.isfile(filepath):
            with open(filepath, "r") as f:
                partitions = json.load(f)
                partitions = {int(k): v for k, v in partitions.items()}
                assert len(partitions) == len(self.graphs)
        else:
            use_ray = True
            if use_ray:
                if not ray.is_initialized():
                    ray.init()
                queue = QueueActor.remote(maxsize=20)
                graphs = self.train_graphs + self.val_graphs + self.test_graphs
                parallel_partition.remote(queue, graphs, self.n_subgraphs)
                partitions = []
                while True:
                    res = ray.get(queue.get.remote())
                    if res == QueueActor.DONE:
                        ray.kill(queue)
                        break
                    else:
                        partitions.append(ray.get(res[0]))
                partitions = {
                    g.graph["gid"]: p for g, p in zip(graphs, partitions)}
            with open(filepath, "w") as f:
                json.dump(partitions, f)
        for g in itertools.chain(self.train_graphs, self.val_graphs,
                                 self.test_graphs):
            g.graph["partition"] = partitions[g.graph["gid"]]

    def to_nx_graph(self, d):
        if d["n_num"] < 10 or d["n_num"] > 200:
            return None
        node_features = np.array(d["features"], dtype=np.float32)
        if not np.all(node_features.sum(1)):
            return None
        g = nx.Graph(fname=d["fname"])
        g.add_nodes_from(range(d["n_num"]))
        for i, ns in enumerate(d["succs"]):
            for n in ns:
                g.add_edge(i, n)
        features = preprocess_features(node_features)
        for n in g.nodes():
            g.nodes[n]["feat"] = features[n]
        if not nx.is_connected(g):
            return None
        return g

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

    def legacy_split_data(self):
        frac_array = np.array([0.9, 0.0, 0.1])
        n_labels = len(self.labels)
        lengths = (n_labels * frac_array).astype(np.int32)
        extra = n_labels - np.sum(lengths)
        lengths[1] += np.floor(extra / 2)
        lengths[2] += np.ceil(extra / 2)
        assert n_labels == np.sum(lengths)
        label_name = list(self.labels.keys())
        self.rng_preprocess.shuffle(label_name)
        train_slices, val_slices, test_slices = [
            slice(offset - length, offset)
            for offset, length in zip(np.cumsum(lengths), lengths)]
        train_labels = label_name[train_slices]
        val_labels = label_name[val_slices]
        test_labels = label_name[test_slices]

        self.train_graphs, self.val_graphs, self.test_graphs = [], [], []
        for g in self.graphs:
            la = g.graph["fname"]
            if la in train_labels:
                self.train_graphs.append(g)
            elif la in val_labels:
                self.val_graphs.append(g)
            elif la in test_labels:
                self.test_graphs.append(g)
            else:
                raise ValueError
        self.val_graphs = self.train_graphs[-1000:]
        self.train_graphs = self.train_graphs[:1000]

        self.train_gid_to_index = {
            g.graph["gid"]: index for index, g in enumerate(self.train_graphs)}
        self.val_gid_to_index = {
            g.graph["gid"]: index for index, g in enumerate(self.val_graphs)}
        self.test_gid_to_index = {
            g.graph["gid"]: index for index, g in enumerate(self.test_graphs)}

        self.train_pos_pairs = []
        for g in self.train_graphs:
            gid = g.graph["gid"]
            for ggid in self.labels[g.graph["fname"]]:
                if ggid != gid:
                    self.train_pos_pairs.append((gid, ggid))
        self.rng_preprocess.shuffle(self.train_pos_pairs)
        self.val_pos_pairs = []
        for g in self.val_graphs:
            gid = g.graph["gid"]
            for ggid in self.labels[g.graph["fname"]]:
                if ggid != gid:
                    self.val_pos_pairs.append((gid, ggid))
        self.rng_preprocess.shuffle(self.val_pos_pairs)
        self.test_pos_pairs = []
        for g in self.test_graphs:
            gid = g.graph["gid"]
            for ggid in self.labels[g.graph["fname"]]:
                if ggid != gid:
                    self.test_pos_pairs.append((gid, ggid))
        self.rng_preprocess.shuffle(self.test_pos_pairs)
