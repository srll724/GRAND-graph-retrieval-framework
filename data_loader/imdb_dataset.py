import os.path as osp

from data_loader.aids_dataset import AIDSDataset


class IMDBDataset(AIDSDataset):
    def init(self):
        self.data_name = "imdb"
        self.data_dir = osp.join(
            osp.abspath(osp.dirname(__file__)), "../data", self.data_name)
        self.node_feat_name = None
        self.node_feat_type = "constant_1"
        self.val_ratio = 0.25
