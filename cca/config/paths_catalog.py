"""Centralized catalog of paths."""
import os

class DatasetCatalog(object):
    DATA_DIR = "./data"
    tacos_file = {"VGG": "vgg_fc7.hdf5", "C3D": "tall_c3d_features.hdf5", "I3D": "i3d_imagenet.hdf5"}
    acnet_file = {"C3D": "sub_activitynet_v1-3.c3d.hdf5", "I3D": "cmcs_features.hdf5"}
    DATASETS = {
        "tacos_train":{
            "video_dir": "tacos/videos",
            "ann_file": "TACoS/train.json",
            "feat_file": "TACoS/{}".format(tacos_file['C3D']),
        },
        "tacos_val":{
            "video_dir": "tacos/videos",
            "ann_file": "TACoS/val.json",
            "feat_file": "TACoS/{}".format(tacos_file['C3D']),
        },
        "tacos_test":{
            "video_dir": "tacos/videos",
            "ann_file": "TACoS/test.json",
            "feat_file": "TACoS/{}".format(tacos_file['C3D']),
        },
        "activitynet_train":{
            "video_dir": "ActivityNet/videos",
            "ann_file": "ActivityNet/annotations/train.json",
            "feat_file": "ActivityNet/{}".format(acnet_file["C3D"]),
        },
        "activitynet_val":{
            "video_dir": "ActivityNet/videos",
            "ann_file": "ActivityNet/annotations/val.json",
            "feat_file": "ActivityNet/{}".format(acnet_file["C3D"]),
        },
        "activitynet_test":{
            "video_dir": "ActivityNet/videos",
            "ann_file": "ActivityNet/annotations/test.json",
            "feat_file": "ActivityNet/{}".format(acnet_file["C3D"]),
        }
    }

    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]
        args = dict(
            root=os.path.join(data_dir, attrs["video_dir"]),
            ann_file=os.path.join(data_dir, attrs["ann_file"]),
            feat_file=os.path.join(data_dir, attrs["feat_file"]),
        )
        if "tacos" in name:
            return dict(
                factory="TACoSDataset",
                args=args,
            )
        elif "activitynet" in name:
            return dict(
                factory = "ActivityNetDataset",
                args = args
            )
        raise RuntimeError("Dataset not available: {}".format(name))
