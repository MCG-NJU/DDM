import os
import os.path as osp
import random
import sys
import pickle
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from datasets.augmentation import Scale, ToTensor, Normalize
except:
    from augmentation import Scale, ToTensor, Normalize


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


multi_frames_transform = transforms.Compose(
    [Scale(size=(224, 224)), ToTensor(), Normalize()]
)


class KineticsGEBDMulFrames(Dataset):
    def __init__(
        self,
        mode="train",
        dataroot="PATH_TO/Kinetics_GEBD_frame",
        frames_per_side=5,
        tmpl="img_{:05d}.png",
        transform=None,
        seed=666,
        args=None,
    ):
        assert mode.lower() in ["train", "val", "valnew", "test"], "Wrong mode for k400"
        self.mode = mode
        self.split_folder = mode + "_" + "split"
        self.train = self.mode.lower() == "train"
        self.dataroot = dataroot
        if self.mode == "train":
            self.dataroot = "../../data/kinetics_GEBD_RGB/train"
        if self.mode == "val":
            self.dataroot = "../../data/kinetics_GEBD_RGB/val"
        if self.mode == "test":
            self.dataroot = "../../data/kinetics_GEBD_RGB/test"
        self.frame_per_side = frames_per_side
        self.tmpl = tmpl
        # assert negtive_step > 0, f'negtive_step = {negtive_step} is illegal!'
        # self.negtive_step = negtive_step
        self.seed = seed
        self.train_file = "multi-frames-GEBD-train-{}.pkl".format(frames_per_side)
        self.val_file = "multi-frames-GEBD-{}-{}.pkl".format(mode, frames_per_side)
        self.load_file = self.train_file if self.mode == "train" else self.val_file
        self.load_file_path = os.path.join("./DataAssets", self.load_file)

        if not (
            os.path.exists(self.load_file_path) and os.path.isfile(self.load_file_path)
        ):
            if (args is not None and args.rank == 0) or args is None:
                print("Preparing pickle file ...")
                self._prepare_pickle(
                    anno_path="../../data/k400_mr345_{}_min_change_duration0.3.pkl".format(
                        mode
                    ),
                    downsample=3,
                    min_change_dur=0.3,
                    keep_rate=1,
                    load_file_path=self.load_file_path,
                )
        if transform is not None:
            self.transform = transform
        else:
            self.transform = multi_frames_transform

        self.seqs = pickle.load(open(self.load_file_path, "rb"), encoding="lartin1")
        self.seqs = np.array(self.seqs, dtype=object)

        self.labels_set = list(np.arange(args.num_classes))
        if self.mode == "train":
            self.train_labels = torch.LongTensor([dta["label"] for dta in self.seqs])
            self.label_to_indices = {
                label: np.where(self.train_labels.numpy() == label)[0]
                for label in self.labels_set
            }
            self.ratios = [
                len(self.label_to_indices[0]) / len(self.label_to_indices[1]),
                1,
            ]
        elif self.mode == "val":
            self.val_labels = torch.LongTensor([dta["label"] for dta in self.seqs])
            self.label_to_indices = {
                label: np.where(self.val_labels.numpy() == label)[0]
                for label in self.labels_set
            }
        elif self.mode == "test":
            self.test_labels = torch.LongTensor([dta["label"] for dta in self.seqs])
            self.label_to_indices = {
                label: np.where(self.test_labels.numpy() == label)[0]
                for label in self.labels_set
            }

        self.img = None

    def _get_training_samples(self, index):
        indices = []
        for class_ in self.labels_set:
            real_index = self.label_to_indices[class_][int(index * self.ratios[class_])]
            indices.append(real_index)
        return indices

    def _read_data(self, index):
        item = self.seqs[index]
        block_idx = item["block_idx"]
        folder = item["folder"]
        current_idx = item["current_idx"]

        img = self.transform(
            [pil_loader(os.path.join(folder, self.tmpl.format(i))) for i in block_idx]
        )
        img = torch.stack(img, dim=0)

        return img, item["label"], os.path.join(folder, self.tmpl.format(current_idx))

    def __getitem__(self, index):
        if self.train:
            indices = self._get_training_samples(index)
            img_list = []
            label_list = []
            path_list = []
            for real_index in indices:
                img, label, img_path = self._read_data(real_index)
                img_list.append(img)
                label_list.append(label)
                path_list.append(img_path)
        else:
            img, label, img_path = self._read_data(index)
            img_list = [
                img,
            ]
            label_list = [
                label,
            ]
            path_list = [
                img_path,
            ]

        return {
            "inp": torch.stack(img_list, dim=0),
            "label": torch.LongTensor(label_list),
            "path": path_list,
        }

    def shuffle(self):
        np.random.seed(self.seed)
        for class_ in self.labels_set:
            np.random.shuffle(self.label_to_indices[class_])

    def __len__(self):
        if self.mode == "train":
            return len(self.label_to_indices[1])
        # from functools import reduce
        # return reduce(sum, [len(v) for v in self.label_to_indices.values()])
        return sum([len(v) for v in self.label_to_indices.values()])

    def _prepare_pickle(
        self,
        anno_path="/PATH_TO/k400_mr345_train_min_change_duration0.3.pkl",
        downsample=3,
        min_change_dur=0.3,
        keep_rate=0.8,
        load_file_path="./data/multi-frames-train.pkl",
    ):
        # prepare file for multi-frames-GEBD
        # dict_train_ann
        with open(anno_path, "rb") as f:
            dict_train_ann = pickle.load(f, encoding="lartin1")

        # downsample factor: sample one every `ds` frames
        ds = downsample

        SEQ = []
        neg = 0
        pos = 0

        for vname in dict_train_ann.keys():
            if not osp.exists(osp.join(self.dataroot, vname)):
                continue

            vdict = dict_train_ann[vname]
            vlen = vdict["num_frames"]
            vlen = min(vlen, len(os.listdir(osp.join(self.dataroot, vname))))
            fps = vdict["fps"]
            f1_consis = vdict["f1_consis"]
            path_frame = vdict["path_frame"]

            cls, frame_folder = path_frame.split("/")[:2]

            # select the annotation with highest f1 score
            highest = np.argmax(f1_consis)
            change_idices = vdict["substages_myframeidx"][highest]

            # (float)num of frames with min_change_dur/2
            half_dur_2_nframes = min_change_dur * fps / 2.0
            # (int)num of frames with min_change_dur/2
            ceil_half_dur_2_nframes = int(np.ceil(half_dur_2_nframes))

            start_offset = np.random.choice(ds) + 1
            selected_indices = np.arange(start_offset, vlen, ds)

            # idx chosen after from downsampling falls in the time range [change-dur/2, change+dur/2]
            # should be tagged as positive(bdy), otherwise negative(bkg)
            GT = []
            for i in selected_indices:
                GT.append(0)
                for change in change_idices:
                    if (
                        i >= change - half_dur_2_nframes
                        and i <= change + half_dur_2_nframes
                    ):
                        GT.pop()  # pop '0'
                        GT.append(1)
                        break
            # assert(len(selected_indices)==len(GT),'length frame indices is not equal to length GT.')
            assert len(selected_indices) == len(
                GT
            ), "length frame indices is not equal to length GT."

            for idx, (current_idx, lbl) in enumerate(zip(selected_indices, GT)):
                # for multi-frames input
                if self.train and random.random() > keep_rate:
                    continue

                record = dict()
                shift = np.arange(-self.frame_per_side, self.frame_per_side)
                shift[shift >= 0] += 1
                shift = shift * ds
                block_idx = shift + current_idx
                block_idx[block_idx < 1] = 1
                block_idx[block_idx > vlen] = vlen
                block_idx = block_idx.tolist()

                record["folder"] = f"{cls}/{frame_folder}"
                record["current_idx"] = current_idx
                record["block_idx"] = block_idx
                record["label"] = lbl
                SEQ.append(record)

                if lbl == 0:
                    neg += 1
                else:
                    pos += 1
        print(f" #bdy-{pos}\n #bkg-{neg}\n #total-{pos+neg}.")
        folder = "/".join(load_file_path.split("/")[:-1])
        if not os.path.exists(folder):
            os.makedirs(folder)
        pickle.dump(SEQ, open(load_file_path, "wb"))
        print(len(SEQ))


class TaposGEBDMulFrames(Dataset):
    def __init__(
        self,
        mode="train",
        dataroot="PATH_TO/TAPOS_GEBD_frame",
        frames_per_side=5,
        tmpl="img_{:05d}.png",
        transform=None,
        seed=666,
        args=None,
    ):
        assert mode.lower() in ["train", "val"], "Wrong mode for TAPOS"
        self.mode = mode
        self.split_folder = mode + "_" + "split"
        self.train = self.mode.lower() == "train"
        self.dataroot = dataroot
        self.frame_per_side = frames_per_side
        self.tmpl = tmpl
        # assert negtive_step > 0, f'negtive_step = {negtive_step} is illegal!'
        # self.negtive_step = negtive_step
        self.seed = seed
        self.train_file = "multi-frames-TAPOS-GEBD-train-{}.pkl".format(frames_per_side)
        self.val_file = "multi-frames-TAPOS-GEBD-{}-{}.pkl".format(mode, frames_per_side)
        self.load_file = self.train_file if self.mode == "train" else self.val_file
        self.load_file_path = os.path.join("./DataAssets", self.load_file)

        if not (
            os.path.exists(self.load_file_path) and os.path.isfile(self.load_file_path)
        ):
            if (args is not None and args.rank == 0) or args is None:
                print("Preparing pickle file ...")
                self._prepare_pickle(
                    anno_path="PATH_TO/TAPOS_{}_anno.pkl".format(
                        mode
                    ),
                    downsample=3,
                    min_change_dur=0.3,
                    keep_rate=1,
                    load_file_path=self.load_file_path,
                )
        if transform is not None:
            self.transform = transform
        else:
            self.transform = multi_frames_transform

        self.seqs = pickle.load(open(self.load_file_path, "rb"), encoding="lartin1")
        self.seqs = np.array(self.seqs, dtype=object)

        self.labels_set = list(np.arange(args.num_classes))
        if self.mode == "train":
            self.train_labels = torch.LongTensor([dta["label"] for dta in self.seqs])
            self.label_to_indices = {
                label: np.where(self.train_labels.numpy() == label)[0]
                for label in self.labels_set
            }
            self.ratios = [
                len(self.label_to_indices[0]) / len(self.label_to_indices[1]),
                1,
            ]
        elif self.mode == "val":
            self.val_labels = torch.LongTensor([dta["label"] for dta in self.seqs])
            self.label_to_indices = {
                label: np.where(self.val_labels.numpy() == label)[0]
                for label in self.labels_set
            }
        elif self.mode == "test":
            self.test_labels = torch.LongTensor([dta["label"] for dta in self.seqs])
            self.label_to_indices = {
                label: np.where(self.test_labels.numpy() == label)[0]
                for label in self.labels_set
            }

        self.img = None

    def _get_training_samples(self, index):
        indices = []
        for class_ in self.labels_set:
            real_index = self.label_to_indices[class_][int(index * self.ratios[class_])]
            indices.append(real_index)
        return indices

    def _read_data(self, index):
        item = self.seqs[index]
        block_idx = item['block_idx']
        folder = item['folder']
        current_idx = item['current_idx']
        # '''
        
        img = self.transform([pil_loader(
            os.path.join(folder, self.tmpl.format(i))
        ) for i in block_idx])
        img = torch.stack(img, dim=0)
        # '''
        # print('img = ', img.shape)
        # if self.img is None:
        #     img = self.transform([pil_loader(
        #         os.path.join(folder, self.tmpl.format(i))
        #     ) for i in block_idx])
        #     img = torch.stack(img, dim=0)
        #     self.img = img
        # else:
        #     img = self.img
        return img, item['label'], os.path.join(folder, self.tmpl.format(current_idx))

    def __getitem__(self, index):
        if self.train:
            indices = self._get_training_samples(index)
            img_list = []
            label_list = []
            path_list = []
            for real_index in indices:
                img, label, img_path = self._read_data(real_index)
                img_list.append(img)
                label_list.append(label)
                path_list.append(img_path)
        else:
            img, label, img_path = self._read_data(index)
            img_list = [img, ]
            label_list = [label, ]
            path_list = [img_path, ]
        # print('img2 = ', torch.stack(img_list, dim=0).shape)
        return {
            'inp': torch.stack(img_list, dim=0),
            'label': torch.LongTensor(label_list),
            'path': path_list
        } 

    def shuffle(self):
        np.random.seed(self.seed)
        for class_ in self.labels_set:
            np.random.shuffle(self.label_to_indices[class_])

    def __len__(self):
        if self.mode == "train":
            return len(self.label_to_indices[1])
        # from functools import reduce
        # return reduce(sum, [len(v) for v in self.label_to_indices.values()])
        return sum([len(v) for v in self.label_to_indices.values()])

    def _prepare_pickle(
        self,
        anno_path="/PATH_TO/TAPOS/save_output/TAPOS_train_anno.pkl",
        downsample=3,
        min_change_dur=0.3,
        keep_rate=0.8,
        load_file_path="./data/multi-frames-train.pkl",
    ):
        # prepare file for multi-frames-GEBD
        # dict_train_ann
        with open(anno_path, "rb") as f:
            dict_train_ann = pickle.load(f, encoding="lartin1")
        
        # Some fields in anno for reference
        # {'raw': {'action': 11, 'substages': [0, 79, 195], 'total_frames': 195, 'shot_timestamps': [43.36, 53.48], 'subset': 'train'},
        # 'path': 'yMK2zxDDs2A/s00004_0_100_7_931',
        # 'myfps': 25.0,
        # 'my_num_frames': 197,
        # 'my_duration': 7.88,
        # 'my_substages_frameidx': [79]
        # }
        
        # downsample factor: sample one every `ds` frames
        ds = downsample

        SEQ = []
        neg = 0
        pos = 0

        for vname in dict_train_ann.keys():
            if not osp.exists(osp.join(self.dataroot, vname)):
                continue

            vdict = dict_train_ann[vname]
            vlen = vdict["my_num_frames"]
            vlen = min(vlen, len(os.listdir(osp.join(self.dataroot, vname))))
            fps = vdict["myfps"]
            path_frame = vdict["path"]

            change_idices = vdict["my_substages_frameidx"]

            # (float)num of frames with min_change_dur/2
            half_dur_2_nframes = min_change_dur * fps / 2.0
            # (int)num of frames with min_change_dur/2
            ceil_half_dur_2_nframes = int(np.ceil(half_dur_2_nframes))

            start_offset = np.random.choice(ds) + 1
            selected_indices = np.arange(start_offset, vlen, ds)

            # idx chosen after from downsampling falls in the time range [change-dur/2, change+dur/2]
            # should be tagged as positive(bdy), otherwise negative(bkg)
            GT = []
            for i in selected_indices:
                GT.append(0)
                for change in change_idices:
                    if (
                        i >= change - half_dur_2_nframes
                        and i <= change + half_dur_2_nframes
                    ):
                        GT.pop()  # pop '0'
                        GT.append(1)
                        break
            # assert(len(selected_indices)==len(GT),'length frame indices is not equal to length GT.')
            assert len(selected_indices) == len(
                GT
            ), "length frame indices is not equal to length GT."

            for idx, (current_idx, lbl) in enumerate(zip(selected_indices, GT)):
                # for multi-frames input
                if self.train and random.random() > keep_rate:
                    continue

                record = dict()
                shift = np.arange(-self.frame_per_side, self.frame_per_side)
                shift[shift >= 0] += 1
                shift = shift * ds
                block_idx = shift + current_idx
                block_idx[block_idx < 1] = 1
                block_idx[block_idx > vlen] = vlen
                block_idx = block_idx.tolist()

                record["folder"] = path_frame
                record["current_idx"] = current_idx
                record["block_idx"] = block_idx
                record["label"] = lbl
                SEQ.append(record)

                if lbl == 0:
                    neg += 1
                else:
                    pos += 1
        print(f" #bdy-{pos}\n #bkg-{neg}\n #total-{pos+neg}.")
        folder = "/".join(load_file_path.split("/")[:-1])
        if not os.path.exists(folder):
            os.makedirs(folder)
        pickle.dump(SEQ, open(load_file_path, "wb"))
        print(len(SEQ))


###################################################################################
# Dummy dataset for debugging
###################################################################################
class MultiFDummyDataSet(Dataset):
    def __init__(self, mode="train", transform=None, args=None):
        assert mode.lower() in ["train", "val", "test", "valnew"], "Wrong split"
        self.mode = mode
        self.train = self.mode.lower() == "train"
        self.args = args

        if transform is not None:
            self.transform = transform

        self.train_labels = torch.LongTensor(np.random.choice([0, 1], 1000000))
        self.val_labels = torch.LongTensor(np.random.choice([0, 1], 1000000))
        self.load_file = self.train_labels if self.mode == "train" else self.val_labels
        self.load_file = self.train_labels if self.mode == "train" else self.val_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, label) where target is class_index of the target class.
        """
        label = self.load_file[index]
        inp = torch.randn(10, 3, 224, 224)

        return {"inp": inp, "label": label}

    def __len__(self):
        return len(self.load_file)


if __name__ == "__main__":
    # KineticsGEBDMulFrames
    dataset = KineticsGEBDMulFrames(mode="val")
    print(dataset[24511])
