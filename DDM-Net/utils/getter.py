"""
Dataset getter
Dataloader getter
Model selector
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import dataset
import torchvision as tv
from modeling.resnetGEBD import resnetGEBD
from utils.sampler import (
    BalancedBatchSampler,
    OrderedDistributedSampler,
    DistBalancedBatchSampler,
)
from utils.augmentation import *
from datasets.MultiFDataset import (
    KineticsGEBDMulFrames,
    TaposGEBDMulFrames,
    MultiFDummyDataSet,
)
from torch.utils.data.dataloader import default_collate

transform_series = transforms.Compose(
    [
        Scale(size=(224, 224)),
        ToTensor(),
        Normalize(),  # mean std refer to utils/augmentation.py
    ]
)
transform_tv = tv.transforms.Compose(
    [
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def my_collate_fn(batch):
    batch = list(filter(lambda x: x["inp"] is not None, batch))
    if len(batch) == 0:
        return torch.Tensor()
    return default_collate(batch)


class PrefetchLoader:
    # prefetch dataloader for kinetics400-GEBD
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for item in self.loader:
            with torch.cuda.stream(stream):
                next_input = item["inp"].cuda(non_blocking=True)
                next_target = item["label"].cuda(non_blocking=True)
            if not first:
                yield {"inp": next_input, "label": next_target, "path": item["path"]}
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target
            path = item["path"]

        yield {"inp": input, "label": target, "path": path}

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


class data_prefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.nextitem = next(self.loader)
            self.next_input = self.nextitem["inp"]
            self.next_target = self.nextitem["label"]
            self.next_path = self.nextitem["path"]
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_path = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        path = self.next_path
        self.preload()
        return input, target, path


def getDataset(dataset_name, mode, args):
    if dataset_name.lower() == "multif_dummy":
        dataset = MultiFDummyDataSet(mode=mode)
    elif dataset_name.lower() == "kinetics_multiframes":
        dataroot = "../../data/kinetics_GEBD_RGB"  # FIXME
        dataset = KineticsGEBDMulFrames(
            mode=mode,
            dataroot=dataroot,
            frames_per_side=5,
            transform=transform_series,
            args=args,
        )
    elif dataset_name.lower() == "tapos_multiframes":
        dataroot = "/PATH_TO/TAPOS_instances_frame256"
        dataset = TaposGEBDMulFrames(
            mode=mode,
            dataroot=dataroot,
            frames_per_side=5,
            tmpl="image_{:05d}.jpg",
            transform=transform_series,
            args=args,
        )
    else:
        raise NotImplementedError
    return dataset


def getDataLoader_for_test(dataset, args=None):
    if args is not None:
        batchsize = args.batch_size
    else:
        batchsize = 64
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        collate_fn=my_collate_fn,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )
    return loader


def getDataLoader(dataset, is_training=True, args=None):
    sampler = None
    if args.distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            if args.balance_batch:
                assert args.batch_size == args.n_sample_classes * args.n_samples
                sampler = DistBalancedBatchSampler(
                    dataset,
                    args.num_classes,
                    args.n_sample_classes,
                    args.n_samples,
                    args.seed,
                )
                print(f"rank{args.rank} using balanced batch sampling.")
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            sampler = OrderedDistributedSampler(
                dataset,
            )

    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=args.batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset)
        and sampler is None
        and is_training,
        collate_fn=my_collate_fn,
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=is_training,
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop("persistent_workers")  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)
    return loader


def getModel(model_name="multiframes_resnet", args=None):
    if model_name.lower() == "multiframes_resnet":  # resnet-multi_frames
        model = resnetGEBD(
            backbone="resnet50",
            pretrained=True,
            num_classes=args.num_classes,
            frames_per_side=5,
        )
    # elif con:
    else:
        raise NotImplementedError("Model {} not implemented.".format(model_name))
    return model
