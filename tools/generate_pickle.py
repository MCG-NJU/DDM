from ast import arg
import pickle
import numpy as np
import random
import os
import os.path as osp
import argparse


def generate_test_pickle(root, frame_per_side=5, ds=3):
    frame_folders = os.listdir(root)

    SEQ = []
    for frame_folder in frame_folders:
        path = osp.join(root, frame_folder)
        vlen = len(os.listdir(path))

        start_offset = np.random.choice(ds) + 1
        selected_indices = np.arange(start_offset, vlen + 1, ds)

        for current_idx in selected_indices:
            record = dict()
            shift = np.arange(-frame_per_side, frame_per_side + 1)
            
            shift = shift * ds
            block_idx = shift + current_idx
            block_idx[block_idx < 1] = 1
            block_idx[block_idx > vlen] = vlen
            block_idx = block_idx.tolist()

            record["folder"] = path
            record["current_idx"] = current_idx
            record["block_idx"] = block_idx
            record["label"] = 0
            SEQ.append(record)

    pickle.dump(SEQ, open(f"multi-frames-GEBD-test-{frame_per_side}.pkl", "wb"))


def generate_pickle(
    split,
    anno_path,
    dataroot,
    frame_per_side=5,
    downsample=3,
    min_change_dur=0.3,
    keep_rate=1,
):
    frame_per_side = 5
    load_file_path = f"multi-frames-GEBD-{split}-{frame_per_side}.pkl"

    with open(anno_path, "rb") as f:
        dict_train_ann = pickle.load(f, encoding="lartin1")

    # downsample factor: sample one every `ds` frames
    ds = downsample

    SEQ = []
    neg = 0
    pos = 0

    for vname in dict_train_ann.keys():
        if not osp.exists(osp.join(dataroot, vname)):
            continue

        vdict = dict_train_ann[vname]
        vlen = vdict["num_frames"]
        vlen = min(vlen, len(os.listdir(osp.join(dataroot, vname))))
        fps = vdict["fps"]
        f1_consis = vdict["f1_consis"]
        path_frame = vdict["path_frame"]
        # print(path_frame.split('/'))
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
        assert (
            len(selected_indices) == len(GT),
            "length frame indices is not equal to length GT.",
        )

        for idx, (current_idx, lbl) in enumerate(zip(selected_indices, GT)):
            # for multi-frames input
            if random.random() > keep_rate:
                continue

            record = dict()
            shift = np.arange(-frame_per_side, frame_per_side + 1)
  
            shift = shift * ds
            block_idx = shift + current_idx
            block_idx[block_idx < 1] = 1
            block_idx[block_idx > vlen] = vlen
            block_idx = block_idx.tolist()

            record["folder"] = osp.join(dataroot, vname)
            record["current_idx"] = current_idx
            record["block_idx"] = block_idx
            record["label"] = lbl
            SEQ.append(record)

            if lbl == 0:
                neg += 1
            else:
                pos += 1

    print(f" #bdy-{pos}\n #bkg-{neg}\n #total-{pos+neg}.")
    # folder = '/'.join(load_file_path.split('/')[:-1])
    # if not os.path.exists(folder):
    #     os.makedirs(folder)

    pickle.dump(SEQ, open(load_file_path, "wb"))
    print(len(SEQ))


def parse_args():
    parser = argparse.ArgumentParser(description="generate pickles")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="generate pkl of which split",
    )
    parser.add_argument("dataroot", type=str, help="root of input data")
    parser.add_argument(
        "--anno_path", type=str, default=None, help="path of annotation"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    if args.split == "test":
        generate_test_pickle(args.dataroot)
    else:
        generate_pickle(
            split=args.split, dataroot=args.dataroot, anno_path=args.anno_path
        )
