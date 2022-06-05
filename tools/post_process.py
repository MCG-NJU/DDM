import json
import os
import pickle
import numpy as np
import argparse


def get_boundary_idx_per_video(
    scope, threshold, begin_ignore=2, end_ignore=3, seq_indices=None, seq_scores=None
):
    seq_indices = np.array(seq_indices)
    seq_scores = np.array(seq_scores)
    bdy_indices_in_video = []

    for i in range(begin_ignore, len(seq_scores) - end_ignore):
        if seq_scores[i] >= threshold:
            sign = 1
            for j in range(max(0, i - scope), min(i + scope + 1, len(seq_scores))):
                if seq_scores[j] > seq_scores[i]:
                    sign = 0
                if seq_scores[j] == seq_scores[i]:
                    if i != j:
                        sign = 0

            if sign == 1:
                bdy_indices_in_video.append(seq_indices[i])

    return bdy_indices_in_video


def parse_args():
    parser = argparse.ArgumentParser(description="extract optical flows")
    parser.add_argument(
        "result_pkl", type=str, help="path of result(boundary score sequence) pickle"
    )
    parser.add_argument("gt_pkl", type=str, help="path of ground truth pickle")
    parser.add_argument(
        "--scope",
        type=int,
        default=5,
        help="If the boundary probability of current frame is not the maximum within the scope/range, it will be suppressed.",
    )
    parser.add_argument("--threshold", type=float, default=0.53)
    parser.add_argument("--begin_ignore", type=int, default=2)
    parser.add_argument("--end_ignore", type=int, default=3)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    with open(args.result_pkl, "rb") as f2:
        predictions = pickle.load(f2, encoding="lartin1")

    with open(args.gt_pkl, "rb") as f:
        gt_dict = pickle.load(f, encoding="lartin1")

    nms_result = {}
    for vid in predictions:
        if vid in gt_dict:
            # detect boundaries, convert frame_idx to timestamps
            fps = gt_dict[vid]["fps"]
            det_t = (
                np.array(
                    get_boundary_idx_per_video(
                        scope=args.scope,
                        threshold=args.threshold,
                        begin_ignore=args.begin_ignore,
                        end_ignore=args.end_ignore,
                        seq_indices=predictions[vid]["frame_idx"],
                        seq_scores=predictions[vid]["scores"],
                    )
                )
                / fps
            )
            nms_result[vid] = det_t.tolist()
    print(len(nms_result))
    pickle.dump(
        nms_result, open("submission.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL
    )
