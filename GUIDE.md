## GUIDE for DDM-Net usage

Here is the guide for DDM-Net on Kinetics-GEBD dataset, you can also refer to [instructions of LOVEU Challenge](https://github.com/StanLei52/GEBD/blob/main/INSTRUCTIONS.md) for help.

### 1. Data Preparation

**1-a**. Download [Kinetics-GEBD annotation](https://drive.google.com/drive/folders/1AlPr63Q9D-HAGc5bOUNTzjCiWOC1a3xo): `k400_train_raw_annotation.pkl` and `k400_val_raw_annotation.pkl`.

**1-b**. Download videos listed in the [Kinetics-GEBD annotation](https://drive.google.com/drive/folders/1AlPr63Q9D-HAGc5bOUNTzjCiWOC1a3xo). Note that videos in the Kinetics-GEBD dataset are a subset of Kinetics-400 dataset. You can either download the whole Kinetics-400 dataset or just download the part in Kinetics-GEBD dataset.

**1-c**. Trim the videos according to the Kinetics-400 annotations. E.g., after you downloading `3y1V7BNNBds.mp4`, trim this video into a 10-second video `3y1V7BNNBds_000000_000010.mp4` from 0s to 10s in the original video. Note that the start time and end time for each video can be found at the Kinetics-400 annotations.

**1-d**. Extract frames.
For example, use the following script to extract frames of video M99mgKKmPqs.
```
ffmpeg -i $video_root/M99mgKKmPqs.mp4 -f image2 -qscale:v 2 -loglevel quiet $frame_root/M99mgKKmPqs/img_%05d.png
```

**1-e**. Generate GT files `k400_mr345_train_min_change_duration0.3.pkl` and `k400_mr345_val_min_change_duration0.3.pkl` . Refer to  [prepare_k400_release.ipynb](https://github.com/StanLei52/GEBD/blob/main/data/export/prepare_k400_release.ipynb) for generating the GT files. Specifically, you should prepare the <u>train</u> and <u>val</u> split:

```python
generate_frameidx_from_raw(split='train')
generate_frameidx_from_raw(split='val')
```
To reproduce DDM-Net result, you can directly use [my GT files](https://drive.google.com/drive/folders/10daNvdsW1phKg9POh_gGhXQctel_eF3t?usp=sharing). (prepare_k400_release.ipynb was [changed](https://github.com/StanLei52/GEBD/issues/3) in 2022.5, the new annotation may lead to a slighly different result.)

### 2. Train your model

In this part, you can customize your own training procedure. We take our `DDM-Net` baseline as example here:

**2-a**. Check `PC/datasets/MultiFDataset.py` or use `tools/generate_pickle.py` to generate pickle files for training, validation and testing.
```
python tools/generate_pickle.py $TRAIN_DATA_ROOT --split train --anno_path $TRAIN_ANNO_PATH
python tools/generate_pickle.py $VAL_DATA_ROOT --split val --anno_path $VAL_ANNO_PATH
python tools/generate_pickle.py $TEST_DATA_ROOT --split test
```
Place training, validation and testing pickles into the folder `DataAssets`.

**2-b**. Train on Kinetics-GEBD:

```shell
python DDM-Net/train.py \
--dataset kinetics_multiframes \
--train-split train \
--val-split val \
--num-classes 2 \
--batch-size 16 \
--n-sample-classes 2 \
--n-samples 16 \
--lr 0.00001 \
--warmup-epochs 0 \
--epochs 5 \
--decay-epochs 2 \
--model multiframes_resnet \
--pin-memory \
--sync-bn \
--amp \
--native-amp \
--distributed \
--eval-metric loss \
--log-interval 50 \
--port 16580 \
--eval-freq 1
```

**2-c**. Test on Kinetics-GEBD:

Generate the boundary score sequence for videos in validation/testing set with checkpoint:

```shell
python DDM-Net/test.py \
--dataset kinetics_multiframes \
--val-split val \
-b 128 \
--resume checkpoint.pth.tar
```



### 3. Generate Submission File and Evaluate(val).

Participants are required to submit their results in a pickle file. The pickle format is composed of a dictionary containing keys with video identifiers and values with boundary lists. For example,

```shell
{
‘6Tz5xfnFl4c’: [5.9, 9.4], # boundaries detected at 5.9s, 9.4s of this video
‘zJki61RMxcg’: [0.6, 1.5, 2.7] # boundaries detected at 0.6s, 1.5s, 2.7s of this video
...
}
```

Generally, you should generate such detect boundaries according to your own configuration. All you need is to generate the detected boundaries for each video. For our `PC` baseline, we obtained a score sequence for each video after executing `PC_test.py`. For example,

```shell
{
'S4hYqZbu0kQ': {'frame_idx': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99], 'scores': [0.87680495, 0.89294374, 0.7491671, 0.685366, 0.3733009, 0.118334584, 0.1789392, 0.32489958, 0.39417332, 0.59364367, 0.78907657, 0.38130152, 0.14496064, 0.07595703, 0.069147676, 0.017675582, 0.048309233, 0.15596619, 0.17298242, 0.046014242, 0.12774284, 0.20063624, 0.20420128, 0.41636664, 0.7791878, 0.8065185, 0.80875623, 0.794678, 0.6545196, 0.54669106, 0.60019726, 0.6882232, 0.52349806]}
...
}
```

For this video, we recorded the frame index after downsampling (`ds=3`) and the corresponding score. One can determine the position of boundaries according to the score sequence. A simple way is to set a threshold(i.e. 0.5) to filter out some low probability timestamps and keep those high. For our baseline, we group consecutive frames with high probability(i.e. 0.5) and mark their center as a boundary; also you can try to use Gaussian smoothing, etc.

We provide an example to obtain the submission file (just for reference).

**3-a**. Generate submission file from sequence score:
```
python tools/post_process.py $RESULT_PKL $GT_PKL
```

**3-b**. Evaluate F1-score of predicted boundaries:
```
python tools/eval.py $BOUNDARY_PKL $GT_PKL
```
