import os
import sys
import time
import argparse
from contextlib import suppress

import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import pickle
import torch.distributed as dist

import logging
from datetime import datetime
from collections import OrderedDict
import json

from utils.getter import getModel, getDataset, getDataLoader, data_prefetcher
from utils.optim_factory import create_optimizer
from utils.scheduler import create_scheduler
from utils.checkpoint_saver import CheckpointSaver
from utils.log import setup_default_logging
from utils.cuda import NativeScaler
from utils.model_ema import ModelEmaV2
from utils.clip_grad import dispatch_clip_grad
from utils.helper import resume_checkpoint, load_checkpoint, get_outdir
from utils.helper import update_summary, model_parameters
from utils.dist_utils import setup_distributed, reduce_tensor, distribute_bn
from utils.dist_utils import get_local_rank, get_rank, get_world_size
from utils.metric import AverageMeter, accuracy

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

parser = argparse.ArgumentParser(description="Training Config", add_help=False)

# Distributed
parser.add_argument(
    "--dist_type", type=str, choices=["pytorch", "slurm"], default="slurm"
)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--port", type=int, default=150001)
# Dist_bn
parser.add_argument(
    "--sync-bn",
    action="store_true",
    default=False,
    help="Enable NVIDIA Apex or Torch synchronized BatchNorm.",
)

# Dataset / Model parameters
parser.add_argument(
    "--data_dir", metavar="DIR", default="/kinetics400/", help="path to dataset"
)
parser.add_argument(
    "--dataset",
    "-d",
    metavar="NAME",
    default="",
    help="dataset type (default: ImageFolder/ImageTar if empty)",
)
parser.add_argument(
    "--train-split",
    metavar="NAME",
    default="train",
    help="dataset train split (default: train)",
)
parser.add_argument(
    "--val-split",
    metavar="NAME",
    default="val",
    help="dataset validation split (default: validation)",
)
parser.add_argument(
    "--crop-pct",
    default=None,
    type=float,
    metavar="N",
    help="Input image center crop percent (for validation only)",
)
parser.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override mean pixel value of dataset",
)
parser.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of of dataset",
)
parser.add_argument(
    "--interpolation",
    default="",
    type=str,
    metavar="NAME",
    help="Image resize interpolation type (overrides model)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=32,
    metavar="N",
    help="input batch size for training (default: 32)",
)
parser.add_argument(
    "-vb",
    "--validation-batch-size-multiplier",
    type=int,
    default=1,
    metavar="N",
    help="ratio of validation batch size to training batch size (default: 1)",
)

parser.add_argument(
    "--model",
    default="resnet50",
    type=str,
    metavar="MODEL",
    help='Name of model to train (default: "countception"',
)
parser.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Start with pretrained version of specified network (if avail)",
)
parser.add_argument(
    "--initial-checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="Initialize model from this checkpoint (default: none)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Resume full model and optimizer state from checkpoint (default: none)",
)
parser.add_argument(
    "--no-resume-opt",
    action="store_true",
    default=False,
    help="prevent resume of optimizer state when resuming model",
)
parser.add_argument(
    "--num-classes",
    type=int,
    default=2,
    metavar="N",
    help="number of label classes (Model default if None)",
)
parser.add_argument(
    "--img-size",
    type=int,
    default=None,
    metavar="N",
    help="Image patch size (default: None => model default)",
)

# Optimizer parameters
parser.add_argument(
    "--opt",
    default="adamw",
    type=str,
    metavar="OPTIMIZER",
    help='Optimizer (default: "sgd"',
)
parser.add_argument(
    "--opt-eps",
    default=None,
    type=float,
    metavar="EPSILON",
    help="Optimizer Epsilon (default: None, use opt default)",
)
parser.add_argument(
    "--opt-betas",
    default=None,
    type=float,
    nargs="+",
    metavar="BETA",
    help="Optimizer Betas (default: None, use opt default)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="Optimizer momentum (default: 0.9)",
)
parser.add_argument(
    "--weight-decay", type=float, default=0.0001, help="weight decay (default: 0.0001)"
)
parser.add_argument(
    "--clip-grad",
    type=float,
    default=None,
    metavar="NORM",
    help="Clip gradient norm (default: None, no clipping)",
)
parser.add_argument(
    "--eval-freq", type=int, default=1, help="The frequency of evaluation."
)

parser.add_argument(
    "--clip-mode",
    type=str,
    default="norm",
    help='Gradient clipping mode. One of ("norm", "value", "agc")',
)

# Learning rate schedule parameters
parser.add_argument(
    "--sched",
    default="step",
    type=str,
    metavar="SCHEDULER",
    help='LR scheduler (default: "step"',
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--lr-noise",
    type=float,
    nargs="+",
    default=None,
    metavar="pct, pct",
    help="learning rate noise on/off epoch percentages",
)
parser.add_argument(
    "--lr-noise-pct",
    type=float,
    default=0.67,
    metavar="PERCENT",
    help="learning rate noise limit percent (default: 0.67)",
)
parser.add_argument(
    "--lr-noise-std",
    type=float,
    default=1.0,
    metavar="STDDEV",
    help="learning rate noise std-dev (default: 1.0)",
)
parser.add_argument(
    "--lr-cycle-mul",
    type=float,
    default=1.0,
    metavar="MULT",
    help="learning rate cycle len multiplier (default: 1.0)",
)
parser.add_argument(
    "--lr-cycle-limit",
    type=int,
    default=1,
    metavar="N",
    help="learning rate cycle limit",
)
parser.add_argument(
    "--warmup-lr",
    type=float,
    default=0.0001,
    metavar="LR",
    help="warmup learning rate (default: 0.0001)",
)
parser.add_argument(
    "--min-lr",
    type=float,
    default=1e-5,
    metavar="LR",
    help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    metavar="N",
    help="number of epochs to train (default: 2)",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--decay-epochs",
    type=float,
    default=30,
    metavar="N",
    help="epoch interval to decay LR",
)
parser.add_argument(
    "--warmup-epochs",
    type=int,
    default=2,
    metavar="N",
    help="epochs to warmup LR, if scheduler supports",
)
parser.add_argument(
    "--cooldown-epochs",
    type=int,
    default=10,
    metavar="N",
    help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
)
parser.add_argument(
    "--patience-epochs",
    type=int,
    default=10,
    metavar="N",
    help="patience epochs for Plateau LR scheduler (default: 10",
)
parser.add_argument(
    "--decay-rate",
    "--dr",
    type=float,
    default=0.1,
    metavar="RATE",
    help="LR decay rate (default: 0.1)",
)

# Model Exponential Moving Average
parser.add_argument(
    "--model-ema",
    action="store_true",
    default=False,
    help="Enable tracking moving average of model weights",
)
parser.add_argument(
    "--model-ema-force-cpu",
    action="store_true",
    default=False,
    help="Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.",
)
parser.add_argument(
    "--model-ema-decay",
    type=float,
    default=0.9998,
    help="decay factor for model weights moving average (default: 0.9998)",
)

# Training via mixed precision
parser.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="use NVIDIA Apex AMP or Native AMP for mixed precision training",
)
parser.add_argument(
    "--apex-amp",
    action="store_true",
    default=False,
    help="Use NVIDIA Apex AMP mixed precision",
)
parser.add_argument(
    "--native-amp",
    action="store_true",
    default=False,
    help="Use Native Torch AMP mixed precision",
)

# Balanced batch sampler
parser.add_argument(
    "--balance-batch",
    action="store_true",
    default=False,
    help="use balanced batch sampler for training",
)
parser.add_argument(
    "--n-sample-classes",
    type=int,
    default=2,
    help="#num of classes sampled in one batch",
)
parser.add_argument(
    "--n-samples",
    type=int,
    default=16,
    help="#samples per class in balanced batch sampling.",
)
# Misc
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=50,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--recovery-interval",
    type=int,
    default=0,
    metavar="N",
    help="how many batches to wait before writing recovery checkpoint",
)
parser.add_argument(
    "--checkpoint-hist",
    type=int,
    default=25,
    metavar="N",
    help="number of checkpoints to keep (default: 10)",
)
parser.add_argument(
    "-j",
    "--workers",
    type=int,
    default=8,
    metavar="N",
    help="how many training processes to use (default: 1)",
)
parser.add_argument(
    "--save-images",
    action="store_true",
    default=False,
    help="save images of input bathes every log interval for debugging",
)
parser.add_argument(
    "--channels-last",
    action="store_true",
    default=False,
    help="Use channels_last memory layout",
)
parser.add_argument(
    "--num-workers", type=int, default=8, help="num workers of dataloader."
)
parser.add_argument(
    "--pin-memory",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
parser.add_argument(
    "--use-prefetcher",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--output",
    default="",
    type=str,
    metavar="PATH",  ### config this for output folder
    help="path to output folder (default: none, current dir)",
)
parser.add_argument(
    "--bdy-scores-dir",
    default="bdy_scores",
    type=str,
    metavar="PATH",  ### config this for output folder
    help="path to output folder (default: none, current dir)",
)
parser.add_argument(
    "--eval-metric",
    default="top1",
    type=str,
    metavar="EVAL_METRIC",
    help='Best metric (default: "top1"',
)
parser.add_argument(
    "--tta",
    type=int,
    default=0,
    metavar="N",
    help="Test/inference time augmentation (oversampling) factor. 0=None (default: 0)",
)
parser.add_argument(
    "--use-multi-epochs-loader",
    action="store_true",
    default=False,
    help="use the multi-epochs-loader to save time at the beginning of every epoch",
)
parser.add_argument(
    "--torchscript",
    dest="torchscript",
    action="store_true",
    help="convert model torchscript for inference",
)

args = parser.parse_args()

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger("train")


def main():
    args = parser.parse_args()
    setup_default_logging(log_path=os.path.join(args.output, "log.txt"))

    if args.distributed:
        port = os.environ.get("MASTER_PORT", None)
        if port is not None:
            args.port = port
        setup_distributed(args.dist_type, port=args.port, local_rank=args.local_rank)
        args.world_size = get_world_size()
        args.rank = get_rank()
        _logger.info(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        args.device = "cuda:0"
        args.rank = 0  # global rank
        args.world_size = 1
        _logger.info("Training with a single process on 1 GPUs.")
    assert args.rank >= 0
    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp and has_native_amp:
        use_amp = "native"
        _logger.info(f"Using Pytorch {torch.__version__} amp...")

    torch.manual_seed(args.seed + args.rank)

    model = getModel(model_name=args.model, args=args)

    if args.rank == 0:
        _logger.info(
            "Model %s created, param count: %d"
            % (args.model, sum([m.numel() for m in model.parameters()]))
        )

    # move model to gpu
    model.cuda()

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.rank == 0:
            _logger.info(
                "Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using "
                "zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled."
            )

    # optimizer
    optimizer = create_optimizer(args, model)

    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == "native":
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.rank == 0:
            _logger.info("Using native Torch AMP. Training in mixed precision.")
    else:
        if args.rank == 0:
            _logger.info("AMP not enabled. Training in float32.")

    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.rank == 0,
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else None,
        )
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # set up distributed training
    if args.distributed:
        if args.rank == 0:
            _logger.info("Using native Torch DistributedDataParallel.")
        model = NativeDDP(
            model,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            find_unused_parameters=True,
        )  # can use device str in Torch >= 1.1

    # lr schedule
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.rank == 0:
        _logger.info("Scheduled epochs: {}".format(num_epochs))

    # create the train and eval dataset
    dataset_train = getDataset(
        dataset_name=args.dataset, mode=args.train_split, args=args
    )
    dataset_eval = getDataset(dataset_name=args.dataset, mode=args.val_split, args=args)

    # create loader
    loader_train = getDataLoader(dataset_train, is_training=True, args=args)
    loader_eval = getDataLoader(dataset_eval, is_training=False, args=args)

    # set_up loss function
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # set_up checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = ""
    if args.rank == 0:
        output_base = args.output if args.output else "./output"
        exp_name = "-".join(
            [
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                args.dataset,
                args.model,
            ]
        )
        output_dir = get_outdir(output_base, "train", exp_name)
        decreasing = True if eval_metric == "loss" else False
        saver = CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist,
        )

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, "set_epoch"):
                loader_train.sampler.set_epoch(epoch)

            loader_train.dataset.shuffle()
            train_metrics = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                args,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_ema=model_ema,
            )
            eval_metrics = dict()
            if epoch % args.eval_freq == 0:
                eval_metrics = validate(
                    model,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    epoch,
                    output_dir=output_dir,
                    amp_autocast=amp_autocast,
                )

                if model_ema is not None and not args.model_ema_force_cpu:
                    if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                        distribute_bn(
                            model_ema, args.world_size, args.dist_bn == "reduce"
                        )
                    ema_eval_metrics = validate(
                        model_ema.module,
                        loader_eval,
                        validate_loss_fn,
                        args,
                        amp_autocast=amp_autocast,
                        log_suffix=" (EMA)",
                    )
                    eval_metrics = ema_eval_metrics

            if args.rank == 0 and lr_scheduler is not None:
                metric_of_this_batch = eval_metrics.get(eval_metric, None)
                lr_scheduler.step(epoch + 1, metric_of_this_batch)

            if args.rank == 0:
                update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    os.path.join(output_dir, "summary.csv"),
                    write_header=best_metric is None,
                )

            if saver is not None and args.rank == 0:
                # save proper checkpoint with eval metric
                if eval_metrics.get(eval_metric, None) is not None:
                    save_metric = eval_metrics[eval_metric]
                else:
                    save_metric = -1
                best_metric, best_epoch = saver.save_checkpoint(
                    epoch, metric=save_metric
                )

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        _logger.info("*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch))


def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    args,
    lr_scheduler=None,
    saver=None,
    output_dir="",
    amp_autocast=suppress,
    loss_scaler=None,
    model_ema=None,
):

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)

    prefetcher = data_prefetcher(loader)
    input, target, path = prefetcher.next()
    batch_idx = 0

    while input is not None:
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        input = input.view((-1,) + input.size()[2:])
        target = target.view((-1,))
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            loss = 0

            outputs, rgbs, ddms = model(input)
            for output in outputs:
                loss += loss_fn(output, target)
            for rgb in rgbs:
                loss += loss_fn(rgb, target)
            for ddm in ddms:
                loss += loss_fn(ddm, target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=args.clip_grad,
                clip_mode=args.clip_mode,
                parameters=model_parameters(
                    model, exclude_head="agc" in args.clip_mode
                ),
                create_graph=second_order,
            )
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head="agc" in args.clip_mode),
                    value=args.clip_grad,
                    mode=args.clip_mode,
                )
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.rank == 0:
                _logger.info(
                    "Train: {} [{:>4d}/{} ({:>3.0f}%)]  "
                    "Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  "
                    "Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  "
                    "({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                    "LR: {lr:.3e}  "
                    "Data: {data_time.val:.3f} ({data_time.avg:.3f})".format(
                        epoch,
                        batch_idx,
                        len(loader),
                        100.0 * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m,
                    )
                )

        if (
            saver is not None
            and args.recovery_interval
            and (last_batch or (batch_idx + 1) % args.recovery_interval == 0)
        ):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        batch_idx += 1

        if args.distributed:
            torch.distributed.barrier()
        input, target, path = prefetcher.next()
        # end for/while

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict([("loss", losses_m.avg)])


def validate(
    model,
    loader,
    loss_fn,
    args,
    epoch,
    output_dir,
    amp_autocast=suppress,
    log_suffix="",
):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    dataset = loader.dataset

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    results = []
    with torch.no_grad():
        prefetcher = data_prefetcher(loader)
        input, target, fetch_path = prefetcher.next()

        batch_idx = 0
        while input is not None:
            last_batch = batch_idx == last_idx
            input = input.view((-1,) + input.size()[2:])
            target = target.view((-1,))
            path = fetch_path[0]

            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                loss = 0

                output, rgbs, ddms = model(input)
                for outp in output:
                    loss += loss_fn(outp, target)
                for rgb in rgbs:
                    loss += loss_fn(rgb, target)
                for ddm in ddms:
                    loss += loss_fn(ddm, target)

            if isinstance(output, (tuple, list)):
                output = output[-1]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0 : target.size(0) : reduce_factor]

            acc1 = accuracy(output, target, topk=(1,))
            if isinstance(acc1, (list, tuple)):
                acc1 = acc1[0]
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()
            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = "Test" + log_suffix
                _logger.info(
                    "{0}: [{1:>4d}/{2}]  "
                    "Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  ".format(
                        log_name,
                        batch_idx,
                        len(loader),
                        batch_time=batch_time_m,
                        loss=losses_m,
                        top1=top1_m,
                    )
                )

            bdy_scores = F.softmax(output, dim=1)[:, 1].cpu().numpy()

            vname = []
            frame_indices = []
            scores = []
            for idx in range(len(path)):
                vname.append(path[idx].split("/")[-2])
                frame_indices.append(int(path[idx].split("/")[-1][4:9]))
                scores.append(float(bdy_scores[idx]))

            result = [vname, frame_indices, scores]
            results.append(result)

            torch.cuda.synchronize()

            batch_idx += 1
            input, target, fetch_path = prefetcher.next()

    results = collect_results_gpu(results, len(dataset))
    torch.distributed.barrier()

    if args.rank == 0:

        predictions = {}
        for result in results:
            vnames = result[0]
            frame_indices = result[1]
            scores = result[2]

            for idx in range(len(frame_indices)):
                vname = vnames[idx]
                if vname not in predictions:
                    predictions[vname] = {}
                if frame_indices[idx] not in predictions[vname]:
                    predictions[vname][frame_indices[idx]] = scores[idx]

        save_predictions(predictions, args, epoch, output_dir)

        if(args.dataset == 'kinetics_multiframes'):
            with open(
                "data/k400_mr345_val_min_change_duration0.3.pkl",
                "rb",
            ) as f:
                gt_dict = pickle.load(f, encoding="lartin1")
        elif(args.dataset == 'tapos_multiframes'):
             with open(
                "data/TAPOS_val_anno.pkl",
                "rb",
            ) as f:
                gt_dict = pickle.load(f, encoding="lartin1")

        if not os.path.exists(os.path.join(output_dir, args.bdy_scores_dir)):
            os.makedirs(os.path.join(output_dir, args.bdy_scores_dir))
        with open(
            os.path.join(
                output_dir, args.bdy_scores_dir, "epoch_" + str(epoch) + ".json"
            ),
            "r",
        ) as f:
            nms_result = json.load(f)
        if(args.dataset == 'kinetics_multiframes'):
            prec, rec, f1 = eval_F1(gt_dict, nms_result)
        elif(args.dataset == 'tapos_multiframes'):
            prec, rec, f1 = eval_TAPOS_F1(gt_dict, nms_result)
        metrics = OrderedDict(
            [
                ("F1_score", f1),
                ("loss", losses_m.avg),
                ("top1", top1_m.avg),
                ("precision", prec),
                ("recall", rec),
            ]
        )
        return metrics


def get_idx_from_score_by_threshold(
    scope=5, threshold=0.5, seq_indices=None, seq_scores=None
):
    seq_indices = np.array(seq_indices)
    seq_scores = np.array(seq_scores)
    bdy_indices = []
    internals_indices = []
    bdy_indices_in_video = []

    for i in range(2, len(seq_scores) - 2):
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


def eval_F1(gt_dict, pred_dict):
    # recall precision f1 for threshold 0.05(5%)
    threshold = 0.05
    tp_all = 0
    num_pos_all = 0
    num_det_all = 0

    for vid_id in list(gt_dict.keys()):

        # filter by avg_f1 score
        if gt_dict[vid_id]["f1_consis_avg"] < 0.3:
            continue

        if vid_id not in pred_dict.keys():
            num_pos_all += len(gt_dict[vid_id]["substages_timestamps"][0])
            continue

        # detected timestamps
        bdy_timestamps_det = pred_dict[vid_id]

        myfps = gt_dict[vid_id]["fps"]
        my_dur = gt_dict[vid_id]["video_duration"]
        ins_start = 0
        ins_end = my_dur

        # remove detected boundary outside the action instance
        tmp = []
        for det in bdy_timestamps_det:
            tmpdet = det + ins_start
            if tmpdet >= (ins_start) and tmpdet <= (ins_end):
                tmp.append(tmpdet)
        bdy_timestamps_det = tmp
        if bdy_timestamps_det == []:
            num_pos_all += len(gt_dict[vid_id]["substages_timestamps"][0])
            continue
        num_det = len(bdy_timestamps_det)
        num_det_all += num_det

        # compare bdy_timestamps_det vs. each rater's annotation, pick the one leading the best f1 score
        bdy_timestamps_list_gt_allraters = gt_dict[vid_id]["substages_timestamps"]
        f1_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
        tp_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
        num_pos_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))

        for ann_idx in range(len(bdy_timestamps_list_gt_allraters)):
            bdy_timestamps_list_gt = bdy_timestamps_list_gt_allraters[ann_idx]
            num_pos = len(bdy_timestamps_list_gt)
            tp = 0
            offset_arr = np.zeros(
                (len(bdy_timestamps_list_gt), len(bdy_timestamps_det))
            )
            for ann1_idx in range(len(bdy_timestamps_list_gt)):
                for ann2_idx in range(len(bdy_timestamps_det)):
                    offset_arr[ann1_idx, ann2_idx] = abs(
                        bdy_timestamps_list_gt[ann1_idx] - bdy_timestamps_det[ann2_idx]
                    )
            for ann1_idx in range(len(bdy_timestamps_list_gt)):
                if offset_arr.shape[1] == 0:
                    break
                min_idx = np.argmin(offset_arr[ann1_idx, :])
                if offset_arr[ann1_idx, min_idx] <= threshold * my_dur:
                    tp += 1
                    offset_arr = np.delete(offset_arr, min_idx, 1)

            num_pos_tmplist[ann_idx] = num_pos
            fn = num_pos - tp
            fp = num_det - tp
            if num_pos == 0:
                rec = 1
            else:
                rec = tp / (tp + fn)
            if (tp + fp) == 0:
                prec = 0
            else:
                prec = tp / (tp + fp)
            if (rec + prec) == 0:
                f1 = 0
            else:
                f1 = 2 * rec * prec / (rec + prec)
            tp_tmplist[ann_idx] = tp
            f1_tmplist[ann_idx] = f1

        ann_best = np.argmax(f1_tmplist)
        tp_all += tp_tmplist[ann_best]
        num_pos_all += num_pos_tmplist[ann_best]

    fn_all = num_pos_all - tp_all
    fp_all = num_det_all - tp_all
    if num_pos_all == 0:
        rec = 1
    else:
        rec = tp_all / (tp_all + fn_all)
    if (tp_all + fp_all) == 0:
        prec = 0
    else:
        prec = tp_all / (tp_all + fp_all)
    if (rec + prec) == 0:
        f1 = 0
    else:
        f1 = 2 * rec * prec / (rec + prec)
    return prec, rec, f1


def eval_TAPOS_F1(gt_dict, pred_dict):
    # recall precision f1 for threshold 0.05(5%)
    threshold = 0.05
    tp_all = 0
    num_pos_all = 0
    num_det_all = 0

    for vid_id in list(gt_dict.keys()):

        # filter by avg_f1 score
        #if gt_dict[vid_id]["f1_consis_avg"] < 0.3:
        #    continue

        if vid_id not in pred_dict.keys():
            num_pos_all += len(gt_dict[vid_id]["my_substages_frameidx"])
            continue

        # detected timestamps
        bdy_timestamps_det = pred_dict[vid_id]
        
        myfps = gt_dict[vid_id]["myfps"]
        my_dur = gt_dict[vid_id]["my_duration"]
        my_num_frames = gt_dict[vid_id]['my_num_frames']
        
        ins_start = 0
        ins_end = my_num_frames

        # remove detected boundary outside the action instance
        tmp = []
        for det in bdy_timestamps_det:
            tmpdet = det + ins_start
            if tmpdet >= (ins_start) and tmpdet <= (ins_end):
                tmp.append(tmpdet)
        bdy_timestamps_det = tmp
        if bdy_timestamps_det == []:
            num_pos_all += len(gt_dict[vid_id]["my_substages_frameidx"])
            continue
        num_det = len(bdy_timestamps_det)
        num_det_all += num_det

        # compare bdy_timestamps_det vs. each rater's annotation, pick the one leading the best f1 score
        bdy_timestamps_list_gt_allraters = [gt_dict[vid_id]["my_substages_frameidx"]]
        f1_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
        tp_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
        num_pos_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))

        for ann_idx in range(len(bdy_timestamps_list_gt_allraters)):
            bdy_timestamps_list_gt = bdy_timestamps_list_gt_allraters[ann_idx]
            num_pos = len(bdy_timestamps_list_gt)
            tp = 0
            offset_arr = np.zeros(
                (len(bdy_timestamps_list_gt), len(bdy_timestamps_det))
            )
            for ann1_idx in range(len(bdy_timestamps_list_gt)):
                for ann2_idx in range(len(bdy_timestamps_det)):
                    offset_arr[ann1_idx, ann2_idx] = abs(
                        bdy_timestamps_list_gt[ann1_idx] - bdy_timestamps_det[ann2_idx]
                    )
            for ann1_idx in range(len(bdy_timestamps_list_gt)):
                if offset_arr.shape[1] == 0:
                    break
                min_idx = np.argmin(offset_arr[ann1_idx, :])
                if offset_arr[ann1_idx, min_idx] <= threshold * my_num_frames:
                    tp += 1
                    offset_arr = np.delete(offset_arr, min_idx, 1)

            num_pos_tmplist[ann_idx] = num_pos
            fn = num_pos - tp
            fp = num_det - tp
            if num_pos == 0:
                rec = 1
            else:
                rec = tp / (tp + fn)
            if (tp + fp) == 0:
                prec = 0
            else:
                prec = tp / (tp + fp)
            if (rec + prec) == 0:
                f1 = 0
            else:
                f1 = 2 * rec * prec / (rec + prec)
            tp_tmplist[ann_idx] = tp
            f1_tmplist[ann_idx] = f1

        ann_best = np.argmax(f1_tmplist)
        tp_all += tp_tmplist[ann_best]
        num_pos_all += num_pos_tmplist[ann_best]

    fn_all = num_pos_all - tp_all
    fp_all = num_det_all - tp_all
    if num_pos_all == 0:
        rec = 1
    else:
        rec = tp_all / (tp_all + fn_all)
    if (tp_all + fp_all) == 0:
        prec = 0
    else:
        prec = tp_all / (tp_all + fp_all)
    if (rec + prec) == 0:
        f1 = 0
    else:
        f1 = 2 * rec * prec / (rec + prec)
    return prec, rec, f1


def save_predictions(predictions, args, epoch, output_dir):
    bdy_score_dir = args.bdy_scores_dir
    if not os.path.exists(os.path.join(output_dir, bdy_score_dir)):
        os.makedirs(os.path.join(output_dir, bdy_score_dir))

    result = {}
    for vid, info in predictions.items():
        result_dict = {}
        result_dict["frame_idx"] = []
        result_dict["scores"] = []

        for key in sorted(info.keys()):
            result_dict["frame_idx"].append(key)
            result_dict["scores"].append(info[key])
        result[vid] = result_dict
    if(args.dataset == 'kinetics_multiframes'):
        with open(
            "data/k400_mr345_val_min_change_duration0.3.pkl",
            "rb",
        ) as f:
            gt_dict = pickle.load(f, encoding="lartin1")
    elif(args.dataset == 'tapos_multiframes'):
        with open(
            "data/TAPOS_val_anno.pkl",
            "rb",
        ) as f:
            gt_dict = pickle.load(f, encoding="lartin1")

    filename = os.path.join(output_dir, "epoch_" + str(epoch) + ".pkl")
    print(filename)

    with open(filename, "wb") as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    nms_result = {}

    if(args.dataset == 'kinetics_multiframes'):
        for vid in result:
            if vid in gt_dict:
                # detect boundaries, convert frame_idx to timestamps
                fps = gt_dict[vid]["fps"]
                det_t = (
                    np.array(
                        get_idx_from_score_by_threshold(
                            threshold=0.5,
                            seq_indices=result[vid]["frame_idx"],
                            seq_scores=result[vid]["scores"],
                        )
                    )
                    / fps
                )
                nms_result[vid] = det_t.tolist()
    elif(args.dataset == 'tapos_multiframes'):
        for vid in result:
            if vid in gt_dict:
                # detect boundaries, convert frame_idx to timestamps
                fps = gt_dict[vid]["myfps"]
                det_t = (
                    np.array(
                        get_idx_from_score_by_threshold(
                            scope=11,
                            threshold=0.48,
                            seq_indices=result[vid]["frame_idx"],
                            seq_scores=result[vid]["scores"],
                        )
                    )
                )
                nms_result[vid] = det_t.tolist()

    with open(
        os.path.join(output_dir, bdy_score_dir, "epoch_" + str(epoch) + ".json"), "w"
    ) as f:
        json.dump(nms_result, f, sort_keys=True, indent=4)


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device="cuda"
    )
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device="cuda")
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device="cuda")
    part_send[: shape_tensor[0]] = part_tensor
    part_recv_list = [part_tensor.new_zeros(shape_max) for _ in range(world_size)]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(pickle.loads(recv[: shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


if __name__ == "__main__":
    main()
