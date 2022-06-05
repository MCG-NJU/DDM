import os
import random
import subprocess
import torch
import torch.distributed as dist


def get_ip(ip_str):
    """
    input format: SH-IDC1-10-5-30-[137,152] or SH-IDC1-10-5-30-[137-142,152] or SH-IDC1-10-5-30-[152, 137-142]
    output format 10.5.30.137
    """
    import re

    # return ".".join(ip_str.replace("[", "").split(',')[0].split("-")[2:])
    return ".".join(re.findall(r"\d+", ip_str)[1:5])


def setup_distributed(dist_type="slurm", backend="nccl", port=None, local_rank=-1):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if dist_type == "slurm":
        rank = int(os.environ["SLURM_PROCID"])

        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        # ip = get_ip(os.environ['SLURM_STEP_NODELIST'])
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is None:
            port = 25001
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = f"{port}"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        # os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        if local_rank == -1:
            local_rank = rank % num_gpus
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )


def get_local_rank():
    if "LOCAL_RANK" in os.environ:
        return os.environ["LOCAL_RANK"]
    elif "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])
    return 0


def get_rank():
    return torch.distributed.get_rank()


def get_world_size():
    world_size = dist.get_world_size()
    if world_size != -1:
        return world_size
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    elif "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"])
    return 1


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def distribute_bn(model, world_size, reduce=False):
    # ensure every node has the same running bn stats
    for bn_name, bn_buf in unwrap_model(model).named_buffers(recurse=True):
        if ("running_mean" in bn_name) or ("running_var" in bn_name):
            if reduce:
                # average bn stats across whole group
                torch.distributed.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                bn_buf /= float(world_size)
            else:
                # broadcast bn stats from rank 0 to whole group
                torch.distributed.broadcast(bn_buf, 0)
