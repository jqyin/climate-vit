import sys
import os
import time
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import ReduceOp

import logging
from utils import logging_utils

logging_utils.config_logger()
from utils.YParams import YParams
from utils import get_data_loader_distributed
from utils import comm
from utils.loss import l2_loss, l2_loss_opt
from utils.metrics import weighted_rmse
from networks import vit

from distributed.mappings import init_ddp_model_and_reduction_hooks
from distributed.helpers import init_params_for_shared_weights

from utils.plots import generate_images

def save_checkpoint(model, optimizer, scheduler, epoch, save_path):
    """
    Saves checkpoint for the given epoch into a sub-directory named epoch_{epoch}.
    Also updates the 'latest' file to record the most recently saved epoch.
    """
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=comm.get_group("dp"))
    assert counts[0].item() == torch.distributed.get_world_size(group=comm.get_group("dp"))

    rank = torch.distributed.get_rank()

    epoch_dir = os.path.join(save_path, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all(),
    }

    ckpt_path = os.path.join(epoch_dir, f"rank{rank}.pt")
    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint for epoch {epoch} saved for rank {rank} -> {ckpt_path}")

    latest_file_path = os.path.join(save_path, "latest")
    with open(latest_file_path, "w") as f:
        f.write(str(epoch))


def load_checkpoint(model, optimizer, scheduler, load_path):
    """
    Loads the most recent checkpoint according to the epoch stored in 'latest'.
    Assumes sub-directories are named as epoch_{epoch}, and each rank file is rank{rank}.pt.
    """
    rank = 0 #torch.distributed.get_rank()

    latest_file_path = os.path.join(load_path, "latest")
    if not os.path.exists(latest_file_path):
        raise FileNotFoundError(
            f"No 'latest' file found in {load_path}! Make sure checkpoints have been saved."
        )

    with open(latest_file_path, "r") as f:
        latest_epoch = int(f.read().strip())

    checkpoint_path = os.path.join(load_path, f"epoch_{latest_epoch}", f"rank{rank}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")

    checkpoint = torch.load(checkpoint_path, map_location='cuda')
     
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint 
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    
    if optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    cpu_rng_state = checkpoint['rng_state'].to(dtype=torch.uint8, device='cpu')
    torch.set_rng_state(cpu_rng_state)

    cuda_rng_states = [s.to(dtype=torch.uint8, device='cpu') for s in checkpoint['cuda_rng_state']]
    torch.cuda.set_rng_state_all(cuda_rng_states)

    print(f"Checkpoint loaded for rank {rank} from {checkpoint_path}")
    return checkpoint['epoch']


def inference(params, args, local_rank, world_rank, world_size):
    # set device and benchmark mode
    #torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda:%d" % local_rank)

    # get data loader
    logging.info("rank %d, begin data loader init" % world_rank)
    train_data_loader, train_dataset, train_sampler = get_data_loader_distributed(
        params, params.train_data_path, params.distributed, train=True
    )
    val_data_loader, valid_dataset = get_data_loader_distributed(
        params, params.valid_data_path, params.distributed, train=False
    )
    logging.info("rank %d, data loader initialized" % (world_rank))

    # create model
    model = vit.ViT(params).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if world_rank == 0:
        logging.info(f"Total parameters: {num_params}")
        logging.info(f"Trainable parameters: {num_trainable_params}")

    if params.enable_jit:
        model = torch.compile(model)

    if params.amp_dtype == torch.float16:
        scaler = GradScaler()

    # weight initialization needs to be synced across shared weights
    if comm.get_size("tp-cp") > 1:
        init_params_for_shared_weights(model)

    if params.distributed and not args.noddp:
        model = init_ddp_model_and_reduction_hooks(model, device_ids=[local_rank],
                                                   output_device=[local_rank],
                                                   bucket_cap_mb=args.bucket_cap_mb)

    if params.enable_fused:
        optimizer = optim.Adam(
            model.parameters(), lr=params.lr, fused=True, betas=(0.9, 0.95)
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.95))

    if world_rank == 0:
        logging.info(model)

    iters = 0
    startEpoch = 0

    if params.lr_schedule == "cosine":
        if params.warmup > 0:
            lr_scale = lambda x: min(
                (x + 1) / params.warmup,
                0.5 * (1 + np.cos(np.pi * x / params.num_iters)),
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=params.num_iters
            )
    else:
        scheduler = None

    # select loss function
    if params.enable_jit:
        loss_func = l2_loss_opt
    else:
        loss_func = l2_loss


    last_epoch = load_checkpoint(model, None, None, params['save_path'])
    print(f"checkpoint loaded: start from epoch {last_epoch+1}")

    # Log initial loss on train and validation to tensorboard
    with torch.no_grad():
        #inp, tar = map(lambda x: x.to(device), next(iter(train_data_loader)))
        #gen = model(inp)
       
        #torch.save({
        #    'inp': inp.cpu(),  # move back to CPU if needed
        #    'tar': tar.cpu(),
        #    'gen': gen.cpu()
        #}, 'tensors.pt')
        datapoint = torch.load('tensors.pt')
        inp = datapoint['inp'].to("cuda")
        tar = datapoint['tar'].to("cuda")
        gen = model(inp)

        loss = loss_func(gen, tar)
        fig = generate_images([inp, tar, gen])
        print(f"loss {loss}") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_num",
        default="00",
        type=str,
        help="tag for indexing the current experiment",
    )
    parser.add_argument(
        "--yaml_config",
        default="./config/ViT.yaml",
        type=str,
        help="path to yaml file containing training configs",
    )
    parser.add_argument(
        "--config", default="base", type=str, help="name of desired config in yaml file"
    )
    parser.add_argument(
        "--amp_mode",
        default="none",
        type=str,
        choices=["none", "fp16", "bf16"],
        help="select automatic mixed precision mode",
    )
    parser.add_argument(
        "--enable_fused", action="store_true", help="enable fused Adam optimizer"
    )
    parser.add_argument(
        "--enable_jit", action="store_true", help="enable JIT compilation"
    )
    parser.add_argument(
        "--local_batch_size",
        default=None,
        type=int,
        help="local batchsize (manually override global_batch_size config setting)",
    )
    parser.add_argument(
        "--num_iters", default=None, type=int, help="number of iters to run"
    )
    parser.add_argument(
        "--num_data_workers",
        default=None,
        type=int,
        help="number of data workers for data loader",
    )
    parser.add_argument(
        "--data_loader_config",
        default=None,
        type=str,
        choices=["pytorch", "dali"],
        help="dataloader configuration. choices: 'pytorch', 'dali'",
    )
    parser.add_argument(
        "--bucket_cap_mb", default=25, type=int, help="max message bucket size in mb"
    )
    parser.add_argument(
        "--disable_broadcast_buffers",
        action="store_true",
        help="disable syncing broadcasting buffers",
    )
    parser.add_argument(
        "--noddp", action="store_true", help="disable DDP communication"
    )

    # model parallelism arguments
    parser.add_argument(
        "--tensor_parallel",
        default=1,
        type=int,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--context_parallel",
        default=1,
        type=int,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--parallel_order",
        default="tp-cp-dp",
        type=str,
        help="Order of ranks for parallelism",
    )

    args = parser.parse_args()

    run_num = args.run_num

    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # Update config with modified args
    # set up amp
    if args.amp_mode != "none":
        params.update({"amp_mode": args.amp_mode})
    amp_dtype = torch.float32
    if params.amp_mode == "fp16":
        amp_dtype = torch.float16
    elif params.amp_mode == "bf16":
        amp_dtype = torch.bfloat16
    params.update(
        {
            "amp_enabled": amp_dtype is not torch.float32,
            "amp_dtype": amp_dtype,
            "enable_fused": args.enable_fused,
            "enable_jit": args.enable_jit,
        }
    )

    if args.data_loader_config:
        params.update({"data_loader_config": args.data_loader_config})

    if args.num_iters:
        params.update({"num_iters": args.num_iters})

    if args.num_data_workers:
        params.update({"num_data_workers": args.num_data_workers})

    params.distributed = False

    # setup model parallel sizes
    params["tp"] = args.tensor_parallel
    params["cp"] = args.context_parallel
    params["order"] = args.parallel_order
    # initialize comm
    #comm.init(params, verbose=True)

    # get info from comm
    world_size = comm.get_world_size()
    world_rank = comm.get_world_rank()
    local_rank = comm.get_local_rank()
    params.distributed = world_size > 1

    assert (
        params["global_batch_size"] % comm.get_size("dp") == 0
    ), f"Error, cannot evenly distribute {params['global_batch_size']} across {comm.get_size('dp')} GPU."

    if args.local_batch_size:
        # Manually override batch size
        params.local_batch_size = args.local_batch_size
        params.update(
            {"global_batch_size": comm.get_size("dp") * args.local_batch_size}
        )
    else:
        # Compute local batch size based on number of ranks
        params.local_batch_size = int(
            params["global_batch_size"] // comm.get_size("dp")
        )

    # for data loader, set the actual number of data shards and id
    params.data_num_shards = comm.get_size("dp")
    params.data_shard_id = comm.get_rank("dp")

    # Set up directory
    baseDir = params.expdir
    expDir = os.path.join(
        baseDir, args.config + "/%dMP/" % (comm.get_size("tp-cp")) + str(run_num) + "/"
    )
    if world_rank == 0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
        logging_utils.log_to_file(
            logger_name=None, log_filename=os.path.join(expDir, "out.log")
        )
        params.log()
        args.tboard_writer = SummaryWriter(log_dir=os.path.join(expDir, "logs/"))

    params.experiment_dir = os.path.abspath(expDir)

    inference(params, args, local_rank, world_rank, world_size)

    if params.distributed:
        torch.distributed.barrier()
    logging.info("DONE ---- rank %d" % world_rank)
