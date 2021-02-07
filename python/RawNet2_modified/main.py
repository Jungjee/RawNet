import argparse
import json
import os
import warnings
import zipfile
from collections import OrderedDict
from importlib import import_module
from parser import get_args

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from tqdm import tqdm

from dataloader import get_loader
from optimizer import get_optimizer
from trainer import *
from utils import *

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    # parse arguments
    args = get_args()

    # make experiment reproducible
    # Possible slower training
    if args.reproducible:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # DDP setting
    # Single node, single&multi GPU environment is covered
    args.ngpus_per_node = torch.cuda.device_count()
    args.bs = args.bs // args.ngpus_per_node
    args.world_size = args.ngpus_per_node
    args.rank = 0
    mp.spawn(
        main_worker,
        nprocs=args.ngpus_per_node,
        args=(args.ngpus_per_node, args)
    )


def main_worker(gpu, ngpus_per_node, args):
    # DDP setting
    cuda = torch.cuda.is_available()
    if not cuda:
        raise NotImplementedError(
            "This script is written for single-node single&multi-GPUs env only."
        )
    print("Using GPU:{} for training".format(gpu))
    torch.cuda.set_device(gpu)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.ddp_port
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(
        backend="nccl", world_size=args.world_size, rank=args.rank
    )

    # load dataset
    with open(args.DB + "veri_test.txt", "r") as f:
        l_eval_trial = f.readlines()
    trnset_gen, trnset_sampler, evlset_gen, d_label = get_loader(args)
    args.model["nb_spk"] = len(d_label)

    save_dir = args.save_dir + args.name + "/"
    if gpu == 0:
        # set save directory, and log code
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(save_dir + "results/"):
            os.makedirs(save_dir + "results/")
        if not os.path.exists(save_dir + "models/"):
            os.makedirs(save_dir + "models/")
        if not os.path.exists(save_dir + "embds/"):
            os.makedirs(save_dir + "embds/")
        with zipfile.ZipFile(
            save_dir + "codes.zip", "w", zipfile.ZIP_DEFLATED
        ) as f_zip:
            zipdir(".", f_zip)

        # log experiment parameters to local and comet_ml server
        f_params = open(save_dir + "f_params.txt", "w")
        for k, v in sorted(vars(args).items()):
            if args.verbose > 0:
                print(k, v)
            f_params.write("{}:\t{}\n".format(k, v))
        f_params.close()

    # define model
    module = import_module("models.{}".format(args.module_name))
    _model = getattr(module, args.model_name)
    args.model["device"] = gpu
    model = _model(**args.model).cuda(gpu)
    # FIXME : syncBN leaks CPU memory (empirical)
    # -->> currently disabled
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[gpu], find_unused_parameters=True
    )
    if args.load_model:
        dev = torch.cuda.current_device()
        checkpoint = torch.load(
            args.load_model_dir, map_location=lambda storage, loc: storage.cuda(
                dev)
        )
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        torch.cuda.empty_cache()

    if args.verbose > 0 and gpu == 0:
        print("nb_params: {}".format(nb_params))

    # set optimizer
    args.nb_iter_per_epoch = (
        int(len(trnset_gen) - (len(trnset_gen) % 100))
        if args.lr_decay == "triangular2"
        else len(trnset_gen)
    )
    optimizer, scaler, lr_scheduler = get_optimizer(args, model)

    # Train
    callback = EpochEndCallback(
        patience=args.patience,
        save_dir=save_dir,
        verbose=args.verbose,
        early_stop=args.early_stop,
    )
    for epoch in tqdm(range(args.epoch)):
        trnset_sampler.set_epoch(epoch)
        train_model(
            model=model,
            db_gen=trnset_gen,
            args=args,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            gpu=gpu,
            scaler=scaler,
            epoch=epoch,
        )

        # extract embeddings multi-GPU and save them respectively
        extract_embd(model=model, db_gen=evlset_gen,
                     save_dir=save_dir, gpu=gpu)
        dist.barrier()

        # do evaluation on the 0th GPU
        if gpu == 0:
            y_score, y_label = get_score(
                l_trials=l_eval_trial, ngpu=args.ngpus_per_node, save_dir=save_dir
            )
            write_scores(
                save_dir=save_dir, epoch=epoch, gpu=0, y_label=y_label, y_score=y_score
            )
            fpr, tpr, _ = roc_curve(y_label, y_score, pos_label=1)
            eval_eer = brentq(lambda x: 1.0 - x -
                              interp1d(fpr, tpr)(x), 0.0, 1.0)

            draw_histogram(l_score=y_score, l_target=y_label,
                           save_dir=save_dir)

            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            }
            callback.at_epoch_end(
                eer=eval_eer, epoch=epoch, checkpoint=checkpoint)
        dist.barrier()
    callback.at_exp_end()


if __name__ == "__main__":
    main()
