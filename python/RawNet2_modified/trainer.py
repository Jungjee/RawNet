import os
import gc
import numpy as np
import pickle as pk

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.cuda.amp as amp

from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from utils import cos_sim


def train_model(
    model,
    db_gen,
    optimizer,
    epoch,
    args,
    gpu,
    lr_scheduler,
    scaler,
    experiment,
):
    pid = os.getpid()
    model.train()
    idx_ct_start = int(len(db_gen) * epoch)
    loss = 0.0
    with tqdm(total=len(db_gen), ncols=70) as pbar:
        for idx, (m_batch, m_label) in enumerate(db_gen):
            optimizer.zero_grad()

            #####
            # forward data
            #####
            m_batch = m_batch.cuda(
                gpu, non_blocking=True
            )  # print(m_batch.size())#(60,1,69049)
            m_label = m_label.cuda(gpu, non_blocking=True)
            with amp.autocast():
                _loss = model(m_batch, label=m_label)
            loss += _loss.item()

            #####
            # https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
            #####
            scaler.scale(_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #####
            # log result
            #####
            if idx % args.nb_iter_per_log == 0:
                if idx != 0:
                    loss /= args.nb_iter_per_log

                desc_txt = "%s,GPU%s,pid:%s,epoch:%d,loss:%.4f" % (
                    args.name,
                    gpu,
                    pid,
                    epoch,
                    loss,
                )
                pbar.set_description(desc_txt)
                if idx == 0:
                    pbar.update(1)
                else:
                    pbar.update(args.nb_iter_per_log)

                for p_group in optimizer.param_groups:
                    lr = p_group["lr"]
                    break
                if gpu == 0:
                    experiment.log_metric(
                        "loss", loss, step=idx_ct_start + idx)
                    experiment.log_metric("lr", lr, step=idx_ct_start + idx)
                loss = 0.0

            #####
            # learning rate decay
            #####
            if args.do_lr_decay:
                if args.lr_decay in ["keras", "cosine"]:
                    lr_scheduler.step()
    return


def extract_embd(model, db_gen, save_dir, gpu):
    model.eval()
    with torch.set_grad_enabled(False):
        l_embeddings = []
        l_keys = []
        with tqdm(total=len(db_gen), ncols=70) as pbar:
            for idx, (m_batch, keys) in enumerate(db_gen):
                m_batch = m_batch.cuda(gpu, non_blocking=True)
                org_size = list(m_batch.size())[:2] + [-1]
                l_code = model(x=m_batch.flatten(0, 1), is_test=True).view(
                    org_size
                    # )  # (bs, 10,1024), tensor on gpu
                ).detach().cpu() 
                l_embeddings.append(l_code)
                l_keys.extend(keys)
                if idx % 20 == 0:
                    if idx == 0:
                        pbar.update(1)
                    else:
                        pbar.update(20)
    pk.dump(l_keys, open(save_dir + "embds/keys_{}.pk".format(gpu), "wb"))
    torch.save(
        torch.cat(l_embeddings, dim=0), save_dir +
        "embds/embd_{}.pt".format(gpu)
    )
    return


def get_score(l_trials, ngpu, save_dir):
    l_keys = []
    l_embd = []
    for i in range(ngpu):
        l_keys.extend(
            pk.load(open(save_dir + "embds/keys_{}.pk".format(i), "rb")))
        l_embd.append(
            torch.load(save_dir + "embds/embd_{}.pt".format(i),
                       map_location="cuda:0")
        )
    l_embd = torch.cat(l_embd, dim=0)
    try:
        assert len(l_keys) == l_embd.size(0)
    except:
        raise ValueError(
            "Number of eval set utt not correct, got:{}{}".format(
                len(l_keys), l_embd.size()
            )
        )

    # compose dictionary because of multiple time used embeddings
    d_embd = {}
    for k, v in zip(l_keys, l_embd):
        d_embd[k] = v

    y_score = []  # score for each sample
    y_label = []  # label for each sample
    for line in l_trials:
        trg, utt_a, utt_b = line.strip().split(" ")
        utt_a, utt_b = d_embd[utt_a], d_embd[utt_b]
        utt_a = F.normalize(utt_a, p=2, dim=1)
        utt_b = F.normalize(utt_b, p=2, dim=1)
        y_label.append(int(trg))
        score = (
            F.pairwise_distance(
                utt_a.unsqueeze(-1), utt_b.unsqueeze(-1).transpose(0, 2)
            )
            .detach()
            .cpu()
            .numpy()
        )
        y_score.append(-1 * np.mean(score))
    return y_score, y_label
