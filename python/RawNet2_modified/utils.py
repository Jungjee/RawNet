import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def keras_lr_decay(step, decay=0.0001):
    return 1.0 / (1.0 + decay * step)


def init_weights(m):
    print(m)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.0001)
    elif isinstance(m, nn.BatchNorm1d):
        pass
    else:
        if hasattr(m, "weight"):
            torch.nn.init.kaiming_normal_(m.weight, a=0.01)
        else:
            print("no weight", m)


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_val_utts(l_val_trial):
    l_utt = []
    for line in l_val_trial:
        _, utt_a, utt_b = line.strip().split(" ")
        if utt_a not in l_utt:
            l_utt.append(utt_a)
        if utt_b not in l_utt:
            l_utt.append(utt_b)
    return l_utt


def get_utt_list(src_dir):
    """
    Designed for VoxCeleb
    """
    l_utt = []
    for path, dirs, files in os.walk(src_dir):
        base = "/".join(path.split("/")[-2:]) + "/"
        for file in files:
            if file[-3:] != "wav":
                continue
            l_utt.append(base + file)
    return l_utt


def get_label_dic_Voxceleb(l_utt):
    d_label = {}
    idx_counter = 0
    for utt in l_utt:
        spk = utt.split("/")[0]
        if spk not in d_label:
            d_label[spk] = idx_counter
            idx_counter += 1
    return d_label


def make_validation_trial(l_utt, nb_trial, dir_val_trial):
    f_val_trial = open(dir_val_trial, "w")
    # trg trial: 1, non-trg: 0
    nb_trg_trl = int(nb_trial / 2)
    d_spk_utt = {}
    # make a dictionary that has keys as speakers
    for utt in l_utt:
        spk = utt.split("/")[0]
        if spk not in d_spk_utt:
            d_spk_utt[spk] = []
        d_spk_utt[spk].append(utt)

    l_spk = list(d_spk_utt.keys())
    # compose trg trials
    selected_spks = np.random.choice(l_spk, size=nb_trg_trl, replace=True)
    for spk in selected_spks:
        l_cur = d_spk_utt[spk]
        utt_a, utt_b = np.random.choice(l_cur, size=2, replace=False)
        f_val_trial.write("1 %s %s\n" % (utt_a, utt_b))
    # compose non-trg trials
    for i in range(nb_trg_trl):
        spks_cur = np.random.choice(l_spk, size=2, replace=False)
        utt_a = np.random.choice(d_spk_utt[spks_cur[0]], size=1)[0]
        utt_b = np.random.choice(d_spk_utt[spks_cur[1]], size=1)[0]
        f_val_trial.write("0 %s %s\n" % (utt_a, utt_b))
    f_val_trial.close()
    return


def draw_histogram(l_score, l_target, save_dir):
    l_pos = []
    l_neg = []
    for s, t in zip(l_score, l_target):
        if int(t) == 1:
            l_pos.append(s)
        else:
            l_neg.append(s)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.ylim([0, 500])
    patches = ax.hist(l_pos, label="target", bins=200, color="blue", alpha=0.5)
    patches = ax.hist(l_neg, label="nontarget", bins=200, color="red", alpha=0.5)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    plt.ylabel("#trial")
    plt.legend(loc="best")
    plt.savefig(save_dir + "histogram.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    return


def write_scores(save_dir, epoch, gpu, y_label, y_score):
    f_res = open(save_dir + "results/eval_epoch{}_{}.txt".format(epoch, gpu), "w")
    # f_res.write('label\tscore\t(1:target)\n')
    for l, s in zip(y_label, y_score):
        f_res.write("{}\t{}\n".format(l, s))
    f_res.close()
    return


def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            fn, ext = os.path.splitext(file)
            if ext != ".py":
                continue

            ziph.write(os.path.join(root, file))


def gather_score_files_ddp(save_dir, epoch):
    cmd = "cat {}results/eval_epoch{}_*.txt > {}results/eval_epoch{}.txt".format(
        save_dir, epoch, save_dir, epoch
    )
    os.system(cmd)
    return


def read_scores(save_dir, epoch):
    y_label, y_score = [], []
    with open(save_dir + "results/eval_epoch{}.txt".format(epoch), "r") as f:
        lines = f.readlines()
    # for l in lines[1:]:
    for l in lines:
        lab, sco = l.strip().split("\t")
        y_label.append(int(lab))
        y_score.append(float(sco))
    return y_label, y_score


class EpochEndCallback:
    """
    ver. 201111.
    Terminates experiment when the Eval EER does not improve for (patience) epochs.

    Updates.
    - add scaler to checkpoint
    """

    def __init__(self, patience, save_dir, experiment, early_stop = True, verbose=1, delta=0):
        #####
        # given args
        #####
        self.patience = patience
        self.save_dir = save_dir
        self.experiment = experiment
        self.early_stop = early_stop
        self.verbose = verbose
        self.delta = delta

        #####
        # made
        #####
        self.best_eer = None
        self.counter = 0
        self.f_eer = open(save_dir + "eers.txt", "a", buffering=1)

    def at_epoch_end(self, eer, epoch, checkpoint):
        self.f_eer.write("epoch:%d\teval_eer:%.4f\n" % (epoch, eer))
        self.experiment.log_metric("eval_eer", eer, step=epoch)
        eer = float(eer)

        if self.best_eer == None:
            self.best_eer = eer
            print("New best EER: %f" % eer)
            self.experiment.log_metric("best_eval_eer", eer, step=epoch)
            self.save_checkpoint(eer, epoch, checkpoint)
        elif eer > self.best_eer:  # when worse
            self.counter += 1
            if self.counter == self.patience and self.early_stop:
                self.at_exp_end()
                print("Early stopping at epoch{}".format(epoch))
                exit()
        else:
            print("New best EER: %f" % eer)
            self.best_eer = eer
            self.experiment.log_metric("best_eval_eer", eer, step=epoch)
            self.save_checkpoint(eer, epoch, checkpoint)
            self.counter = 0

    def save_checkpoint(self, eer, epoch, checkpoint):
        torch.save(checkpoint, self.save_dir + "models/%d_%.4f.pt" % (epoch, eer))

    def at_exp_end(self):
        self.f_eer.close()


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
