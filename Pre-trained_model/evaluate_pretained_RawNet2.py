from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from torch.optim.lr_scheduler import _LRScheduler
from math import cos, pi

import os
import argparse
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data

import soundfile as sf
import pickle as pk

from model_RawNet2_original_code import RawNet

def cos_sim(a,b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Dataset_embd(data.Dataset):
    def __init__(self, d_embd, trials='', nb_trials = 0, mode = 'trn'):
        self.d_embd = d_embd
        if mode == 'trn':
            raise ValueError('Removed')
        else:
            self.trials = []
            for l in trials:
                lab, md, tst = l.strip().split(' ')
                self.trials.append([md.split('.')[0], tst.split(',')[0], int(lab)])
        self.mode = mode

    def __len__(self):
        return self.nb_trials if self.mode == 'trn' else len(self.trials)
    
    def __getitem__(self, index):
        if self.mode == 'trn':
            raise ValueError('Removed')

        else:    #val, eval
            trial = self.trials[index]
            return self.d_embd[trial[0].split('.')[0]], self.d_embd[trial[1].split('.')[0]], trial[2]

    def _get_d_meta(self):
        self.d_meta = {}
        for k in self.d_embd.keys():
            spk = k.split('/')[0]
            if spk not in self.d_meta: self.d_meta[spk] = []
            self.d_meta[spk].append(k)
        return

def evaluate_init_model(db_gen):
    y = []
    y_scores = []
    with tqdm(total = len(db_gen), ncols = 70) as pbar:
        for m_batch, m_batch2, m_label in db_gen:
            for l, b, b2 in zip(m_label, m_batch, m_batch2):
                y.append(l)
                y_scores.append(cos_sim(b, b2))
            pbar.update(1)
    fpr, tpr, _ = roc_curve(y, y_scores, pos_label = 1)
    eer = brentq(lambda x:1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def main():
    parser = argparse.ArgumentParser()
    #dir
    parser.add_argument('-name', type = str, required = True)
    parser.add_argument('-save_dir', type = str, default = '/exp/DNNs/')
    parser.add_argument('-embd_dir', type=str, default = '../spk_embd/')

    #hyper-params
    parser.add_argument('-bs', type = int, default = 120)
    parser.add_argument('-lr', type = float, default = 0.001)
    parser.add_argument('-wd', type = float, default = 0.0000)
    parser.add_argument('-epoch', type = int, default = 80)
    parser.add_argument('-optimizer', type = str, default = 'Adam') #'Adam'
    parser.add_argument('-nb_worker', type = int, default = 2)
    parser.add_argument('-seed', type = int, default = 1234) 
    parser.add_argument('-opt_mom', type = float, default = 0.9) 
    parser.add_argument('-am_margin', type = float, default = 0.35) 
    parser.add_argument('-lr_decay', type = str, default = 'keras') 
    parser.add_argument('-iter_per_epoch', type = int, default = 1000) 
    parser.add_argument('-model', type = json.loads, default = '{"spk_embd_dim":1024,"l_nodes":[512,512,512,2],"input_drop":0.2}')

    #flag
    parser.add_argument('-make_trial', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-amsgrad', type = str2bool, nargs='?', const=True,  default = True)
    parser.add_argument('-nesterov', type = str2bool, nargs='?', const=True,  default = False)
    parser.add_argument('-comet_disable', type = str2bool, nargs='?', const=True,  default = False)
    parser.add_argument('-save_best_only', type = str2bool, nargs='?', const=True,  default = True)
    parser.add_argument('-do_lr_decay', type = str2bool, nargs='?', const=True,  default = True)
    parser.add_argument('-mg', type = str2bool, nargs='?', const=True,  default = False)
    
    args = parser.parse_args()

    with open(args.embd_dir + 'TTA_vox1_dev.pk', 'rb') as f:
        d_embd_vox1_dev = pk.load(f)

    with open(args.embd_dir + 'TTA_vox1_eval.pk', 'rb') as f:
        d_embd_vox1_eval = pk.load(f)

    d_embd_vox1 = {**d_embd_vox1_dev, **d_embd_vox1_eval}

    f_eer = open('./eers.txt', 'a', buffering = 1)
    with open('../trials/vox_original.txt' , 'r') as f:
        l_eval_trial = f.readlines()
    evalset_sv = Dataset_embd(
        d_embd = d_embd_vox1_eval,
        trials = l_eval_trial,
        mode = 'eval')
    evalset_sv_gen = data.DataLoader(evalset_sv,
        batch_size = args.bs, 
        shuffle = False,
        drop_last = False,
        num_workers = args.nb_worker)
    eval_eer = evaluate_init_model(
        db_gen = evalset_sv_gen 
        )
    text = 'Vox original evaluation EER: {}'.format(eval_eer)
    print(text)
    f_eer.write(text+'\n')

    with open('/DB/VoxCeleb2/list_test_all_cleaned.txt' , 'r') as f:
        l_eval_trial = f.readlines()
    evalset_sv = Dataset_embd(
        d_embd = d_embd_vox1,
        trials = l_eval_trial,
        mode = 'eval')
    evalset_sv_gen = data.DataLoader(evalset_sv,
        batch_size = args.bs, 
        shuffle = False,
        drop_last = False,
        num_workers = args.nb_worker)
    eval_eer = evaluate_init_model(
        db_gen = evalset_sv_gen 
        )
    text = 'Vox1_E_cleaned evaluation EER: {}'.format(eval_eer)
    print(text)
    f_eer.write(text+'\n')

    with open('/DB/VoxCeleb2/list_test_hard_cleaned.txt' , 'r') as f:
        l_eval_trial = f.readlines()
    evalset_sv = Dataset_embd(
        d_embd = d_embd_vox1,
        trials = l_eval_trial,
        mode = 'eval')
    evalset_sv_gen = data.DataLoader(evalset_sv,
        batch_size = args.bs, 
        shuffle = False,
        drop_last = False,
        num_workers = args.nb_worker)
    eval_eer = evaluate_init_model(
        db_gen = evalset_sv_gen 
        )
    text = 'Vox1_H_cleaned evaluation EER: {}'.format(eval_eer)
    print(text)
    f_eer.write(text+'\n')
    f_eer.close()

if __name__ == '__main__':
    main()