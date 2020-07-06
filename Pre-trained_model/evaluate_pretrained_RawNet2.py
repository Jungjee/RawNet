from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import argparse
import numpy as np
import pickle as pk

import torch
import torch.nn as nn
from torch.utils import data

from model_RawNet2_original_code import RawNet

def cos_sim(a,b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

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
    parser.add_argument('-embd_dir', type=str, default = '../spk_embd/')
    parser.add_argument('-bs', type=int, default = 100) #batch size
    parser.add_argument('-nb_worker', type=int, default = 4) #number of workers for PyTorch DataLoader
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

    with open('../trials/list_test_all_cleaned.txt' , 'r') as f:
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

    with open('../trials/list_test_hard_cleaned.txt' , 'r') as f:
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