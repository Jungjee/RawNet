import os
import numpy as np
import torch
import torch.nn as nn

def keras_lr_decay(step, decay = 0.0001):
    return 1./(1. + decay * step)

def init_weights(m):
    print(m)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.0001)
    elif isinstance(m, nn.BatchNorm1d):
        pass
    else:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_normal_(m.weight, a=0.01)
        else:
            print('no weight',m)

def cos_sim(a,b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_val_utts(l_val_trial):
    l_utt = []
    for line in l_val_trial:
        _, utt_a, utt_b = line.strip().split(' ')
        if utt_a not in l_utt: l_utt.append(utt_a)
        if utt_b not in l_utt: l_utt.append(utt_b)
    return l_utt

def get_utt_list(src_dir):
    '''
    Designed for VoxCeleb
    '''
    l_utt = []
    for path, dirs, files in os.walk(src_dir):
        base = '/'.join(path.split('/')[-2:])+'/'
        for file in files:
            if file[-3:] != 'wav':
                continue
            l_utt.append(base+file)
    return l_utt

def get_label_dic_Voxceleb(l_utt):
    d_label = {}
    idx_counter = 0
    for utt in l_utt:
        spk = utt.split('/')[0]
        if spk not in d_label:
            d_label[spk] = idx_counter
            idx_counter += 1
    return d_label

def make_validation_trial(l_utt, nb_trial, dir_val_trial):
    f_val_trial = open(dir_val_trial, 'w')
    #trg trial: 1, non-trg: 0
    nb_trg_trl = int(nb_trial / 2)
    d_spk_utt = {}
    #make a dictionary that has keys as speakers
    for utt in l_utt:
        spk = utt.split('/')[0]
        if spk not in d_spk_utt: d_spk_utt[spk] = []
        d_spk_utt[spk].append(utt)
        
    l_spk = list(d_spk_utt.keys())
    #compose trg trials
    selected_spks = np.random.choice(l_spk, size=nb_trg_trl, replace=True)
    for spk in selected_spks:
        l_cur = d_spk_utt[spk]
        utt_a, utt_b = np.random.choice(l_cur, size=2, replace=False)
        f_val_trial.write('1 %s %s\n'%(utt_a, utt_b))
    #compose non-trg trials
    for i in range(nb_trg_trl):
        spks_cur = np.random.choice(l_spk, size=2, replace = False)
        utt_a = np.random.choice(d_spk_utt[spks_cur[0]], size=1)[0]
        utt_b = np.random.choice(d_spk_utt[spks_cur[1]], size=1)[0]
        f_val_trial.write('0 %s %s\n'%(utt_a, utt_b))
    f_val_trial.close()
    return