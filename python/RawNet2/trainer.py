import torch
import numpy as np

from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from utils import cos_sim

def train_model(model, db_gen, optimizer, epoch, args, device, lr_scheduler, criterion):
    model.train()
    with tqdm(total = len(db_gen), ncols = 70) as pbar:
        for m_batch, m_label in db_gen:
            
            m_batch, m_label = m_batch.to(device), m_label.to(device)

            output = model(m_batch, m_label)
            cce_loss = criterion['cce'](output, m_label)
            loss = cce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description('epoch: %d, cce:%.3f'%(epoch, cce_loss))
            pbar.update(1)
            if args.do_lr_decay:
                if args.lr_decay == 'keras': lr_scheduler.step()

def time_augmented_evaluate_model(mode, model, db_gen, l_utt, save_dir, epoch, l_trial, args, device):
    if mode not in ['val','eval']: raise ValueError('mode should be either "val" or "eval"')
    model.eval()
    with torch.set_grad_enabled(False):
        #1st, extract speaker embeddings.
        l_embeddings = []
        with tqdm(total = len(db_gen), ncols = 70) as pbar:
            for m_batch in db_gen:
                l_code = []
                for batch in m_batch:
                    batch = batch.to(device)
                    code = model(x = batch, is_test=True)
                    l_code.extend(code.cpu().numpy())
                l_embeddings.append(np.mean(l_code, axis=0))
                pbar.update(1)
        d_embeddings = {}
        if not len(l_utt) == len(l_embeddings):
            print(len(l_utt), len(l_embeddings))
            exit()
        for k, v in zip(l_utt, l_embeddings):
            d_embeddings[k] = v
            
        #2nd, calculate EER
        y_score = [] # score for each sample
        y = [] # label for each sample 
        f_res = open(save_dir + 'results/{}_epoch{}.txt'.format(mode, epoch), 'w')
        
        for line in l_trial:
            trg, utt_a, utt_b = line.strip().split(' ')
            y.append(int(trg))
            y_score.append(cos_sim(d_embeddings[utt_a], d_embeddings[utt_b]))
            f_res.write('{score} {target}\n'.format(score=y_score[-1],target=y[-1]))
        f_res.close()
        fpr, tpr, _ = roc_curve(y, y_score, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer