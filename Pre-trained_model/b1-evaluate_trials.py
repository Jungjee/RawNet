from comet_ml import Experiment
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

#from model_bvec import Bvec_DNN as Model
from model_catmul import Bvec_DNN as Model
import soundfile as sf
import pickle as pk

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
			self._get_d_meta()
			self.l_spk = sorted(list(self.d_meta.keys())) 
			print('l_spk:',len(self.l_spk))
			self.nb_trials = nb_trials	#for trn
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
			b = bool(np.random.randint(2))
			if b: #compose trg trial, 0
				sel_spk = list(np.random.choice(self.l_spk, 1))[0]
				l_trial = np.random.choice(self.d_meta[sel_spk], 2, replace=False)

				return self.d_embd[l_trial[0].split('.')[0]], self.d_embd[l_trial[1].split('.')[0]], torch.tensor(0)
			
			else: #compose nontrg trial, 1
				sel_spks = list(np.random.choice(self.l_spk, 2, replace=False))
				l_trial = [np.random.choice(self.d_meta[sel_spks[0]], 1)[0].split('.')[0],
						   np.random.choice(self.d_meta[sel_spks[1]], 1)[0].split('.')[0]]
				
				return self.d_embd[l_trial[0]], self.d_embd[l_trial[1]], torch.tensor(1)

		else:	#val, eval
			trial = self.trials[index]
			return self.d_embd[trial[0].split('.')[0]], self.d_embd[trial[1].split('.')[0]], trial[2]

	def _get_d_meta(self):
		self.d_meta = {}
		for k in self.d_embd.keys():
			spk = k.split('/')[0]
			if spk not in self.d_meta: self.d_meta[spk] = []
			self.d_meta[spk].append(k)
		return

class CosineLR(_LRScheduler):
	"""cosine annealing.
	"""
	def __init__(self, optimizer, step_size_min=1e-5, t0=100, tmult=2, curr_epoch=-1, last_epoch=-1):
		self.step_size_min = step_size_min
		self.t0 = t0
		self.tmult = tmult
		self.epochs_since_restart = curr_epoch
		super(CosineLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		self.epochs_since_restart += 1

		if self.epochs_since_restart > self.t0:
			self.t0 *= self.tmult
			self.epochs_since_restart = 0

		lrs = [self.step_size_min + (
				0.5 * (base_lr - self.step_size_min) * (1 + cos(self.epochs_since_restart * pi / self.t0)))
			   for base_lr in self.base_lrs]

		return lrs

def get_eer(Pfa,Pmiss):
	"""
	Calculate EER
	"""
	idxeer=np.argmin(np.abs(Pfa-Pmiss))
	return 0.5*(Pfa[idxeer]+Pmiss[idxeer])

def keras_lr_decay(step, decay = 0.00005):
	return 1./(1.+decay*step)

def train_model(model, db_gen, optimizer, epoch, args, experiment, device, lr_scheduler, criterion):
	model.train()
	with tqdm(total = len(db_gen), ncols = 70) as pbar:
		for idx_ct, (m_batch, m_batch2, m_label) in enumerate(db_gen):
					
			m_batch, m_batch2, m_label = m_batch.to(device), m_batch2.to(device), m_label.to(device)

			o = model(m_batch, m_batch2) #output
			cce_loss = criterion['cce'](o, m_label)
			loss = cce_loss

			if idx_ct % 1000 == 0: 
				experiment.log_metric('cce_loss', cce_loss.detach().cpu().numpy())
				for p_group in optimizer.param_groups:
					lr = p_group['lr']
					break
				experiment.log_metric('lr', lr)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			pbar.set_description('epoch%d,lr:%.6f,cce:%.3f'%(epoch, lr, cce_loss))
			pbar.update(1)

			if args.do_lr_decay:
				if args.lr_decay == 'keras':
					lr_scheduler.step()
				elif args.lr_decay == 'cosine':
					lr_scheduler.step()
					

def evaluate_init_model(db_gen):
	y = []
	y_scores = []
	with tqdm(total = len(db_gen), ncols = 70) as pbar:
		for m_batch, m_batch2, m_label in db_gen:
			for l, b, b2 in zip(m_label, m_batch, m_batch2):
				y.append(l)
				y_scores.append(cos_sim(b, b2))
			pbar.update(1)
	fpr, tpr, thresholds = roc_curve(y, y_scores, pos_label = 1)
	eer = brentq(lambda x:1. - x - interp1d(fpr, tpr)(x), 0., 1.)
	return eer

def evaluate_model(mode, model, db_gen, device):
	if mode not in ['val', 'eval']: raise ValueError('mode should be either "val" or "eval"')
	model.eval()
	y = []
	y_scores = []
	with torch.set_grad_enabled(False):
		with tqdm(total = len(db_gen), ncols = 70) as pbar:
			for m_batch, m_batch2, m_label in db_gen:
				m_batch, m_batch2 = m_batch.to(device), m_batch2.to(device)
				o = model(m_batch, m_batch2)
				#print(o)
				y_scores.extend(torch.softmax(o, dim=-1).cpu().numpy()[:,0].tolist()) #>>> (batchsize, 3), 0: trg, 1:non_trg, 2:spoofed
				y.extend(m_label)
				pbar.update(1)
		fpr, tpr, thresholds = roc_curve(y, y_scores, pos_label = 1)
		eer = brentq(lambda x:1. - x - interp1d(fpr, tpr)(x), 0., 1.)

	return eer, y, y_scores

class AdditiveMarginSoftmax(nn.Module):
    # AMSoftmax
    def __init__(self, margin=0.35, s=30):
        super().__init__()

        self.m = margin #
        self.s = s
        self.epsilon = 0.000000000001
        print('AMSoftmax m = ' + str(margin))

    def forward(self, predicted, target):

        # ------------ AM Softmax ------------ #
        predicted = predicted / (predicted.norm(p=2, dim=0) + self.epsilon)
        indexes = range(predicted.size(0))
        cos_theta_y = predicted[indexes, target]
        cos_theta_y_m = cos_theta_y - self.m
        exp_s = np.e ** (self.s * cos_theta_y_m)

        sum_cos_theta_j = (np.e ** (predicted * self.s)).sum(dim=1) - (np.e ** (predicted[indexes, target] * self.s))

        log = -torch.log(exp_s/(exp_s+sum_cos_theta_j+self.epsilon)).mean()

        return log

def main():
	parser = argparse.ArgumentParser()
	#dir
	parser.add_argument('-name', type = str, required = True)
	parser.add_argument('-save_dir', type = str, default = '/exp/DNNs/')
	parser.add_argument('-embd_dir', type=str, default = '/DB/embd/202002_vox_RawNet2/TTA_')

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

	#set random seed
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	#load comet
	experiment = Experiment(api_key="9CueLwB3ujfFlhdD9Z2VpKKaq",
		project_name="RawNet2", workspace="jungjee",
		auto_output_logging = 'simple',
		disabled = args.comet_disable)
	experiment.set_name(args.name)

	#device setting
	cuda = torch.cuda.is_available()
	device = torch.device('cuda' if cuda else 'cpu')
	print('Device: {}'.format(device))
	
	#set save directory
	save_dir = args.save_dir+args.name+'/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	if not os.path.exists(save_dir+'results/'):
		os.makedirs(save_dir+'results/')
	if not os.path.exists(save_dir+'models/'):
		os.makedirs(save_dir+'models/')


	with open(args.embd_dir + 'vox1_dev.pk', 'rb') as f:
		d_embd_vox1_dev = pk.load(f)

	with open(args.embd_dir + 'vox1_eval.pk', 'rb') as f:
		d_embd_vox1_eval = pk.load(f)

	d_embd_vox1 = {**d_embd_vox1_dev, **d_embd_vox1_eval}

	f_eer = open(save_dir + 'eers.txt', 'a', buffering = 1)

	with open('/DB/VoxCeleb2/veri_test.txt' , 'r') as f:
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
	text = 'Original evaluation EER: {}'.format(eval_eer)
	print(text)
	experiment.log_text(text)
	f_eer.write(text+'\n')

	with open('/DB/VoxCeleb2/list_test_all.txt' , 'r') as f:
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
	text = 'Vox1_E evaluation EER: {}'.format(eval_eer)
	print(text)
	experiment.log_text(text)
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
	experiment.log_text(text)
	f_eer.write(text+'\n')

	with open('/DB/VoxCeleb2/list_test_hard.txt' , 'r') as f:
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
	text = 'Vox1_H evaluation EER: {}'.format(eval_eer)
	print(text)
	experiment.log_text(text)
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
	experiment.log_text(text)
	f_eer.write(text+'\n')
	f_eer.close()

if __name__ == '__main__':
	main()