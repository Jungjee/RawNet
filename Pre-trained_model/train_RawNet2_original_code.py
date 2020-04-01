from comet_ml import Experiment
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import os
import argparse
import json
import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
from torch.utils import data

from model_RawNet_filter_attention_ensemble_sinc import RawNet

def str2bool(v):
	if isinstance(v, bool):
	   return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


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

def train_model(model, db_gen, optimizer, epoch, args, experiment, device, lr_scheduler, criterion):
	model.train()
	with tqdm(total = len(db_gen), ncols = 70) as pbar:
		for idx_ct, (m_batch, m_label) in enumerate(db_gen):
			
			m_batch, m_label = m_batch.to(device), m_label.to(device)

			output = model(m_batch, m_label)
			cce_loss = criterion['cce'](output, m_label)
			loss = cce_loss
			if idx_ct % 100 == 0 and not bool(args.debug):
				experiment.log_metric('cce_loss', cce_loss.detach().cpu().numpy())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			pbar.set_description('epoch: %d, cce:%.3f'%(epoch, cce_loss))
			pbar.update(1)
			if args.do_lr_decay:
				if args.lr_decay == 'keras': lr_scheduler.step()

def evaluate_model(mode, model, db_gen, l_utt, save_dir, epoch, l_trial, args, device):
	if mode not in ['val','eval']: raise ValueError('mode should be either "val" or "eval"')
	model.eval()
	with torch.set_grad_enabled(False):
		#1st, extract speaker embeddings.
		l_embeddings = []
		with tqdm(total = len(db_gen), ncols = 70) as pbar:
			for m_batch in db_gen:
				m_batch = m_batch.to(device)
				code = model(x = m_batch, is_test=True)
				l_embeddings.extend(code.cpu().numpy()) #>>> (batchsize, codeDim)
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
		fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
		eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
	return eer

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
		fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
		eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
	return eer

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
			#if file[-3:] != 'npy':
			if file[-3:] != 'wav':
				continue
			#l_utt.append(base+file[:-4])
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

class Dataset_VoxCeleb2(data.Dataset):
	def __init__(self, list_IDs, base_dir, nb_samp = 0, labels = {}, cut = True, return_label = True, norm_scale = True):
		'''
		self.list_IDs	: list of strings (each string: utt key)
		self.labels		: dictionary (key: utt key, value: label integer)
		self.nb_samp	: integer, the number of timesteps for each mini-batch
		cut				: (boolean) adjust utterance duration for mini-batch construction
		return_label	: (boolean) 
		norm_scale		: (boolean) normalize scale alike SincNet github repo
		'''
		self.list_IDs = list_IDs
		self.nb_samp = nb_samp
		self.base_dir = base_dir
		self.labels = labels
		self.cut = cut
		self.return_label = return_label
		self.norm_scale = norm_scale
		if self.cut and self.nb_samp == 0: raise ValueError('when adjusting utterance length, "nb_samp" should be input')

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		ID = self.list_IDs[index]
		try:
			#X, _ = sf.read(self.base_dir+ID, dtype='int16') 
			X, _ = sf.read(self.base_dir+ID) 
			X = X.astype(np.float64)
		except:
			raise ValueError('%s'%ID)

		if self.norm_scale:
			X = self._normalize_scale(X).astype(np.float32)
		X = X.reshape(1,-1) #because of LayerNorm for the input

		if self.cut:
			nb_time = X.shape[1]
			if nb_time > self.nb_samp:
				start_idx = np.random.randint(low = 0, high = nb_time - self.nb_samp)
				X = X[:, start_idx : start_idx + self.nb_samp][0]
			elif nb_time < self.nb_samp:
				nb_dup = int(self.nb_samp / nb_time) + 1
				X = np.tile(X, (1, nb_dup))[:, :self.nb_samp][0]
			else:
				X = X[0]
		if not self.return_label:
			return X
		y = self.labels[ID.split('/')[0]]
		return X, y

	def _normalize_scale(self, x):
		'''
		Normalize sample scale alike SincNet.
		'''
		return x/np.max(np.abs(x))

class TA_Dataset_VoxCeleb2(data.Dataset):
	def __init__(self, list_IDs, base_dir, nb_samp = 0, window_size = 0, labels = {}, cut = True, return_label = True, norm_scale = True):
		'''
		self.list_IDs	: list of strings (each string: utt key)
		self.labels		: dictionary (key: utt key, value: label integer)
		self.nb_samp	: integer, the number of timesteps for each mini-batch
		cut				: (boolean) adjust utterance duration for mini-batch construction
		return_label	: (boolean) 
		norm_scale		: (boolean) normalize scale alike SincNet github repo
		'''
		self.list_IDs = list_IDs
		self.window_size = window_size
		self.nb_samp = nb_samp
		self.base_dir = base_dir
		self.labels = labels
		self.cut = cut
		self.return_label = return_label
		self.norm_scale = norm_scale
		if self.cut and self.nb_samp == 0: raise ValueError('when adjusting utterance length, "nb_samp" should be input')

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		ID = self.list_IDs[index]
		try:
			#X, _ = sf.read(self.base_dir+ID, dtype='int16') 
			X, _ = sf.read(self.base_dir+ID) 
			X = X.astype(np.float64)
		except:
			raise ValueError('%s'%ID)

		if self.norm_scale:
			X = self._normalize_scale(X).astype(np.float32)
		X = X.reshape(1,-1)

		list_X = []
		nb_time = X.shape[1]
		if nb_time < self.nb_samp:
			nb_dup = int(self.nb_samp / nb_time) + 1
			list_X.append(np.tile(X, (1, nb_dup))[:, :self.nb_samp][0])
		elif nb_time > self.nb_samp:
			step = self.nb_samp - self.window_size
			iteration = int( (nb_time - self.window_size) / step ) + 1
			for i in range(iteration):
				if i == 0:
					list_X.append(X[:, :self.nb_samp][0])
				elif i < iteration - 1:
					list_X.append(X[:, i*step : i*step + self.nb_samp][0])
				else:
					list_X.append(X[:, -self.nb_samp:][0])
		else :
			list_X.append(X[0])

		if not self.return_label:
			return list_X
		y = self.labels[ID.split('/')[0]]
		return list_X, y 

	def _normalize_scale(self, x):
		'''
		Normalize sample scale alike SincNet.
		'''
		return x/np.max(np.abs(x))

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
	#print('nb_spk: %d'%len(l_spk))
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

def main():
	parser = argparse.ArgumentParser()
	#dir
	parser.add_argument('-name', type = str, required = True)
	parser.add_argument('-save_dir', type = str, default = '/exp/DNNs/')
	parser.add_argument('-DB', type = str, default = '/DB/VoxCeleb1/')
	parser.add_argument('-DB_vox2', type = str, default = '/DB/VoxCeleb2/')
	parser.add_argument('-dev_wav', type = str, default = 'wav/')
	parser.add_argument('-val_wav', type = str, default = 'dev_wav/')
	parser.add_argument('-eval_wav', type = str, default = 'eval_wav/')
	
	#hyper-params
	parser.add_argument('-bs', type = int, default = 120)
	parser.add_argument('-lr', type = float, default = 0.001)
	parser.add_argument('-nb_samp', type = int, default = 59049)
	parser.add_argument('-window_size', type = int, default = 11810)
	
	parser.add_argument('-wd', type = float, default = 0.0001)
	parser.add_argument('-epoch', type = int, default = 40)
	parser.add_argument('-optimizer', type = str, default = 'Adam')
	parser.add_argument('-nb_worker', type = int, default = 8)
	parser.add_argument('-temp', type = float, default = .5)
	parser.add_argument('-seed', type = int, default = 1234) 
	parser.add_argument('-nb_val_trial', type = int, default = 40000) 
	parser.add_argument('-lr_decay', type = str, default = 'keras')
	parser.add_argument('-model', type = json.loads, default = 
		'{"first_conv":251, "in_channels":1, "filts":[128, [128,128], [128,256], [256,256]],' \
		'"blocks":[2,4], "nb_fc_att_node":[1], "nb_fc_node":1024, "gru_node":1024, "nb_gru_layer":1}')
	parser.add_argument('-load_model_dir', type = str, default = '')
	parser.add_argument('-load_model_opt_dir', type = str, default = '')
	
	#flag
	parser.add_argument('-amsgrad', type = str2bool, nargs='?', const=True, default = True)
	parser.add_argument('-make_val_trial', type = str2bool, nargs='?', const=True, default = False)
	parser.add_argument('-debug', type = str2bool, nargs='?', const=True, default = False)
	parser.add_argument('-comet_disable', type = str2bool, nargs='?', const=True, default = False)
	parser.add_argument('-save_best_only', type = str2bool, nargs='?', const=True, default = False)
	parser.add_argument('-do_lr_decay', type = str2bool, nargs='?', const=True, default = True)
	parser.add_argument('-mg', type = str2bool, nargs='?', const=True, default = True)
	parser.add_argument('-load_model', type = str2bool, nargs='?', const=True, default = False)

	args = parser.parse_args()
	args.model['nb_samp'] = args.nb_samp

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

	#get utt_lists & define labels
	l_dev = sorted(get_utt_list(args.DB_vox2 + args.dev_wav))
	l_val = sorted(get_utt_list(args.DB + args.val_wav))
	l_eval = sorted(get_utt_list(args.DB + args.eval_wav))
	d_label_vox2 = get_label_dic_Voxceleb(l_dev)
	args.model['nb_classes'] = len(list(d_label_vox2.keys()))
	#print(len(l_dev), len(l_val), len(l_eval))
	#exit()

	#def make_validation_trial(l_utt, nb_trial, dir_val_trial):
	if bool(args.make_val_trial):
		make_validation_trial(l_utt = l_val, nb_trial = args.nb_val_trial, dir_val_trial = args.DB + 'val_trial.txt')
	with open(args.DB + 'val_trial.txt', 'r') as f:
		l_val_trial = f.readlines()
	with open(args.DB + 'veri_test.txt', 'r') as f:
		l_eval_trial = f.readlines()
		
	# for debugging
	if bool(args.debug):
		l_dev = l_dev[:120]

	#define dataset generators
	devset = Dataset_VoxCeleb2(list_IDs = l_dev,
		labels = d_label_vox2,
		nb_samp = args.nb_samp,
		base_dir = args.DB_vox2 + args.dev_wav)
	devset_gen = data.DataLoader(devset,
		batch_size = args.bs,
		shuffle = True,
		drop_last = True,
		num_workers = args.nb_worker)
	valset = Dataset_VoxCeleb2(list_IDs = l_val,
		return_label = False,
		nb_samp = args.nb_samp,
		base_dir = args.DB + args.val_wav)
	valset_gen = data.DataLoader(valset,
		batch_size = args.bs,
		shuffle = False,
		drop_last = False,
		num_workers = args.nb_worker)
	TA_evalset = TA_Dataset_VoxCeleb2(list_IDs = l_eval,
		return_label = False,
		window_size = args.window_size, # 20% of nb_samp
		nb_samp = args.nb_samp, 
		base_dir = args.DB+args.eval_wav)
	TA_evalset_gen = data.DataLoader(TA_evalset,
		batch_size = 1, 
		shuffle = False,
		drop_last = False,
		num_workers = args.nb_worker)

	#set save directory
	save_dir = args.save_dir + args.name + '/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	if not os.path.exists(save_dir+'results/'):
		os.makedirs(save_dir+'results/')
	if not os.path.exists(save_dir+'models/'):
		os.makedirs(save_dir+'models/')
		
	#log experiment parameters to local and comet_ml server
	f_params = open(save_dir + 'f_params.txt', 'w')
	for k, v in sorted(vars(args).items()):
		print(k, v)
		f_params.write('{}:\t{}\n'.format(k, v))
	for k, v in sorted(args.model.items()):
		print(k, v)
		f_params.write('{}:\t{}\n'.format(k, v))
	f_params.close()
	if not bool(args.debug):
		experiment.log_parameters(vars(args))
		experiment.log_parameters(args.model)
	
	#define model
	if bool(args.mg):
		model_1gpu = RawNet(args.model, device)
		if args.load_model: model_1gpu.load_state_dict(torch.load(args.load_model_dir))
		nb_params = sum([param.view(-1).size()[0] for param in model_1gpu.parameters()])
		model = nn.DataParallel(model_1gpu).to(device)
	else:
		model = RawNet(args.model, device).to(device)
		if args.load_model: model.load_state_dict(torch.load(args.load_model_dir))
		nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
	if not args.load_model: model.apply(init_weights)
	print('nb_params: {}'.format(nb_params))

	#set ojbective funtions
	criterion = {}
	criterion['cce'] = nn.CrossEntropyLoss()
	
	#set optimizer
	params = [
		{
			'params': [
				param for name, param in model.named_parameters()
				if 'bn' not in name
			]
		},
		{
			'params': [
				param for name, param in model.named_parameters()
				if 'bn' in name
			],
			'weight_decay':
			0
		},
	]
	
	#params = list(model.parameters())
	if args.optimizer.lower() == 'sgd':
		optimizer = torch.optim.SGD(params,
			lr = args.lr,
			momentum = args.opt_mom,
			weight_decay = args.wd,
			nesterov = args.nesterov)
	elif args.optimizer.lower() == 'adam':
		optimizer = torch.optim.Adam(params,
			lr = args.lr,
			weight_decay = args.wd,
			amsgrad = args.amsgrad)
	else:
		raise NotImplementedError('Add other optimizers if needed')
	if args.load_model: optimizer.load_state_dict(torch.load(args.load_model_opt_dir))
	
	#set learning rate decay
	if bool(args.do_lr_decay):
		if args.lr_decay == 'keras':
			lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: keras_lr_decay(step))
		elif args.lr_decay == 'cosine':
			raise NotImplementedError('Not implemented yet')
		else:
			raise NotImplementedError('Not implemented yet')
	 
	##########################################
	#Train####################################
	##########################################
	best_val_eer = 99.
	best_eval_eer = 99.
	best_TA_eval_eer = 99.
	f_eer = open(save_dir + 'eers.txt', 'a', buffering = 1)
	for epoch in tqdm(range(args.epoch)):
		#train phase
		train_model(model = model,
			db_gen = devset_gen,
			experiment = experiment,
			args = args,
			optimizer = optimizer,
			lr_scheduler = lr_scheduler,
			criterion = criterion,
			device = device,
			epoch = epoch)


		#validation phase
		val_eer = evaluate_model(mode = 'val',
			model = model,
			db_gen = valset_gen, 
			l_utt = l_val,
			save_dir = save_dir,
			epoch = epoch,
			device = device,
			l_trial = l_val_trial,
			args = args)
		f_eer.write('epoch:%d, val_eer:%.4f\n'%(epoch, val_eer))
		if not bool(args.debug): experiment.log_metric('val_eer', val_eer)
		
		TA_eval_eer = time_augmented_evaluate_model(mode = 'eval',
			model = model,
			db_gen = TA_evalset_gen, 
			l_utt = l_eval,
			save_dir = save_dir,
			epoch = epoch,
			device = device,
			l_trial = l_eval_trial,
			args = args)
		f_eer.write('epoch:%d, TA_eval_eer:%.4f\n'%(epoch, TA_eval_eer))
		if not bool(args.debug): experiment.log_metric('TA_eval_eer', TA_eval_eer)
		
		save_model_dict = model_1gpu.state_dict() if args.mg else model.state_dict()

		#record best validation model
		if float(val_eer) < best_val_eer:
			print('New best validation EER: %f'%float(val_eer))
			best_val_eer = float(val_eer)

			torch.save(save_model_dict, save_dir +  'models/best_val.pt')
			torch.save(optimizer.state_dict(), save_dir + 'models/best_opt_val.pt')
			if not bool(args.debug): experiment.log_metric('best_val_eer', val_eer)
			
		if float(TA_eval_eer) < best_TA_eval_eer:
			print('New best TA_EER: %f'%float(TA_eval_eer))
			best_TA_eval_eer = float(TA_eval_eer)

			torch.save(save_model_dict, save_dir +  'models/TA_%d_%.4f.pt'%(epoch, TA_eval_eer))
			torch.save(optimizer.state_dict(), save_dir + 'models/best_opt_eval.pt')
			if not bool(args.debug): experiment.log_metric('best_TA_eval_eer', TA_eval_eer)
		
	f_eer.close()

if __name__ == '__main__':
	main()
