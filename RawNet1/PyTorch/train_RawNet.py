from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import os
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data

from model_RawNet import RawNet

def keras_lr_decay(step, decay = 0.00005):
	return 1./(1.+decay*step)
	
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


def train_model(model, device, db_gen, optimizer, epoch):
	model.train()
	with tqdm(total = len(db_gen), ncols = 70) as pbar:
		for idx_ct, (m_batch, m_label) in enumerate(db_gen):
			if bool(parser['do_lr_decay']):
				if parser['lr_decay'] == 'keras': lr_scheduler.step()
					
			m_batch = m_batch.to(device)
			m_label= m_label.to(device)

			output = model(m_batch, m_label) #output
			'''
			#for future updates including bc_loss and h_loss
			if bool(parser['mg']):
				norm = torch.norm(model.module.fc2_gru.weight, dim=1, keepdim = True) / (5. ** 0.5)
				normed_weight = torch.div(model.module.fc2_gru.weight, norm)
			else:
				norm = torch.norm(model.fc2_gru.weight, dim=1, keepdim = True) / (5. ** 0.5)
				normed_weight = torch.div(model.fc2_gru.weight, norm)
			'''
			cce_loss = criterion(output, m_label)
			# bc_loss, h_loss currently removed.
			loss = cce_loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if idx_ct % 100 == 0:
				for p in optimizer.param_groups:
					lr_cur = p['lr']
					#print('lr_cur', lr_cur)
					break
			pbar.set_description('epoch%d,cce:%.3f, cur_lr:%.6f'%(epoch, cce_loss,float(lr_cur)))
			pbar.update(1)

def evaluate_model(mode, model, db_gen, device, l_utt, save_dir, epoch, l_trial):
	if mode not in ['val', 'eval']: raise ValueError('mode should be either "val" or "eval"')
	model.eval()
	with torch.set_grad_enabled(False):
		#1st, extract speaker embeddings.
		l_embeddings = []
		with tqdm(total = len(db_gen), ncols = 70) as pbar:
			for m_batch in db_gen:
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
		if mode == 'val':
			f_res = open(save_dir + 'results/epoch%s.txt'%(epoch), 'w')
		else:
			f_res = open(save_dir + 'results/eval.txt', 'w')

		for line in l_trial:
			trg, utt_a, utt_b = line.strip().split(' ')
			y.append(int(trg))
			y_score.append(cos_sim(d_embeddings[utt_a], d_embeddings[utt_b]))
			f_res.write('{score} {target}\n'.format(score=y_score[-1],target=y[-1]))
		f_res.close()
		fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
		eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
		print('eer', eer)
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
	for r, ds, fs in os.walk(src_dir):
		base = '/'.join(r.split('/')[-2:])+'/'
		for f in fs:
			if f[-3:] != 'npy':
				continue
			l_utt.append(base+f[:-4])

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
	def __init__(self, list_IDs, base_dir, nb_time = 0, labels = {}, cut = True, return_label = True, pre_emp = True):
		'''
		self.list_IDs	: list of strings (each string: utt key)
		self.labels		: dictionary (key: utt key, value: label integer)
		self.nb_time	: integer, the number of timesteps for each mini-batch
		cut				: (boolean) adjust utterance duration for mini-batch construction
		return_label	: (boolean) 
		pre_emp			: (boolean) do pre-emphasis with coefficient = 0.97
		'''
		self.list_IDs = list_IDs
		self.nb_time = nb_time
		self.base_dir = base_dir
		self.labels = labels
		self.cut = cut
		self.return_label = return_label
		self.pre_emp = pre_emp
		if self.cut and self.nb_time == 0: raise ValueError('when adjusting utterance length, "nb_time" should be input')

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		ID = self.list_IDs[index]
		try:
			X = np.load(self.base_dir+ID+'.npy')
		except:
			raise ValueError('%s'%ID)
		if self.pre_emp: X = self._pre_emphasis(X)
		if self.cut:
			nb_time = X.shape[1]
			if nb_time > self.nb_time:
				start_idx = np.random.randint(low = 0,
					high = nb_time - self.nb_time)
				X = X[:, start_idx:start_idx+self.nb_time]
			elif nb_time < self.nb_time:
				nb_dup = int(self.nb_time / nb_time) + 1
				X = np.tile(X, (1, nb_dup))[:, :self.nb_time]
			else:
				X = X
		if not self.return_label:
			return X
		y = self.labels[ID.split('/')[0]]
		return X, y

	def _pre_emphasis(self, x):
		'''
		Pre-emphasis for single channel input
		'''
		return np.asarray(x[:,1:] - 0.97 * x[:, :-1], dtype=np.float32) 

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

if __name__ == '__main__':
	#load yaml file & set comet_ml config
	_abspath = os.path.abspath(__file__)
	dir_yaml = os.path.splitext(_abspath)[0] + '.yaml'
	with open(dir_yaml, 'r') as f_yaml:
		parser = yaml.load(f_yaml)
	np.random.seed(parser['seed'])
	
	#device setting
	cuda = torch.cuda.is_available()
	device = torch.device('cuda' if cuda else 'cpu')
	print(device)

	#get utt_lists & define labels
	l_dev  = sorted(get_utt_list(parser['DB_vox2']+parser['dev_wav']))
	l_val  = sorted(get_utt_list(parser['DB']+parser['val_wav']))
	l_eval  = sorted(get_utt_list(parser['DB']+parser['eval_wav']))
	d_label_vox2 = get_label_dic_Voxceleb(l_dev)
	parser['model']['nb_classes'] = len(list(d_label_vox2.keys()))

	#def make_validation_trial(l_utt, nb_trial, dir_val_trial):
	if bool(parser['make_val_trial']):
		make_validation_trial(l_utt=l_val, nb_trial=parser['nb_val_trial'], dir_val_trial=parser['DB']+'val_trial.txt')
	with open(parser['DB']+'val_trial.txt', 'r') as f:
		l_val_trial = f.readlines()
	with open(parser['DB']+'veri_test.txt', 'r') as f:
		l_eval_trial = f.readlines()

	'''
	# for debugging
	if bool(parser['comet_disable']):
		l_dev = l_dev[:2000]
		l_val_trial = l_val_trial[:2000]
		l_eval_trial = l_eval_trial[:2000]
		l_eval = get_val_utts(l_eval_trial)
	'''

	#define dataset generators
	l_val = get_val_utts(l_val_trial)
	devset = Dataset_VoxCeleb2(list_IDs = l_dev,
		labels = d_label_vox2,
		nb_time = parser['nb_time'],
		base_dir = parser['DB_vox2']+parser['dev_wav'])
	devset_gen = data.DataLoader(devset,
		batch_size = parser['batch_size'],
		shuffle = True,
		drop_last = True,
		num_workers = parser['nb_proc_db'])
	valset = Dataset_VoxCeleb2(list_IDs = l_val,
		return_label = False,
		nb_time = parser['nb_time'],
		base_dir = parser['DB']+parser['val_wav'])
	valset_gen = data.DataLoader(valset,
		batch_size = parser['batch_size'],
		shuffle = False,
		drop_last = False,
		num_workers = parser['nb_proc_db'])
	evalset = Dataset_VoxCeleb2(list_IDs = l_eval,
		cut = False,
		return_label = False,
		base_dir = parser['DB']+parser['eval_wav'])
	evalset_gen = data.DataLoader(evalset,
		batch_size = 1, #because for evaluation, we do not modify its duration, thus cannot compose mini-batches
		shuffle = False,
		drop_last = False,
		num_workers = parser['nb_proc_db'])

	#set save directory
	save_dir = parser['save_dir'] + parser['name'] + '/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	if not os.path.exists(save_dir  + 'results/'):
		os.makedirs(save_dir + 'results/')
	if not os.path.exists(save_dir  + 'models/'):
		os.makedirs(save_dir + 'models/')
	
	#log experiment parameters to local and comet_ml server
	f_params = open(save_dir + 'f_params.txt', 'w')
	for k, v in parser.items():
		print(k, v)
		f_params.write('{}:\t{}\n'.format(k, v))
	f_params.write('DNN model params\n')
	
	for k, v in parser['model'].items():
		f_params.write('{}:\t{}\n'.format(k, v))
	f_params.close()

	#define model
	if bool(parser['mg']):
		model_1gpu = RawNet(parser['model'], device)
		nb_params = sum([param.view(-1).size()[0] for param in model_1gpu.parameters()])
		model = nn.DataParallel(model_1gpu).to(device)
	else:
		model = RawNet(parser['model'], device).to(device)
		nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
	model.apply(init_weights)
	print('nb_params: {}'.format(nb_params))

	#set ojbective funtions
	criterion = nn.CrossEntropyLoss()

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
	if parser['optimizer'].lower() == 'sgd':
		optimizer = torch.optim.SGD(params,
			lr = parser['lr'],
			momentum = parser['opt_mom'],
			weight_decay = parser['wd'],
			nesterov = bool(parser['nesterov']))
	elif parser['optimizer'].lower() == 'adam':
		optimizer = torch.optim.Adam(params,
			lr = parser['lr'],
			weight_decay = parser['wd'],
			amsgrad = bool(parser['amsgrad']))
	else:
		raise NotImplementedError('Add other optimizers if needed')

	if bool(parser['do_lr_decay']):
		if parser['lr_decay'] == 'keras':
			lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: keras_lr_decay(step))

	##########################################
	#Train####################################
	##########################################
	best_eer = 99.
	f_eer = open(save_dir + 'eers.txt', 'a', buffering = 1)
	for epoch in tqdm(range(parser['epoch'])):
		#train phase
		train_model(model = model,
			device = device,
			db_gen = devset_gen,
			optimizer = optimizer,
			epoch = epoch)

		#validation phase
		val_eer = evaluate_model(mode = 'val',
			model = model,
			db_gen = valset_gen, 
			device = device,
			l_utt = l_val,
			save_dir = save_dir,
			epoch = epoch,
			l_trial = l_val_trial)
		f_eer.write('epoch:%d,val_eer:%f\n'%(epoch, val_eer))

		save_model_dict = model_1gpu.state_dict() if bool(parser['mg']) else model.state_dict()
		#record best validation model
		if float(val_eer) < best_eer:
			print('New best validation EER: %f'%float(val_eer))
			best_eer = float(val_eer)
			#save best model
			torch.save(save_model_dict, save_dir +  'models/best.pt')
			torch.save(optimizer.state_dict(), save_dir + 'models/best_opt.pt')
			
		if not bool(parser['save_best_only']):
			torch.save(save_model_dict, save_dir +  'models/%d-%.6f.pt'%(epoch, val_eer))
			torch.save(optimizer.state_dict(), save_dir + 'models/opt_%d-%.6f.pt'%(epoch, val_eer))

		eval_eer = evaluate_model(mode = 'eval',
			model = model,
			db_gen = evalset_gen, 
			device = device,
			l_utt = l_eval,
			save_dir = save_dir,
			epoch = epoch,
			l_trial = l_eval_trial)
		f_eer.write('epoch:%d,Eval eer:%f\n'%(epoch, eval_eer))
	f_eer.close()












