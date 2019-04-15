import os
import numpy as np
np.random.seed(1016)
import yaml
import queue
import struct
from multiprocessing import Process
from threading import Thread
from tqdm import tqdm
from time import sleep
from keras.utils import multi_gpu_model, plot_model, to_categorical
from keras.optimizers import *
from keras.models import Model
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from model_rwcnn_gru_cLoss_bsLoss import get_model

def cos_sim(a,b):
	return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def simple_loss(y_true, y_pred):
	return K.mean(y_pred)

def zero_loss(y_true, y_pred):
	return 0.5 * K.sum(y_pred, axis=0)

def compose_spkFeat_dic(lines, model, f_desc_dic, base_dir):
	dic_spkFeat = {}
	for line in tqdm(lines, desc='extracting spk feats'):
		k, f, p = line.strip().split(' ')
		p = int(p)
		if f not in f_desc_dic:
			f_tmp = '/'.join([base_dir, f])
			f_desc_dic[f] = open(f_tmp, 'rb')

		f_desc_dic[f].seek(p)
		l = struct.unpack('i', f_desc_dic[f].read(4))[0]
		utt = np.asarray(struct.unpack('%df'%l, f_desc_dic[f].read(l * 4)), dtype=np.float32)
		spkFeat = model.predict(utt.reshape(1,-1,1))[0]
		dic_spkFeat[k] = spkFeat

	return dic_spkFeat

def make_spkdic(lines):
	idx = 0
	dic_spk = {}
	list_spk = []
	for line in lines:
		k, f, p = line.strip().split(' ')
		spk = k.split('/')[0]
		if spk not in dic_spk:
			dic_spk[spk] = idx
			list_spk.append(spk)
			idx += 1
	return (dic_spk, list_spk)

def compose_batch(lines, f_desc_dic, dic_spk, nb_samp, base_dir):
	'''
	designed to read pre-emphasized floats!
	'''
	batch = []
	ans = []
	for line in lines:
		k, f, p = line.strip().split(' ')
		ans.append(dic_spk[k.split('/')[0]])
		p = int(p)
		if f not in f_desc_dic:
			f_tmp = '/'.join([base_dir, f])
			f_desc_dic[f] = open(f_tmp, 'rb')

		f_desc_dic[f].seek(p)
		l = struct.unpack('i', f_desc_dic[f].read(4))[0]
		utt = struct.unpack('%df'%l, f_desc_dic[f].read(l * 4))
		_nb_samp = len(utt)
		#need to verify this part later!!!!!!
		assert _nb_samp >= nb_samp
		cut = np.random.randint(low = 0, high = _nb_samp - nb_samp)
		utt = utt[cut:cut+nb_samp]
		batch.append(utt)

	return (np.asarray(batch, dtype=np.float32).reshape(len(lines), -1, 1), np.asarray(ans))

def process_epoch(lines, q, batch_size, nb_samp, dic_spk, base_dir): 
	f_desc_dic = {}
	nb_batch = int(len(lines) / batch_size)
	for i in range(nb_batch):
		while True:
			if q.full():
				sleep(0.1)
			else:
				q.put(compose_batch(lines = lines[i*batch_size: (i+1)*batch_size],
					f_desc_dic = f_desc_dic,
					dic_spk = dic_spk,
					nb_samp = nb_samp,
					base_dir = base_dir))
				break

	for k in f_desc_dic.keys():
		f_desc_dic[k].close()

	return
		

#======================================================================#
#======================================================================#
_abspath = os.path.abspath(__file__)
dir_yaml = os.path.splitext(_abspath)[0] + '.yaml'
with open(dir_yaml, 'r') as f_yaml:
	parser = yaml.load(f_yaml)

dir_dev_scp = parser['dev_scp']
with open(dir_dev_scp, 'r') as f_dev_scp:
	dev_lines = f_dev_scp.readlines()
dic_spk, list_spk = make_spkdic(dev_lines)
parser['model']['nb_spk'] = len(list_spk)
print('# spk: ', len(list_spk))
parser['model']['batch_size'] = int(parser['batch_size'] / parser['nb_gpu'])
assert parser['batch_size'] % parser['nb_gpu']  == 0

val_lines = []
for l in dev_lines:
	if  l[0] == 'B':
		val_lines.append(l)

eval_lines = open(parser['eval_scp'], 'r').readlines()
trials = open(parser['trials'], 'r').readlines()
val_trials = open(parser['val_trials'], 'r').readlines()


model, m_name = get_model(argDic = parser['model'])
save_dir = parser['save_dir'] + m_name + '_' + parser['name'] + '/'

if bool(parser['restart']):
	model.load_weights(save_dir + parser['restart_model'])
	restart_epo = int(parser['restart_model'].split('-')[0]) + 1
	print('restarting!!')
else:
	#load CNN_model_partially
	model.load_weights(parser['dir_base_model'], by_name = True)
	restart_epo = 0 

if bool(parser['freeze']):
	for layer in model.layers:
		if 'gru' not in layer.name:
			print(layer.name)
			layer.trainable = False

model_pred = Model(inputs=model.get_layer('input_1').input, outputs=model.get_layer('gru_code').output)


#make folders
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

f_params = open(save_dir + 'f_params.txt', 'w')
for k, v in parser.items():
	print(k, v)
	f_params.write('{}:\t{}\n'.format(k, v))
f_params.write('DNN model params\n')

for k, v in parser['model'].items():
	f_params.write('{}:\t{}\n'.format(k, v))
print(m_name)
f_params.write('model_name: %s\n'%m_name)
f_params.close()


#split dev and val set
np.random.seed(1016)
np.random.shuffle(dev_lines)

nb_batch = int(len(dev_lines) / parser['batch_size'])

'''
model_json = model.to_json()
with open(save_dir + 'arc.json', 'w') as f_json:
	f_json.write(model_json)
'''

with open(save_dir + 'summary.txt' ,'w+') as f_summary:
	model.summary(print_fn=lambda x: f_summary.write(x + '\n'))

#plot_model(model, to_file=parser['save_dir'] +'visualization.png', show_shapes=True)

if parser['optimizer'] == 'SGD':
	optimizer = eval(parser['optimizer'])(lr=parser['lr'], momentum=parser['momentum'])
elif parser['optimizer'] == 'Adam':
	optimizer = eval(parser['optimizer'])(lr=parser['lr'], decay = parser['opt_decay'], amsgrad = bool(parser['amsgrad']))
elif parser['optimizer'] == 'RMSprop':
	optimizer = eval(parser['optimizer'])(lr=parser['lr'], decay = parser['opt_decay'])

if bool(parser['mg']):
	model_mg = multi_gpu_model(model, gpus=parser['nb_gpu'])
	model_mg.compile(optimizer = optimizer,
		loss = {'gru_s_bs_loss':simple_loss,
				'gru_c_loss':zero_loss},
		loss_weights = {'gru_s_bs_loss':1, 'gru_c_loss':parser['c_lambda']},
		metrics=['accuracy'])
model.compile(optimizer = optimizer,
		loss = {'gru_s_bs_loss':simple_loss,
				'gru_c_loss':zero_loss},
		loss_weights = {'gru_s_bs_loss':1, 'gru_c_loss': parser['c_lambda']},
	metrics=['accuracy'])

global q
q = queue.Queue(maxsize=1000)
dummy_y = np.zeros((parser['batch_size'], 1))

if not os.path.exists(save_dir  + 'results/'):
	os.makedirs(save_dir + 'results/')
f_eer = open(save_dir + 'eers.txt', 'a', buffering=1)

for epoch in tqdm(range(restart_epo, parser['epoch'])):
	np.random.shuffle(dev_lines)
	p = Thread(target = process_epoch, args = (dev_lines,
							q,
							parser['batch_size'],
							parser['nb_samp'],
							dic_spk,
							parser['base_dir']))
	p.start()

	#train!
	loss = 999.
	loss1 = 999.
	loss2 = 999.
	acc = 0.
	pbar = tqdm(range(nb_batch))
	for b in pbar:
		pbar.set_description('epoch: %d, loss: %.3f, loss1: %.3f, loss2: %.3f acc: %f'%(epoch, loss, loss1, loss2, acc))
		while True:
			if q.empty():
				sleep(0.1)

			else:
				x, y = q.get()
				y = to_categorical(y, num_classes=parser['model']['nb_spk'])
				if bool(parser['mg']):
					loss, loss1, loss2, acc1, acc2 = model_mg.train_on_batch([x, y], [dummy_y, dummy_y])
				else:
					loss, loss1, loss2, acc1, acc2 = model.train_on_batch([x, y], [dummy_y, dummy_y])
				
				break


	p.join()

	#validate!
	dic_val = compose_spkFeat_dic(lines = val_lines,
						model = model_pred,
						f_desc_dic = {},
						base_dir = parser['base_dir'])
	
	y = []
	y_score = []
	for smpl in val_trials:
		#target, utt_a, utt_b = smpl.strip().split(' ')
		target, spkMd, utt = smpl.strip().split(' ')
		target = int(target)
		#target = 1 if spkMd == utt.split('')[0]
		cos_score = cos_sim(dic_val[spkMd], dic_val[utt])
		y.append(target)
		y_score.append(cos_score)
	fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
	eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
	print('\nepoch: %d, val_eer: %f'%(int(epoch), eer))
	#thresh = interp1d(fpr, thresholds)(eer)
	f_eer.write('%d %f '%(epoch, eer))

	model.save_weights(save_dir +  '%d-%.4f.h5'%(epoch, eer))

	#evaluate!
	dic_eval = compose_spkFeat_dic(lines = eval_lines,
						model = model_pred,
						f_desc_dic = {},
						base_dir = parser['base_dir'])

	f_res = open(save_dir + 'results/epoch%s.txt'%(epoch), 'w')
	#new style python-driven EER calculation.
	y = []
	y_score = []
	for smpl in trials:
		#target, utt_a, utt_b = smpl.strip().split(' ')
		target, spkMd, utt = smpl.strip().split(' ')
		target = int(target)
		#target = 1 if spkMd == utt.split('')[0]
		cos_score = cos_sim(dic_eval[spkMd], dic_eval[utt])
		y.append(target)
		y_score.append(cos_score)
		f_res.write('{score} {target}\n'.format(score=cos_score,target=target))
	f_res.close()
	fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
	eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
	print('\nepoch: %d, eer: %f'%(int(epoch), eer))
	#thresh = interp1d(fpr, thresholds)(eer)
	f_eer.write('%f\n'%(eer))
f_eer.close()
































