# -*- coding: utf-8 -*-
import os
import numpy as np
np.random.seed(1016)
import yaml
import queue
import struct
import pickle
from multiprocessing import Process
from threading import Thread
from tqdm import tqdm
from time import sleep
from keras.utils import multi_gpu_model, plot_model, to_categorical
from keras.optimizers import *
from keras.models import Model
from keras.layers import Dense, Input
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import pickle as pk
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.backend.tensorflow_backend import set_session
set_session(tf.Session(config=config))

from model_bvec import get_model

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

def get_spk2utt_dic(lines):
	spk2utt = {}
	for line in lines:
		spk = line.strip().split('/')[0]
		if spk not in spk2utt:
			spk2utt[spk] = []
		spk2utt[spk].append(line.strip().split(' ')[0])
	return spk2utt

def compose_batch_e2e(lines_pairs, dic_embeddings):
	'''
	batch construction for end-to-end training.
	lines_pairs >>> (#batch size, 2)
	'''
	dat_enrol = []
	dat_test = []
	ans = []

	for line in lines_pairs:
		dat_enrol.append(dic_embeddings[line[0]])
		dat_test.append(dic_embeddings[line[1]])
		
		ans_cur = 1 if line[0].split('/')[0] == line[1].split('/')[0] else 0
		ans.append(ans_cur)

	return (np.asarray(dat_enrol, dtype=np.float32),
			np.asarray(dat_test, dtype=np.float32),
			np.asarray(ans, dtype=np.int32))

def process_epoch_e2e(lines, q, batch_size, nb_batch_per_epoch, dic_spk2utt, dic_embeddings): 
	list_spk = list(dic_spk2utt.keys())
	for i in range(nb_batch_per_epoch):
		lines_pairs = []
		client_spks = np.random.choice(list_spk, int(batch_size / 2), replace = False)
		for spk in client_spks:
			utt_keys = np.random.choice(dic_spk2utt[spk], 2, replace = False)
			lines_pairs.append([utt_keys[0], utt_keys[1]])

		for j in range(int(batch_size / 2)):
			spks = np.random.choice(list_spk, 2, replace = False)
			u_a = np.random.choice(dic_spk2utt[spks[0]], 1)[0]
			u_b = np.random.choice(dic_spk2utt[spks[1]], 1)[0]
			lines_pairs.append([u_a, u_b])
		while True:
				
			if q.full():
				sleep(0.1)
			else:
				q.put(compose_batch_e2e(lines_pairs = lines_pairs,
						dic_embeddings = dic_embeddings))
				break

	return
		


if __name__ == '__main__':
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
	
	val_lines = []
	for l in dev_lines:
		if  l[0] == 'B':
			val_lines.append(l)
	
	eval_lines = open(parser['eval_scp'], 'r').readlines()
	trials = open(parser['trials'], 'r').readlines()
	val_trials = open(parser['val_trials'], 'r').readlines()
	
	_ = pk.load(open(parser['gru_embeddings'], 'rb'))
	dev_dic_embeddings = _['dev_dic_embeddings']
	eval_dic_embeddings = _['eval_dic_embeddings']
	del _
	
	dev_list_embeddings = []
	for k in dev_dic_embeddings.keys():
		dev_list_embeddings.append(dev_dic_embeddings[k])
	gl_mean = np.mean(dev_list_embeddings, axis = 0)
	gl_std = np.std(dev_list_embeddings, axis = 0)
	dev_list_embeddings = (dev_list_embeddings - gl_mean) / gl_std
	
	for k in dev_dic_embeddings.keys():
		dev_dic_embeddings[k] = (dev_dic_embeddings[k] - gl_mean) / gl_std
	for k in eval_dic_embeddings.keys():
		eval_dic_embeddings[k] = (eval_dic_embeddings[k] - gl_mean) / gl_std
	
	
	#spk2utt for e2e batch composition
	dic_spk2utt = get_spk2utt_dic(dev_lines)
	
	model, m_name = get_model(argDic = parser['model'])
	save_dir = parser['save_dir'] + m_name + '_' + parser['name'] + '/'
	
	restart_epo = 0 
	
	#make folders
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	
	f_params = open(save_dir + 'f_params_bvec.txt', 'w')
	for k, v in parser.items():
		print(k, v)
		f_params.write('{}:\t{}\n'.format(k, v))
	f_params.write('DNN model params\n')
	
	for k, v in parser['model'].items():
		f_params.write('{}:\t{}\n'.format(k, v))
	print(m_name)
	f_params.write('model_name: %s\n'%m_name)
	f_params.close()
	
	
	
	nb_batch = parser['nb_batch_per_epoch']
	
	with open(save_dir + 'summary_bvec.txt' ,'w+') as f_summary:
		model.summary(print_fn=lambda x: f_summary.write(x + '\n'))
	
	if parser['optimizer'] == 'SGD':
		optimizer = eval(parser['optimizer'])(lr=parser['lr'], momentum=parser['momentum'])
	elif parser['optimizer'] == 'Adam':
		optimizer = eval(parser['optimizer'])(lr=parser['lr'], decay = parser['opt_decay'], amsgrad = bool(parser['amsgrad']))
	elif parser['optimizer'] == 'RMSprop':
		optimizer = eval(parser['optimizer'])(lr=parser['lr'], decay = parser['opt_decay'])
	
	model.compile(optimizer = optimizer,
		loss = 'sparse_categorical_crossentropy',
		metrics=['accuracy'])
	
	global q
	q = queue.Queue(maxsize=1000)
	dummy_y = np.zeros((parser['batch_size'], 1))
	if not os.path.exists(save_dir  + 'results_bvec/'):
		os.makedirs(save_dir + 'results_bvec/')
	if not os.path.exists(save_dir  + 'models_bvec/'):
		os.makedirs(save_dir + 'models_bvec/')
	f_eer = open(save_dir + 'eers_bvec.txt', 'a', buffering=1)
	for epoch in tqdm(range(restart_epo, parser['epoch'])):
		np.random.shuffle(dev_lines)
		p = Thread(target = process_epoch_e2e, args = (dev_lines,
								q,
								parser['batch_size'],
								parser['nb_batch_per_epoch'],
								dic_spk2utt,
								dev_dic_embeddings))
		p.start()
	
		#train!
		loss = 999.
		loss1 = 999.
		loss2 = 999.
		acc = 0.
		pbar = tqdm(range(nb_batch))
		for b in pbar:
			pbar.set_description('epoch: %d, loss: %.3f acc: %f'%(epoch, loss, acc))
			while True:
				if q.empty():
					sleep(0.1)
	
				else:
					x, x2, y = q.get()
					loss, acc = model.train_on_batch([x, x2], y)
					break
		p.join()
		y = []
		val_enrol = []
		val_test = []
		for smpl in val_trials:
			target, spkMd, utt = smpl.strip().split(' ')
			target = int(target)
			y.append(target)
			val_enrol.append(dev_dic_embeddings[spkMd])
			val_test.append(dev_dic_embeddings[utt])
		val_enrol = np.asarray(val_enrol)
		val_test = np.asarray(val_test)
		y_score = model.predict([val_enrol, val_test])[:, 1]
		
		fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
		eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
		print('\nepoch: %d, val_eer: %f'%(int(epoch), eer))
		f_eer.write('%d %f '%(epoch, eer))
		model.save_weights(save_dir +  'models_bvec/%d-%.4f.h5'%(epoch, eer))
	
		y = []
		eval_enrol = []
		eval_test = []
		for smpl in trials:
			target, spkMd, utt = smpl.strip().split(' ')
			target = int(target)
			y.append(target)
			eval_enrol.append(eval_dic_embeddings[spkMd])
			eval_test.append(eval_dic_embeddings[utt])
		eval_enrol = np.asarray(eval_enrol)
		eval_test = np.asarray(eval_test)
		y_score = model.predict([eval_enrol, eval_test])[:, 1]
		
		fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
		eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
		print('\nepoch: %d, eer: %f'%(int(epoch), eer))
		f_eer.write('%f\n'%(eer))
	f_eer.close()
