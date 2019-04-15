						
import os
import numpy as np
np.random.seed(615)
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
import pickle as pk
from sklearn.svm import SVC

from model_speccnn_cLoss_bsLoss import get_model
from keras import backend as K

import _pickle as pickle

def cos_sim(a,b):
	return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))


def simple_loss(y_true, y_pred):
	return K.mean(y_pred)


def zero_loss(y_true, y_pred):
	return 0.5 * K.sum(y_pred, axis=0)


def make_batch(lines, batch_size, spk_dic, GM, GS):
			
	while True:
	#nb_batch = int(len(lines) / batch_size)
	#for i in range(nb_batch):
		sample_lines = np.random.choice(lines, batch_size)
		num_frames = np.random.randint(300, 500)
		#num_frames = 500
		x = []
		y = []
		
		for line in sample_lines:						
			utt = line.strip()
						
			src = '/DB/AudioSet/scp/fbank64_key_data'
			spec = np.load(src)
		
			spec = spec.reshape(spec.shape[0], spec.shape[1])
			#spec = spec.T
			'''
			spec -= np.mean(spec, axis = 0)
			spec_std = np.std(spec, axis = 0) # + 0.0001	
			spec /= spec_std
			'''
			spec -= GM
			spec /= GS
			
			y.append(spk_dic[utt])
								
						
			while spec.shape[0] < num_frames:
				spec = np.concatenate([spec, spec])
				#print ('Here! 1 ')
				
			margin = int((spec.shape[0] - num_frames)/2)
			
			if margin == 0:
				st_idx = 0
			else:
				st_idx = np.random.randint(0, margin)
				
			ed_idx = st_idx + num_frames
			
			data = spec[st_idx:ed_idx,:]
			data = data.reshape(num_frames, 64, 1)

			x.append(data)
			
		x = np.array(x, np.float32)
		y = np.array(y, np.float32)
		
		q.put([x, y])
		#print ('Here! 2 ')
		
	print('Thread Done!!')


#======================================================================#
#======================================================================#

_abspath = os.path.abspath(__file__)
dir_yaml = os.path.splitext(_abspath)[0] + '.yaml'
with open(dir_yaml, 'r') as f_yaml:
	parser = yaml.load(f_yaml)
parser['model']['batch_size'] = int(parser['batch_size'] / parser['nb_gpu'])
tr_lines = open('/DB/AudioSet/scp/train_0_tr.scp', 'r').readlines()
tst_lines = open('/DB/AudioSet/scp/train_0_tr.scp', 'r').readlines()

fn_label = pickle.load( open('fn_label','rb')) # utt_name : answer_label
label_dic = pickle.load( open('label_dic','rb')) # answer_key : answer_label

data_x_tmp = []
for line in tr_lines:
	utt = line.strip()
						
	src = '/DB/AudioSet/scp/fbank64_key_data'
	
	spec = np.load(src)
	spec = spec.reshape(spec.shape[0], spec.shape[1])
	#spec = spec.T

	data_x_tmp.append(spec)
	

data_x = np.concatenate(data_x_tmp,axis = 0)

GM = np.mean(data_x, axis = 0)
GS = np.std(data_x, axis = 0)


data_y = []
data_x = []

label_fn_dic = {}

for line in tr_lines:
	label = fn_label[line.strip()]
	if label not in label_fn_dic:
		label_fn_dic[label] = []
	label_fn_dic[label].append(line[:-1])

min_len = 626


tr_lines = []	
for label in label_fn_dic:
	
	org_len = len(label_fn_dic[label])
	np.random.shuffle(label_fn_dic[label])
	if (min_len) < org_len:
		np.random.shuffle(label_fn_dic[label])
		label_fn_dic[label] = label_fn_dic[label][:min_len]
	elif (min_len) > org_len:
		label_fn_dic[label].extend(label_fn_dic[label][:min_len - org_len])
	elif (min_len) == org_len:
		continue
		
	for line in label_fn_dic[label]:
		tr_lines.append(line)
	print ('label %s'%(label))
	print(len(label_fn_dic[label]))
		
		
for line in tr_lines:
	data_y.append(fn_label[line.strip()])
	utt = line.strip()
						
	src = '/DB/AudioSet/scp/fbank64_key_data'
	
	spec = np.load(src)#.T
	
	spec = spec.reshape(spec.shape[0], spec.shape[1]) #(64,999)
	#spec = spec.T

	spec -= GM
	spec /= GS
	'''
	spec -= np.mean(spec, axis = 0)
	spec_std = np.std(spec, axis = 0) #+ 0.0001	
	spec /= spec_std
	'''

	data_x.append(spec.reshape(1, -1, 64,1))

data_y = np.array(data_y, np.int32)

data_y_val = []
data_x_val = []
for line in tst_lines:
	data_y_val.append(fn_label[line.strip()])
	utt = line.strip()
						
	#src ='fbank_train/' + utt[:-4] + '.npy'
	src = '/DB/AudioSet/scp/fbank64_key_data'
	spec = np.load(src)
	
	spec = spec.reshape(spec.shape[0], spec.shape[1])
	#spec = spec.T
	
	spec -= GM
	spec /= GS
	
	'''
	spec -= np.mean(spec, axis = 0)
	spec_std = np.std(spec, axis = 0) #+ 0.0001	
	#spec /= spec_std
	'''
	#data_x_val.append(spec.reshape(-1, 64,1))
	data_x_val.append(spec.reshape(1, -1, 64,1))
	
data_y_val = np.array(data_y_val, np.int32)


model, m_name, softmax_output = get_model(argDic = parser['model'])
#model_pred = Model(inputs=model.get_layer('input_1').input, outputs=model.get_layer('code').output)
model_pred = Model(inputs=model.get_layer('input_1').input, outputs=softmax_output)
save_dir = parser['save_dir'] + m_name + '_' + parser['name'] + '/'

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


with open(save_dir + 'summary.txt' ,'w+') as f_summary:
	model.summary(print_fn=lambda x: f_summary.write(x + '\n'))



if parser['optimizer'] == 'Adam':
	optimizer = eval(parser['optimizer'])(lr=parser['lr'], decay = parser['opt_decay'], amsgrad = bool(parser['amsgrad']))
elif parser['optimizer'] == 'SGD':
	optimizer = eval(parser['optimizer'])(lr=parser['lr'], momentum=parser['momentum'],  nesterov=True)
elif parser['optimizer'] == 'RMSprop':
	optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
else:
	print('Optimizer not defined!')
	exit()


if bool(parser['mg']):
	model = multi_gpu_model(model, gpus=parser['nb_gpu'])
	
model.compile(optimizer = optimizer,
		loss = {'s_bs_loss':simple_loss,
				'c_loss':zero_loss},
		loss_weights = {'s_bs_loss':1, 'c_loss': parser['c_lambda']},
	metrics=['accuracy'])

global q
q = queue.Queue(maxsize=1000)

if not os.path.exists(save_dir  + 'results/'):
	os.makedirs(save_dir + 'results/')

#model.load_weights('model/networks/ss_tmp/31-0.02244.h5')
p1 = Thread(target = make_batch, args = (tr_lines,
											parser['batch_size'],
											fn_label, GM, GS))
p1.start()
	
#dummy_y = np.zeros((data_x.shape[0], 1))
#dummy_y_val = np.zeros((data_x_val.shape[0], 1))
dummy_y = np.zeros((parser['batch_size'], 1))
#dummy_y_val = np.zeros((data_x_val.shape[0], 1))

for epoch in tqdm(range(parser['epoch'])):
	
	nb_batch = int(len(tr_lines) / parser['batch_size'])
	print (nb_batch)
	for i in range(nb_batch):
		while True:
			if q.empty():
				#print('sleeping...')
				sleep(0.1)

			else:
				x, y = q.get()
				y_cat = to_categorical(y, num_classes=7)
				loss,s_bs_loss, c_loss,_,_  = model.train_on_batch([x, y_cat], [dummy_y, dummy_y])
				print(loss)
				break
	
	num_corr = 0.	
	num_corr_class = {}
	num_class = {}
	acc_class = {}

	for i in range(len(data_x_val)):
		predicted = model_pred.predict(data_x_val[i])[0]
		answer_label = data_y_val[i] 
		print ('answer: %d'%(answer_label))
		print ('predicted: %d'%(np.argmax(predicted)))
		if answer_label not in num_class.keys():
			num_class[answer_label] = 0
		if answer_label not in acc_class.keys():
			acc_class[answer_label] = 0

		num_class[answer_label] += 1
		
		if answer_label == np.argmax(predicted):
			if answer_label not in num_corr_class.keys():
				num_corr_class[answer_label] = 0
			num_corr += 1.
			num_corr_class[answer_label] += 1
			acc_class[answer_label] += 1
	acc = num_corr/len(data_x_val)

	print(acc)
	print ('num_class: ')
	print (num_class)
	print ('acc_class: ')
	print (acc_class)
	model.save_weights(save_dir +  '%d-%.5f.h5'%(epoch, acc))
	continue
	
'''	
	embedding_trn = []
	for i in range(len(data_x)):
		embedding = model_pred.predict(data_x[i])[0]
		embedding_trn.append(embedding)
		
	embedding_trn = np.array(embedding_trn, np.float32)
	
	embedding_val = []
	for i in range(len(data_x_val)):
		embedding = model_pred.predict(data_x_val[i])[0]
		embedding_val.append(embedding)
	embedding_val = np.array(embedding_val, np.float32)

	acc = []
	classwise_acc = []
	SVM_list = []
	for cov_type in ['rbf', 'sigmoid']:
		score_list = []

		SVM_list.append(SVC( kernel=cov_type, probability=True))
		SVM_list[-1].fit(embedding_trn, data_y)

		num_corr = 0
		

		score_list = SVM_list[-1].predict(embedding_val)
		
		for i in range(embedding_val.shape[0]):
			
			if score_list[i] == data_y_val[i]:
				num_corr += 1
				
		acc.append(float(num_corr)/ embedding_val.shape[0])
			
	#save model
	model.save_weights(save_dir +  '/results/%d-%.5f-%.5f.h5'%(epoch, acc[0], acc[1]))
	#pk.dump((SVM_list[0], classwise_acc[0]), open(save_dir + '%d-rbf.pk'%(epoch), 'wb'))
	#pk.dump((SVM_list[1], classwise_acc[1]), open(save_dir + '%d-sig.pk'%(epoch), 'wb'))
'''		









