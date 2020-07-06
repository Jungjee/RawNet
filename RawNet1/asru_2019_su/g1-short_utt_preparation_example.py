import torch
import numpy as np

from torch.utils import data

class Dataset_VoxCeleb1_su(data.Dataset):
	def __init__(self, list_IDs, nb_time, base_dir, cut = True, pre_emp = True):
		'''
		PyTorch Dataset class for ASRU 2019 paper. 

		==========
		list_IDs	: list of utterance directories
		nb_time		: number of samples for each utterance (16,000 * duration)
					  e.g. 32805 for the setting of the paper
		pre_emp		: Flag for conducting pre-emphasis
		cut			: Flag to decide whether to adjust durations
					  (set False to conduct full length decoding)
		'''
		self.list_IDs = list_IDs
		self.nb_time = nb_time
		self.pre_emp = pre_emp
		self.cut = cut

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		ID = self.list_IDs[index]
		X = np.load(self.base_dir+ID+'.npy')
		if self.pre_emp: X = self.pre_emphasis(X)

		if self.cut:
			nb_time = X.shape[1]
			if nb_time > self.nb_time:
				X = X[:, int(nb_time/2) - int(self.nb_time/2) : int(nb_time/2) + int(self.nb_time/2)]
		
		else:
			return X

		return X

	def pre_emphasis(self, x):
		return np.asarray(x[:,1:] - 0.97 * x[:, :-1], dtype=np.float32) 
