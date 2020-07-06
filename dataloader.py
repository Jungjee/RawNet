import numpy as np
import soundfile as sf

from torch.utils import data

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