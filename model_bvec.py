import numpy as np
import keras
import tensorflow as tf
from keras import regularizers, optimizers, utils, models, initializers, constraints
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, Dense, Activation, Input, Add, Dropout, LeakyReLU, GRU, Subtract, Multiply, Concatenate
import keras.backend as K
from keras.models import Model
from keras.engine.topology import Layer
from keras.activations import softmax


import os
_abspath = os.path.abspath(__file__)
m_name = _abspath.split('/')[-1].split('.')[0][6:]


def simple_loss(y_true, y_pred):
	return K.mean(y_pred)

def zero_loss(y_true, y_pred):
	return 0.5 * K.sum(y_pred, axis=0)

class spk_basis_loss(Dense):
	def __init__(self,
				units,
				s = 5., 
				kernel_initializer='glorot_uniform',
				kernel_regularizer=None,
				kernel_constraint=None,
				**kwargs):
		if 'input_shape' not in kwargs and 'input_dim' in kwargs:
			kwargs['input_shape'] = (kwargs.pop('input_dim'),)
		super(Dense, self).__init__(**kwargs)
		self.units = units
		self.s = s
		
		self.kernel_initializer = initializers.get(kernel_initializer)
		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.kernel_constraint = constraints.get(kernel_constraint)
		

	def build(self, input_shape):
		assert len(input_shape[0]) >= 2
		input_dim = input_shape[0][-1]

		self.kernel = self.add_weight(shape=(input_dim, self.units),
									  initializer=self.kernel_initializer,
									  name='kernel',
									  regularizer=self.kernel_regularizer,
									  constraint=self.kernel_constraint)
		self.bias = None
		self.built = True


	def call(self, inputs):
		inputs_x = inputs[0]
		inputs_y = inputs[1]
		
		input_length = K.sum(inputs_x**2., axis = 1, keepdims = True)**0.5
		input_length /= self.s ** 0.5
		input_length += 0.0001
		
		kernel_length = K.sum(self.kernel**2., axis = 0, keepdims = True)**0.5
		kernel_length /= self.s ** 0.5
		kernel_length += 0.0001
		
		inputs_norm = inputs_x / input_length
		kernel_norm = self.kernel / kernel_length
		
		label_onehot = inputs_y
		
		negative_mask = tf.fill([self.units, self.units], 1.) - tf.eye(self.units)
		
		loss_BS = K.mean(tf.matmul(kernel_norm, kernel_norm,
                     adjoint_a = True # transpose second matrix
                     ) * negative_mask  ) 
					 
			
		inner_output = K.dot(inputs_x, self.kernel)
		softmax_output = softmax(inner_output)
		loss_s = K.categorical_crossentropy(inputs_y, softmax_output)
		
		final_loss = loss_s + loss_BS
		
		
		return final_loss
		
		
	def compute_output_shape(self, input_shape):
		return (input_shape[0][0], 1)


class CenterLossLayer(Layer):

	def __init__(self, alpha, nb_center, dim_embd, **kwargs):
		super().__init__(**kwargs)
		self.alpha = alpha
		self.nb_center = nb_center
		self.dim_embd = dim_embd

	def build(self, input_shape):
		self.centers = self.add_weight(name='centers',
				   shape=(self.nb_center, self.dim_embd),
				   initializer='uniform',
				   trainable=False)
		super().build(input_shape)

	def call(self, x, mask=None):

		delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
		center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
		delta_centers /= center_counts
		new_centers = self.centers - self.alpha * delta_centers
		self.add_update((self.centers, new_centers), x)


		self.result = x[0] - K.dot(x[1], self.centers)
		self.result = K.sum(self.result ** 2, axis=1, keepdims=True) #/ K.dot(x[1], center_counts)
		return self.result # Nx1

	def compute_output_shape(self, input_shape):
		return K.int_shape(self.result)


def get_model(argDic):
	input_enrol = Input(shape = (argDic['dim_embedding'], ))
	enrol = Dropout(0.2)(input_enrol)
	input_test = Input(shape = (argDic['dim_embedding'], ))
	test = Dropout(0.2)(input_test)

	add_et = Add()([enrol, test])
	sub_et = Subtract()([enrol, test])
	mul_et = Multiply()([enrol, test])
	x = Concatenate(axis = -1)([add_et, sub_et, mul_et])

	for i in range(len(argDic['nb_dense_node'])):
		x = Dense(argDic['nb_dense_node'][i],
			kernel_initializer = argDic['initializer'],
			kernel_regularizer = regularizers.l2(argDic['wd']),
			name = 'e2e_dense_%d'%i)(x)
		x = Dropout(argDic['drop_ratio'], name='e2e_Dropout_%d'%i)(x)
		x = LeakyReLU(name = 'e2e_LRelu_%d'%i)(x)

	x = Dense(2,
		kernel_initializer = argDic['initializer'],
		kernel_regularizer = regularizers.l2(argDic['wd']),
		activation = 'softmax',
		name = 'e2e_out')(x)


	return [Model(inputs=[input_enrol, input_test], output = x), m_name]
 
