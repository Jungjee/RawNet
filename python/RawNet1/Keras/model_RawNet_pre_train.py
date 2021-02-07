import numpy as np
import keras
import tensorflow as tf
from keras import regularizers, optimizers, utils, models, initializers, constraints
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, Dense, Activation, Input, Add, Dropout, LeakyReLU
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
	def __init__(self, units,
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
		# shape = [#spk, #spk]

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

def residual_block_conv(input_tensor, filters = [], initializer = None, regularizer = None, base_name = None):

	x = Conv1D(filters[0], 3, strides = 1, activation = None,
			   kernel_initializer = initializer, kernel_regularizer = regularizer,
			   padding = 'same', name = base_name+'_Conv1')(input_tensor)
	x = BatchNormalization(name=base_name+'_BN1')(x)
	x = LeakyReLU(name=base_name+'_Act1')(x)

	x = Conv1D(filters[1], 3, strides = 1, activation = None,
			   kernel_initializer = initializer, kernel_regularizer = regularizer,
			   padding = 'same', name = base_name+'_Conv2')(x)
	x = BatchNormalization(name=base_name+'_BN2')(x)

	#in this case: set filter lenth to 1
	if K.int_shape(input_tensor)[-1] != K.int_shape(x)[-1]:
		input_tensor = Conv1D(filters[1], 1, strides=1, activation = None,
			   kernel_initializer = initializer, kernel_regularizer = regularizer,
			   padding = 'same', name = base_name+'_transform')(input_tensor)
		input_tensor = BatchNormalization(name=base_name+'_BN_transform')(input_tensor)
	x = Add()([input_tensor, x])
	x = LeakyReLU(name=base_name+'_Act2')(x)

	return x



def get_model(argDic):
	inputs = Input(shape = (None, 1), name='input_pretrn')
	c_input = Input(shape = (argDic['nb_spk'],))

	#strided Conv
	x = Conv1D(argDic['nb_s_conv_filt'], 3, strides=3,
			activation = None,
			kernel_initializer = argDic['initializer'],
			kernel_regularizer = regularizers.l2(argDic['wd']),
			padding = 'valid',
			name = 'strided_conv')(inputs)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	for i in range(1, 3):
		x = residual_block_conv(x, argDic['nb_conv_filt'][0],
			initializer = argDic['initializer'],
			regularizer = regularizers.l2(argDic['wd']),
			base_name = 'res_conv_block_%d'%i)
		x = MaxPooling1D(pool_size=3)(x)

	for i in range(3, 7):
		x = residual_block_conv(x, argDic['nb_conv_filt'][1],
			initializer = argDic['initializer'],
			regularizer = regularizers.l2(argDic['wd']),
			base_name = 'res_conv_block_%d'%i)
		x = MaxPooling1D(pool_size=3)(x)

	x = residual_block_conv(x, argDic['nb_conv_filt'][2],
		initializer = argDic['initializer'],
		regularizer = regularizers.l2(argDic['wd']),
		base_name = 'res_conv_block_9')
	x = MaxPooling1D(pool_size=3)(x)

	x = Conv1D(argDic['nb_conv_filt'][3], 1, strides=1,
			activation = None,
			kernel_initializer = argDic['initializer'],
			kernel_regularizer = regularizers.l2(argDic['wd']),
			padding = 'same',
			name = 'last_conv')(x)
	x = BatchNormalization(axis=-1)(x)
	x = LeakyReLU()(x)

	x = Dropout(0.5)(x)
	x = GlobalAveragePooling1D()(x)

	for i in range(len(argDic['nb_dense_node'])):
		if i == len(argDic['nb_dense_node']) -1:
			name = 'code_pretrn'
		else:
			name = 'dense_act_%d'%(i+1)
		x = Dense(argDic['nb_dense_node'][i],
			kernel_initializer = argDic['initializer'],
			kernel_regularizer = regularizers.l2(argDic['wd']))(x)
		x = BatchNormalization(axis=-1)(x)
		x = LeakyReLU(name = name)(x)

	s_bs_out = spk_basis_loss(units = argDic['nb_spk'],
			kernel_initializer = argDic['initializer'],
			kernel_regularizer = regularizers.l2(argDic['wd']),
			name = 's_bs_loss')([x, c_input])

	c_out = CenterLossLayer(alpha = argDic['c_alpha'],
			nb_center = argDic['nb_spk'],
			dim_embd = argDic['nb_dense_node'][-1],
			name='c_loss')([x, c_input])

	return [Model(inputs=[inputs, c_input], output=[s_bs_out, c_out]), m_name]