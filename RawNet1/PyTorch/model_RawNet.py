import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter


class Residual_block(nn.Module):
	def __init__(self, nb_filts, first = False):
		super(Residual_block, self).__init__()
		self.first = first
		
		if not self.first:
			self.bn1 = nn.BatchNorm1d(num_features = nb_filts[0])
		self.lrelu = nn.LeakyReLU()
		self.lrelu_keras = nn.LeakyReLU(negative_slope=0.3)

		self.conv1 = nn.Conv1d(in_channels = nb_filts[0],
			out_channels = nb_filts[1],
			kernel_size = 3,
			padding = 1,
			stride = 1)
		self.bn2 = nn.BatchNorm1d(num_features = nb_filts[1])
		self.conv2 = nn.Conv1d(in_channels = nb_filts[1],
			out_channels = nb_filts[1],
			padding = 1,
			kernel_size = 3,
			stride = 1)

		if nb_filts[0] != nb_filts[1]:
			self.downsample = True
			self.conv_downsample = nn.Conv1d(in_channels = nb_filts[0],
				out_channels = nb_filts[1],
				padding = 0,
				kernel_size = 1,
				stride = 1)
		else:
			self.downsample = False
		self.mp = nn.MaxPool1d(3)

	def forward(self, x):
		identity = x
		if not self.first:
			out = self.bn1(x)
			out = self.lrelu_keras(out)
		else:
			out = x

		out = self.conv1(x)
		out = self.bn2(out)
		out = self.lrelu_keras(out)
		out = self.conv2(out)

		if self.downsample:
			identity = self.conv_downsample(identity)
		
		out += identity
		out = self.mp(out)
		return out

class RawNet(nn.Module):
	def __init__(self, d_args, device):
		super(RawNet, self).__init__()
		#self.negative_k = d_args['negative_k']
		self.first_conv = nn.Conv1d(in_channels = d_args['in_channels'],
			out_channels = d_args['filts'][0],#128
			kernel_size = d_args['first_conv'],#3
			padding = 0,
			stride = d_args['first_conv'])
		self.first_bn = nn.BatchNorm1d(num_features = d_args['filts'][0])
		self.lrelu = nn.LeakyReLU()
		self.lrelu_keras = nn.LeakyReLU(negative_slope = 0.3)

		self.block0 = self._make_layer(nb_blocks = d_args['blocks'][0],
			nb_filts = d_args['filts'][1],
			first = True)
		self.block1 = self._make_layer(nb_blocks = d_args['blocks'][1],
			nb_filts = d_args['filts'][2])

		self.bn_before_gru = nn.BatchNorm1d(num_features = d_args['filts'][2][-1])
		self.gru = nn.GRU(input_size = d_args['filts'][2][-1],
			hidden_size = d_args['gru_node'],
			num_layers = d_args['nb_gru_layer'],
			batch_first = True)
		self.fc1_gru = nn.Linear(in_features = d_args['gru_node'],
			out_features = d_args['nb_fc_node'])
		self.fc2_gru = nn.Linear(in_features = d_args['nb_fc_node'],
			out_features = d_args['nb_classes'],
			bias = True)

	def forward(self, x, y = 0, is_test=False):
		x = self.first_conv(x)
		x = self.first_bn(x)
		x = self.lrelu_keras(x)

		x = self.block0(x)
		x = self.block1(x)

		x = self.bn_before_gru(x)
		x = self.lrelu_keras(x)
		x = x.permute(0, 2, 1)#(batch, filt, time) >> (batch, time, filt)
		x, _ = self.gru(x)
		x = x[:,-1,:]
		code = self.fc1_gru(x)
		if is_test: return code

		code_norm = code.norm(p=2,dim=1, keepdim=True) / 10.
		code = torch.div(code, code_norm)
		out = self.fc2_gru(code)
		return out
		'''
		#for future updates, bc_loss, h_loss
		#h_loss
		norm = torch.norm(self.fc2_gru.weight, dim = 1, keepdim = True)
		normed_weight = torch.div(self.fc2_gru.weight, norm)
		cos_output_tmp = torch.mm(code, normed_weight.t())
		cos_impo = cos_output_tmp.gather(1, y2)
		cos_target = cos_output_tmp.gather(1, y.view(-1, 1))
		hard_negatives, _ = torch.topk(cos_impo, self.negative_k, dim=1, sorted=False)
		hard_negatives = F.relu(hard_negatives, inplace=True)
		trg_score = cos_target*-1.
		h_loss = torch.log(1.+torch.exp(hard_negatives+trg_score).sum(dim=1))
		return out, h_loss
		'''

	def _make_layer(self, nb_blocks, nb_filts, first = False):
		layers = []
		#def __init__(self, nb_filts, first = False):
		for i in range(nb_blocks):
			first = first if i == 0 else False
			layers.append(Residual_block(nb_filts = nb_filts,
				first = first))
			if i == 0: nb_filts[0] = nb_filts[1]

		return nn.Sequential(*layers)

	def summary(self, input_size, batch_size=-1, device="cuda", print_fn = None):
		if print_fn == None: printfn = print
		model = self
	
		def register_hook(module):
	
			def hook(module, input, output):
				class_name = str(module.__class__).split(".")[-1].split("'")[0]
				module_idx = len(summary)
	
				m_key = "%s-%i" % (class_name, module_idx + 1)
				summary[m_key] = OrderedDict()
				summary[m_key]["input_shape"] = list(input[0].size())
				summary[m_key]["input_shape"][0] = batch_size
				if isinstance(output, (list, tuple)):
					summary[m_key]["output_shape"] = [
						[-1] + list(o.size())[1:] for o in output
					]
				else:
					summary[m_key]["output_shape"] = list(output.size())
					if len(summary[m_key]["output_shape"]) != 0:
						summary[m_key]["output_shape"][0] = batch_size
	
				params = 0
				if hasattr(module, "weight") and hasattr(module.weight, "size"):
					params += torch.prod(torch.LongTensor(list(module.weight.size())))
					summary[m_key]["trainable"] = module.weight.requires_grad
				if hasattr(module, "bias") and hasattr(module.bias, "size"):
					params += torch.prod(torch.LongTensor(list(module.bias.size())))
				summary[m_key]["nb_params"] = params
	
			if (
				not isinstance(module, nn.Sequential)
				and not isinstance(module, nn.ModuleList)
				and not (module == model)
			):
				hooks.append(module.register_forward_hook(hook))
	
		device = device.lower()
		assert device in [
			"cuda",
			"cpu",
		], "Input device is not valid, please specify 'cuda' or 'cpu'"
	
		if device == "cuda" and torch.cuda.is_available():
			dtype = torch.cuda.FloatTensor
		else:
			dtype = torch.FloatTensor
		if isinstance(input_size, tuple):
			input_size = [input_size]
		x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
		summary = OrderedDict()
		hooks = []
		model.apply(register_hook)
		model(*x)
		for h in hooks:
			h.remove()
	
		print_fn("----------------------------------------------------------------")
		line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
		print_fn(line_new)
		print_fn("================================================================")
		total_params = 0
		total_output = 0
		trainable_params = 0
		for layer in summary:
			# input_shape, output_shape, trainable, nb_params
			line_new = "{:>20}  {:>25} {:>15}".format(
				layer,
				str(summary[layer]["output_shape"]),
				"{0:,}".format(summary[layer]["nb_params"]),
			)
			total_params += summary[layer]["nb_params"]
			total_output += np.prod(summary[layer]["output_shape"])
			if "trainable" in summary[layer]:
				if summary[layer]["trainable"] == True:
					trainable_params += summary[layer]["nb_params"]
			print_fn(line_new)
	
		# assume 4 bytes/number (float on cuda).
		total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
		total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
		total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
		total_size = total_params_size + total_output_size + total_input_size
	
		print_fn("================================================================")
		print_fn("Total params: {0:,}".format(total_params))
		print_fn("Trainable params: {0:,}".format(trainable_params))
		print_fn("Non-trainable params: {0:,}".format(total_params - trainable_params))
		print_fn("----------------------------------------------------------------")
		print_fn("Input size (MB): %0.2f" % total_input_size)
		print_fn("Forward/backward pass size (MB): %0.2f" % total_output_size)
		print_fn("Params size (MB): %0.2f" % total_params_size)
		print_fn("Estimated Total Size (MB): %0.2f" % total_size)
		print_fn("----------------------------------------------------------------")
		return