#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#	http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr

class MobileNetV2():
	def __init__(self, img, num_classes, img_shape,scale=1.0, change_depth=False):
		self.scale = scale
		self.change_depth=change_depth
		self.img = img
		self.num_classes = num_classes
		self.img_shape = img_shape

	def net(self):
		scale = self.scale
		change_depth = self.change_depth
		#if change_depth is True, the new depth is 1.4 times as deep as before.
		bottleneck_params_list = [
			(1, 16, 1, 1),
			(6, 24, 2, 2),
			(6, 32, 3, 2),
			(6, 64, 4, 2),
			(6, 96, 3, 1),
			(6, 160, 3, 2),
			(6, 320, 1, 1),
		] if change_depth == False else [
			(1, 16, 1, 1), 
			(6, 24, 2, 2), 
			(6, 32, 5, 2), 
			(6, 64, 7, 2), 
			(6, 96, 5, 1), 
			(6, 160, 3, 2), 
			(6, 320, 1, 1), 
		]
		
		feature_maps = []
		print("self.img.shape = ",self.img.shape)
		#conv1 
		input = self.conv_bn_layer(
			self.img,
			num_filters=int(32 * scale),
			filter_size=3,
			stride=2,
			padding=1,
			if_act=True,
			name='conv1_1')
		print("conv1.shape = ",input.shape)
		# bottleneck sequences
		i = 1
		in_c = int(32 * scale)
		for layer_setting in bottleneck_params_list:
			t, c, n, s = layer_setting
			i += 1
			input = self.invresi_blocks(
				input=input,
				in_c=in_c,
				t=t,
				c=int(c * scale),
				n=n,
				s=s,
				name='conv' + str(i))
			in_c = int(c * scale)
			print('conv' + str(i)+".shape = ",input.shape)
			if i == 4 or i == 6 :
				feature_maps.append(input)
		
		#last_conv
		input = self.conv_bn_layer(
			input=input,
			num_filters=int(1280 * scale) if scale > 1.0 else 1280,
			filter_size=1,
			stride=1,
			padding=0,
			if_act=True,
			name='conv9')
		print("last_conv.shape = ",input.shape)
		
		feature_maps.append(input)
		
		i += 2
		input = self.inverted_residual_unit(input,
											num_in_filter=1280,
											num_filters=512,
											ifshortcut=False,
											stride=2,
											filter_size=1,
											padding=0,
											expansion_factor=0.2,
											name='conv' + str(i))
		print("module.shape = ",input.shape)
		feature_maps.append(input)
		i += 1
		input = self.inverted_residual_unit(input,
											num_in_filter=512,
											num_filters=256,
											ifshortcut=False,
											stride=2,
											filter_size=1,
											padding=0,
											expansion_factor=0.25,
											name='conv' + str(i))
		print("module.shape = ",input.shape)
		feature_maps.append(input)
		i += 1
		input = self.inverted_residual_unit(input,
											num_in_filter=256,
											num_filters=256,
											ifshortcut=False,
											stride=2,
											filter_size=1,
											padding=0,
											expansion_factor=0.5,
											name='conv' + str(i))
		print("module.shape = ",input.shape)
		#feature_maps.append(input)
		i += 1
		input = self.inverted_residual_unit(input,
											num_in_filter=256,
											num_filters=64,
											ifshortcut=False,
											stride=2,
											filter_size=1,
											padding=0,
											expansion_factor=0.25,
											name='conv' + str(i))
											
		print("module.shape = ",input.shape)
		feature_maps.append(input)

		mbox_locs, mbox_confs, box, box_var = fluid.layers.multi_box_head(
			inputs=feature_maps,
			image=self.img,
			num_classes=self.num_classes,
			min_ratio=15,
			max_ratio=90,
			min_sizes=[30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
			max_sizes=[60.0, 111.0, 162.0, 213.0, 264.0, 315.0],
			aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.,],[2.]],
			steps=[8,16,32,64,100,300],
			base_size=self.img_shape[2],
			offset=0.5,
			kernel_size=3,
			pad=1,
			flip=True)

		return mbox_locs, mbox_confs, box, box_var
		
	def conv_bn_layer(self,
					  input,
					  filter_size,
					  num_filters,
					  stride,
					  padding,
					  channels=None,
					  num_groups=1,
					  if_act=True,
					  name=None,
					  use_cudnn=True):
		conv = fluid.layers.conv2d(
			input=input,
			num_filters=num_filters,
			filter_size=filter_size,
			stride=stride,
			padding=padding,
			groups=num_groups,
			act=None,
			use_cudnn=use_cudnn,
			param_attr=ParamAttr(name=name + '_weights'),
			bias_attr=False)
		bn_name = name + '_bn'
		bn = fluid.layers.batch_norm(
			input=conv,
			param_attr=ParamAttr(name=bn_name + "_scale"),
			bias_attr=ParamAttr(name=bn_name + "_offset"),
			moving_mean_name=bn_name + '_mean',
			moving_variance_name=bn_name + '_variance')
		if if_act:
			return fluid.layers.relu6(bn)
		else:
			return bn

	def shortcut(self, input, data_residual):
		return fluid.layers.elementwise_add(input, data_residual)

	def inverted_residual_unit(self,
							   input,
							   num_in_filter,
							   num_filters,
							   ifshortcut,
							   stride,
							   filter_size,
							   padding,
							   expansion_factor,
							   name=None):
		num_expfilter = int(round(num_in_filter * expansion_factor))

		channel_expand = self.conv_bn_layer(
			input=input,
			num_filters=num_expfilter,
			filter_size=1,
			stride=1,
			padding=0,
			num_groups=1,
			if_act=True,
			name=name + '_expand')

		bottleneck_conv = self.conv_bn_layer(
			input=channel_expand,
			num_filters=num_expfilter,
			filter_size=filter_size,
			stride=stride,
			padding=padding,
			num_groups=num_expfilter,
			if_act=True,
			name=name + '_dwise',
			use_cudnn=False)

		linear_out = self.conv_bn_layer(
			input=bottleneck_conv,
			num_filters=num_filters,
			filter_size=1,
			stride=1,
			padding=0,
			num_groups=1,
			if_act=False,
			name=name + '_linear')
		if ifshortcut:
			out = self.shortcut(input=input, data_residual=linear_out)
			return out
		else:
			return linear_out

	def invresi_blocks(self, input, in_c, t, c, n, s, name=None):
		first_block = self.inverted_residual_unit(
			input=input,
			num_in_filter=in_c,
			num_filters=c,
			ifshortcut=False,
			stride=s,
			filter_size=3,
			padding=1,
			expansion_factor=t,
			name=name + '_1')

		last_residual_block = first_block
		last_c = c

		for i in range(1, n):
			last_residual_block = self.inverted_residual_unit(
				input=last_residual_block,
				num_in_filter=last_c,
				num_filters=c,
				ifshortcut=True,
				stride=1,
				filter_size=3,
				padding=1,
				expansion_factor=t,
				name=name + '_' + str(i + 1))
		return last_residual_block
	
	
	
def MobileNetV2_x0_25(img, num_classes, img_shape,):
	model = MobileNetV2(img, num_classes, img_shape,scale=0.25)
	return model

def MobileNetV2_x0_5(img, num_classes, img_shape):
	model = MobileNetV2(img, num_classes, img_shape,scale=0.5)
	return model

def MobileNetV2_x1_0(img, num_classes, img_shape):
	model = MobileNetV2(img, num_classes, img_shape,scale=1.0)
	return model

def MobileNetV2_x1_5(img, num_classes, img_shape):
	model = MobileNetV2(img, num_classes, img_shape,scale=1.5)
	return model

def MobileNetV2_x2_0(img, num_classes, img_shape):
	model = MobileNetV2(img, num_classes, img_shape,scale=2.0)
	return model

def MobileNetV2_scale(img, num_classes, img_shape):
	model = MobileNetV2(img, num_classes, img_shape,scale=1.2, change_depth=True)
	return model
	
def build_mobilenet_ssd(img, num_classes, img_shape):
	ssd_model = MobileNetV2_x1_0(img, num_classes, img_shape)
	return ssd_model.net()
