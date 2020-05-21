import os
import sys

sys.path.insert(0, './')

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D

from .pointnet_util import PointNetConv, PointNetMSGConv


class Pointnet2Encoder(Model):

	def __init__(self, batch_size, n_pred=3, activation=tf.nn.relu, msg=False, net_type='loss'):
		super(Pointnet2Encoder, self).__init__()

		self.activation = activation
		self.batch_size = batch_size
		self.n_pred = n_pred
		self.kernel_initializer = 'glorot_normal'
		self.kernel_regularizer = None
		self.msg = msg
		self.net_type = net_type

		if net_type == 'scene':
			self.out_filters = 1024 - (1024 % (self.n_pred * 3))
		elif net_type == 'loss':
			self.out_filters = 1024
		else:
			raise ValueError('Net type not recognised.')

		self.init_network_msg() if msg == True else self.init_network()


	def init_network(self):

		l1_npoint = 1024

		self.pn_conv1 = PointNetConv(
			npoint=l1_npoint,
			radius=0.1,
			nsample=32,
			mlp=[64, 64, 128],
			group_all=False,
			activation=self.activation
		)

		l2_npoint = 256

		self.pn_conv2 = PointNetConv(
			npoint=l2_npoint,
			radius=0.2,
			nsample=32,
			mlp=[64, 64, 128],
			group_all=False,
			activation=self.activation
		)

		l3_npoint = 64 

		self.pn_conv3 = PointNetConv(
			npoint=l3_npoint,
			radius=0.4,
			nsample=64,
			mlp=[128, 128, 256],
			group_all=False,
			activation=self.activation
		)

		self.pn_conv4 = PointNetConv(
			npoint=None,
			radius=None,
			nsample=None,
			mlp=[256, 512, self.out_filters],
			group_all=True,
			activation=self.activation
		)


	def init_network_msg(self):

		self.pn_conv1 = PointNetMSGConv(
			npoint=1024,
			radius_list=[0.1,0.2,0.4],
			nsample_list=[16,32,128],
			mlp=[[32,32,64], [64,64,128], [64,96,128]],
			activation=self.activation
		)

		self.pn_conv2 = PointNetMSGConv(
			npoint=512,
			radius_list=[0.2,0.4,0.8],
			nsample_list=[32,64,128],
			mlp=[[64,64,128], [128,128,256], [128,128,256]],
			activation=self.activation
		)

		self.pn_conv3 = PointNetConv(
			npoint=None,
			radius=None,
			nsample=None,
			mlp=[256, 512, self.out_filters],
			group_all=True,
			activation=self.activation
		)


	def call(self, input):
		xyz, points = self.pn_conv1(input, None)
		xyz, points = self.pn_conv2(xyz, points)
		xyz, points = self.pn_conv3(xyz, points)

		if self.msg == False:
			xyz, points = self.pn_conv4(xyz, points)

		encoded = tf.reshape(points, (self.batch_size, -1))

		return encoded
