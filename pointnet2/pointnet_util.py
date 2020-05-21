import os
import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import MaxPool1D, Layer

from .cpp_modules import (
	farthest_point_sample,
	gather_point,
	query_ball_point,
	group_point,
	knn_point,
	three_nn,
	three_interpolate
)


def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):

	new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
	if knn:
		_,idx = knn_point(nsample, xyz, new_xyz)
	else:
		idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
	grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
	grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
	if points is not None:
		grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
		if use_xyz:
			new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
		else:
			new_points = grouped_points
	else:
		new_points = grouped_xyz

	return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):

	batch_size = xyz.get_shape()[0]
	nsample = xyz.get_shape()[1]

	new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)

	idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
	grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
	if points is not None:
		if use_xyz:
			new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
		else:
			new_points = points
		new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
	else:
		new_points = grouped_xyz
	return new_xyz, new_points, idx, grouped_xyz


class Conv2d(Layer):

	def __init__(self, filters, strides=[1, 1], activation=tf.nn.relu, padding='VALID', initializer='glorot_normal'):
		super(Conv2d, self).__init__()

		self.filters = filters
		self.strides = strides
		self.activation = activation
		self.padding = padding
		self.initializer = initializer

	def build(self, input_shape):

		self.w = self.add_weight(
			shape=(1, 1, input_shape[-1], self.filters),
			initializer=self.initializer,
			trainable=True,
			name='pnet_conv'
		)

		super(Conv2d, self).build(input_shape)

	def call(self, inputs):

		points = tf.nn.conv2d(inputs, filters=self.w, strides=self.strides, padding=self.padding)

		if self.activation: points = self.activation(points)

		return points


class PointNetConv(Layer):

	def __init__(
		self, npoint, radius, nsample, mlp, group_all=False, knn=False, use_xyz=True, activation=tf.nn.relu
	):

		super(PointNetConv, self).__init__()

		self.npoint = npoint
		self.radius = radius
		self.nsample = nsample
		self.mlp = mlp
		self.group_all = group_all
		self.knn = False
		self.use_xyz = use_xyz
		self.activation = activation

		self.mlp_list = []

		self.conv1 = Conv2d(filters=256)

	def build(self, input_shape):

		for i, n_filters in enumerate(self.mlp):
			self.mlp_list.append(Conv2d(n_filters, activation=self.activation))
		super(PointNetConv, self).build(input_shape)

	def call(self, xyz, points):

		if points is not None:
			if len(points.shape) < 3:
				points = tf.expand_dims(points, axis=0)

		if self.group_all:
			nsample = xyz.get_shape()[1]
			new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
		else:
			new_xyz, new_points, idx, grouped_xyz = sample_and_group(
				self.npoint,
				self.radius,
				self.nsample,
				xyz,
				points,
				self.knn,
				use_xyz=self.use_xyz
			)

		for i, mlp_layer in enumerate(self.mlp_list):
			new_points = mlp_layer(new_points)

		new_points = tf.math.reduce_max(new_points, axis=2, keepdims=True)

		return new_xyz, tf.squeeze(new_points)


class PointNetMSGConv(Layer):

	def __init__(
		self, npoint, radius_list, nsample_list, mlp, use_xyz=True, activation=tf.nn.relu
	):

		super(PointNetMSGConv, self).__init__()

		self.npoint = npoint
		self.radius_list = radius_list
		self.nsample_list = nsample_list
		self.mlp = mlp
		self.use_xyz = use_xyz
		self.activation = activation

		self.mlp_list = []

	def build(self, input_shape):

		for i in range(len(self.radius_list)):
			tmp_list = []
			for i, n_filters in enumerate(self.mlp[i]):
				tmp_list.append(Conv2d(n_filters, activation=self.activation))
			self.mlp_list.append(tmp_list)

		super(PointNetMSGConv, self).build(input_shape)

	def call(self, xyz, points):

		new_xyz = gather_point(xyz, farthest_point_sample(self.npoint, xyz))

		new_points_list = []

		for i in range(len(self.radius_list)):
			radius = self.radius_list[i]
			nsample = self.nsample_list[i]
			idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
			grouped_xyz = group_point(xyz, idx)
			grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])

			if points is not None:
				grouped_points = group_point(points, idx)
				if self.use_xyz:
					grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
			else:
				grouped_points = grouped_xyz

			for i, mlp_layer in enumerate(self.mlp_list[i]):
				grouped_points = mlp_layer(grouped_points)

			new_points = tf.math.reduce_max(grouped_points, axis=2)
			new_points_list.append(new_points)

		new_points_concat = tf.concat(new_points_list, axis=-1)

		return new_xyz, new_points_concat
