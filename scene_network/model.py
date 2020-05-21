import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from utils import tf_utils
from utils.halton import halton_batch
from pointnet2.model import Pointnet2Encoder


class MiniNet(keras.Model):

	def __init__(self, batch_size, n_pred, activation=tf.nn.relu, k_init='glorot_normal', k_reg=None):
		super(MiniNet, self).__init__()

		self.batch_size = batch_size
		self.activation = tf.nn.relu
		self.kernel_initializer = 'glorot_normal'
		self.kernel_regularizer = None

		self.encoder = Pointnet2Encoder(
			self.batch_size,
			n_pred,
			self.activation,
			msg = False,
			net_type='scene'
		)

	def call(self, input):

		encoded = self.encoder(input)
		encoded = tf.reshape(encoded, (self.batch_size, -1))

		return encoded


class ObjDetectorModel(keras.Model):

	def __init__(self, batch_size, n_pred=1):
		super(ObjDetectorModel, self).__init__()

		self.batch_size = batch_size
		self.n_pred = n_pred
		self.activation = tf.nn.relu
		self.kernel_initializer = 'glorot_normal'
		self.kernel_regularizer = None

		self.halton_offset = halton_batch(batch_size, 3, n_pred, extent=1.25, z_offset=-1)

		self.encoder = Pointnet2Encoder(
			self.batch_size,
			self.n_pred,
			self.activation,
			msg = False,
			net_type = 'scene'
		)

		self.mininet = MiniNet(
			self.batch_size*n_pred, n_pred,
			self.activation,
			self.kernel_initializer,
			self.kernel_regularizer
		)

		self.init_point_net()
		self.init_score_net()
		self.init_extent_net()


	def init_point_net(self):

		self.dense1 = layers.Dense(
			512,
			activation=self.activation,
			kernel_initializer=self.kernel_initializer,
			kernel_regularizer=self.kernel_regularizer
		)

		self.dense2 = layers.Dense(
			1024,
			activation=self.activation,
			kernel_initializer=self.kernel_initializer,
			kernel_regularizer=self.kernel_regularizer
		)

		self.pred = layers.Dense(
			3,
			activation=None,
			kernel_initializer=self.kernel_initializer,
			kernel_regularizer=self.kernel_regularizer
		)


	def init_score_net(self):

		self.score_dense1 = layers.Dense(
			512,
			activation = self.activation,
			kernel_initializer=self.kernel_initializer,
			kernel_regularizer=self.kernel_regularizer
		)

		self.score_pred = layers.Dense(
			2,
			activation = tf.nn.softmax,
			kernel_initializer=self.kernel_initializer,
			kernel_regularizer=self.kernel_regularizer
		)


	def init_extent_net(self):

		self.ext_dense1 = layers.Dense(
			512,
			activation=self.activation,
			kernel_initializer=self.kernel_initializer,
			kernel_regularizer=self.kernel_regularizer
		)

		self.ext_pred = layers.Dense(
			5,
			activation=None,
			kernel_initializer=self.kernel_initializer,
			kernel_regularizer=self.kernel_regularizer
		)


	def call_point_net(self, net):

		net = tf.reshape(net, (self.batch_size, self.n_pred, -1))

		net = self.dense1(net)
		net = self.dense2(net)
		pred = self.pred(net)

		pred += self.halton_offset
		pred = tf.concat(
		 	[tf.clip_by_value(pred[:, :, :2], -2, 2),
		 	tf.expand_dims(pred[:, :, 2], -1)],
		 	axis=-1
		)

		return pred


	def call_score_net(self, net):

		net = tf.reshape(net, (self.batch_size, self.n_pred, -1))
		net = self.score_dense1(net)
		return self.score_pred(net)


	def call_ext_net(self, net):

		net = tf.reshape(net, (self.batch_size, self.n_pred, -1))
		net = self.ext_dense1(net)
		return self.ext_pred(net)


	def call(self, input):

		scene_code = self.encoder(input)

		points = self.call_point_net(scene_code)

		boxes = tf_utils.pred_to_boxes(points, 1)
		crops = tf_utils.fixed_crop3d(input, boxes, points)
		crops = tf.reshape(crops, (crops.shape[0] * crops.shape[1], -1, 3))

		patch_code = self.mininet(crops)
		scene_code = tf.tile(
			scene_code,
			[tf.cast(patch_code.shape[0]/scene_code.shape[0], tf.int8), 1]
		)

		patch_code = tf.concat([scene_code, patch_code], -1)

		scores = self.call_score_net(patch_code)
		exts = self.call_ext_net(patch_code)

		return points, scores, exts, crops