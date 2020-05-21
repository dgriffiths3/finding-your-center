import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pointnet2.model import Pointnet2Encoder


class LossNetModel(keras.Model):

	def __init__(self, batch_size):
		super(LossNetModel, self).__init__()

		self.batch_size = batch_size
		self.activation = tf.nn.relu
		self.kernel_initializer = 'glorot_normal'
		self.kernel_regularizer = None

		self.dropout = False
		self.keep_prob = 0.4

		self.encoder = Pointnet2Encoder(
			self.batch_size,
			None,
			self.activation,
			msg = False,
			net_type='loss'
		)

		self.init_gradient_net()
		self.init_extent_net()
		self.init_classifier_net()


	def init_gradient_net(self):

		self.reg_dense1 = layers.Dense(
			512,
			activation=self.activation,
			kernel_initializer=self.kernel_initializer,
			kernel_regularizer=self.kernel_regularizer
		)

		self.reg_dropout1 = layers.Dropout(self.keep_prob)

		self.reg_pred = layers.Dense(
			3,
			activation=None,
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

		self.ext_dropout1 = layers.Dropout(self.keep_prob)

		self.ext_pred = layers.Dense(
			5,
			activation=None,
			kernel_initializer=self.kernel_initializer,
			kernel_regularizer=self.kernel_regularizer
		)


	def init_classifier_net(self):

		self.clf_dense1 = layers.Dense(
			512,
			activation=self.activation,
			kernel_initializer=self.kernel_initializer,
			kernel_regularizer=self.kernel_regularizer
		)

		self.clf_dropout1 = layers.Dropout(self.keep_prob)

		self.clf_pred = layers.Dense(
			2,
			activation=tf.nn.softmax,
			kernel_initializer=self.kernel_initializer,
			kernel_regularizer=self.kernel_regularizer
		)


	def call_gradient_net(self, net):

		net = self.reg_dense1(net)
		net = self.reg_dropout1(net)

		return self.reg_pred(net)


	def call_extent_net(self, net):

		net = self.ext_dense1(net)
		net = self.ext_dropout1(net)

		return self.ext_pred(net)


	def call_classifier_net(self, net):

		net = self.clf_dense1(net)
		net = self.clf_dropout1(net)

		return self.clf_pred(net)


	def call(self, input):

		encoded = self.encoder(input)
		encoded = tf.reshape(encoded, (self.batch_size, -1))

		reg_pred = self.call_gradient_net(encoded)
		ext_pred = self.call_extent_net(encoded)
		clf_pred = self.call_classifier_net(encoded)

		return tf.concat([reg_pred, ext_pred, clf_pred], axis=1)
