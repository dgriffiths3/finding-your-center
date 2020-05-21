import os
import sys

sys.path.insert(0, './')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from model import LossNetModel
from utils import helpers, tf_utils
from utils.dataset import load_loss_dataset

tf.random.set_seed(42)


def test_step(model, metrics, test_pts, test_labels):

	clf_labels = tf.one_hot(tf.cast(test_labels[:, 0], tf.int64), depth=2)

	pred = model(test_pts)

	arg_max = tf.cast(tf.math.argmax(pred[:, 8:10], axis=-1), tf.float32)

	dir_loss = tf.keras.losses.MeanSquaredError()(test_labels[:, 2:5], pred[:, 0:3])
	ext_loss = tf.keras.losses.MeanSquaredError()(test_labels[:, 5:10], pred[:, 3:8])
	clf_loss = tf.keras.losses.BinaryCrossentropy()(clf_labels, pred[:, 8:10])
	loss = dir_loss + ext_loss + clf_loss

	metrics[0].update_state([loss])
	metrics[1].update_state([dir_loss])
	metrics[2].update_state([ext_loss])
	metrics[3].update_state([clf_loss])
	metrics[4].update_state(arg_max, test_labels[:, 0])

	return metrics, pred


def train_step(optimizer, model, metrics, train_pts, train_labels):

	clf_labels = tf.one_hot(tf.cast(train_labels[:, 0], tf.int64), depth=2)

	with tf.GradientTape(persistent=True) as tape:

		pred = model(train_pts)

		arg_max = tf.cast(tf.math.argmax(pred[:, 8:10], axis=-1), tf.float32)

		dir_loss = tf.keras.losses.MeanSquaredError()(train_labels[:, 2:5], pred[:, 0:3])
		ext_loss = tf.keras.losses.MeanSquaredError()(train_labels[:, 5:10], pred[:, 3:8])
		clf_loss = tf.keras.losses.BinaryCrossentropy()(clf_labels, pred[:, 8:10])
		loss = dir_loss + ext_loss + clf_loss

	gradients = tape.gradient(loss, model.trainable_variables)

	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	metrics[0].update_state([loss])
	metrics[1].update_state([dir_loss])
	metrics[3].update_state([clf_loss])
	metrics[2].update_state([ext_loss])
	metrics[4].update_state(arg_max, train_labels[:, 0])

	return metrics, pred


def train():

	model = LossNetModel(config['batch_size'])

	optimizer = tf.keras.optimizers.Adam(lr=config['lr'])

	train_metrics = [tf.keras.metrics.Mean() for _ in range(4)]
	test_metrics = [tf.keras.metrics.Mean() for _ in range(4)]

	train_metrics.append(tf.keras.metrics.Accuracy())
	test_metrics.append(tf.keras.metrics.Accuracy())

	train_dataset = load_loss_dataset(config['train_dataset'], config['batch_size'], config['num_points'])
	test_dataset = iter(load_loss_dataset(config['test_dataset'], config['batch_size'], config['num_points']))

	train_summary_writer = tf.summary.create_file_writer(
		os.path.join(config['log_dir'], config['log_code'], 'tensorboard', 'train')
	)
	test_summary_writer = tf.summary.create_file_writer(
		os.path.join(config['log_dir'], config['log_code'], 'tensorboard', 'test')
	)

	checkpoint_dir = os.path.join(config['log_dir'], config['log_code'], 'models')
	if not os.path.isdir(checkpoint_dir): os.makedirs(checkpoint_dir, exist_ok=True)

	tf_utils.print_summary(model, config)

	min_loss = 1e10

	with train_summary_writer.as_default():

		tf.summary.text('config', tf_utils.tb_config(config), step=0)

	for step, (train_pts, train_labels) in enumerate(train_dataset):

		train_metrics, pred = train_step(
			optimizer,
			model,
			train_metrics,
			train_pts,
			train_labels
		)

		with train_summary_writer.as_default():
			if step % config['log_freq'] == 0:
				tf.summary.scalar('total loss', train_metrics[0].result(), step=step)
				tf.summary.scalar('dir loss', train_metrics[1].result(), step=step)
				tf.summary.scalar('ext loss', train_metrics[2].result(), step=step)
				tf.summary.scalar('clf loss', train_metrics[3].result(), step=step)
				tf.summary.scalar('argmax accuracy', train_metrics[4].result(), step=step)

		if step % config['test_freq'] == 0:

			test_pts, test_labels = next(test_dataset)

			test_metrics, pred = test_step(model, test_metrics, test_pts, test_labels)

			with test_summary_writer.as_default():

				tf.summary.scalar('total loss', test_metrics[0].result(), step=step)
				tf.summary.scalar('dir loss', test_metrics[1].result(), step=step)
				tf.summary.scalar('ext loss', test_metrics[2].result(), step=step)
				tf.summary.scalar('clf. loss', test_metrics[3].result(), step=step)
				tf.summary.scalar('argmax accuracy', test_metrics[4].result(), step=step)

			if test_metrics[1].result() < min_loss and step > 1000:
				model.save_weights(
					os.path.join(
						config['log_dir'], config['log_code'], 'models', 'weights.ckpt'
					), overwrite = True
				)

				min_loss = test_metrics[1].result()


if __name__ == '__main__':

	config = {
		'train_dataset' : './data/s3d_patches_train.tfrecord',
		'test_dataset' : './data/s3d_patches_train.tfrecord',
		'log_dir' : './logs',
		'log_code' : 'lossnet_1',
		'test_freq' : 25,
		'log_freq' : 25,
		'n_steps' : 2e+5,
		'batch_size' : 16,
		'num_points' : 4096,
		'lr' : 1e-4
	}

	train()
