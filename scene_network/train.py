import os
import sys

sys.path.insert(0, './')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import metrics_util
from model import ObjDetectorModel
from utils import helpers, tf_utils, losses
from utils.dataset import load_scene_dataset

tf.random.set_seed(42)


def test_step(model, metrics, test_pts, test_labels):

	pred_points, pred_scores, pred_exts, _ = model(test_pts)
	
	loss_xyz, loss_score, loss_ext, _ = losses.supervised_loss(
		test_labels, pred_points, pred_scores, pred_exts, config['loss_size'], not config['unsupervised']
	)
	
	metrics = metrics_util.update_test_metrics(metrics, loss_xyz, loss_ext, loss_score)

	return metrics


def train_step(optimizer, model, metrics, loss_model, train_pts, train_labels, step):

	with tf.GradientTape() as tape:

		pred_points, pred_scores, pred_exts, crops = model(train_pts)

		if config['loss_network_inf']:

			boxes = tf_utils.pred_to_boxes(pred_points, config['loss_size'])
			crops = tf_utils.fixed_crop3d(train_pts, boxes, pred_points)

			lossnet_results = tf.reshape(
				loss_model(tf.reshape(crops, (crops.shape[0]*crops.shape[1], -1, 3))),
				(pred_points.shape[0], -1, 10)
			)

			loss_score, loss_ext, iou_mask = losses.unsupervised_loss(
				pred_points, pred_scores, pred_exts, lossnet_results
			)

			loss_iou = losses.batch_iou_loss_2d(pred_points, lossnet_results, iou_mask)

		else:

			loss_ext  = loss_iou = loss_score = loss_conf = 0

		loss_sup_xyz, loss_sup_score, loss_sup_ext, sup_labels = losses.supervised_loss(
			train_labels, pred_points, pred_scores, pred_exts, config['loss_size'], not config['unsupervised']
		)

		if config['unsupervised'] == True:
			loss = loss_ext + loss_score + loss_iou
		else:
			loss = loss_sup_xyz + loss_sup_score+ loss_sup_ext

	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	metrics = metrics_util.update_train_metrics(
		metrics, loss, loss_ext, loss_score, loss_iou, loss_sup_xyz, loss_sup_ext, loss_sup_score
	)

	return metrics, pred_points


def train():

	model = ObjDetectorModel(config['batch_size'], config['n_pred'])

	loss_model = tf_utils.get_loss_model(config)

	optimizer = tf.keras.optimizers.Adam(lr=config['lr'])

	train_metrics, test_metrics = metrics_util.get_metrics()

	train_dataset = load_scene_dataset(config['train_dataset'], config['batch_size'], config['max_labels'])
	test_dataset = iter(load_scene_dataset(config['test_dataset'], config['batch_size'], config['max_labels']))

	train_summary_writer = tf.summary.create_file_writer(
		os.path.join(config['log_dir'], config['log_code'], 'tensorboard', 'train')
	)
	test_summary_writer = tf.summary.create_file_writer(
		os.path.join(config['log_dir'], config['log_code'], 'tensorboard', 'test')
	)
	checkpoint_dir = os.path.join(config['log_dir'], config['log_code'], 'models')

	tf_utils.print_summary(model, config)

	min_loss = 1e10

	with train_summary_writer.as_default():
		tf.summary.text('config', tf_utils.tb_config(config), step=0)

	for step, (train_pts, train_cols, train_labels) in enumerate(train_dataset):

		train_metrics, pred_points = train_step(
			optimizer,
			model,
			train_metrics,
			loss_model,
			train_pts,
			train_labels,
			step
		)

		if step % config['log_freq'] == 0:
			metrics_util.write_summaries(train_summary_writer, pred_points, train_metrics, step, 'train')

		if step % config['test_freq'] == 0:

			test_pts, test_cols, test_labels = next(test_dataset)
			test_metrics = test_step(
				model,
				test_metrics,
				test_pts,
				test_labels
			)

			metrics_util.write_summaries(test_summary_writer, None, test_metrics, step, 'test')

		if test_metrics[0].result() < min_loss and step>1000:
			model.save_weights(
				os.path.join(checkpoint_dir, 'weights.ckpt')
				, overwrite = True
			)
			min_loss = test_metrics[5].result()


if __name__ == '__main__':

	config = {
		'train_dataset': './data/s3d_scene_train.tfrecord',
		'test_dataset': './data/s3d_scene_test.tfrecord',
		'loss_weights': './logs/lossnet_1/models/weights.ckpt',
		'log_dir': './logs',
		'log_code': 'scenenet_1',
		'log_freq': 10,
		'test_freq': 25,
		'loss_size': 1.,
		'loss_points': 4096,
		'num_points': 32768,
		'batch_size': 2,
		'lr': 1e-4,
		'n_pred': 15,
		'max_labels': 25,
		'unsupervised': True,
		'loss_network_inf': True,
	}

	train()
