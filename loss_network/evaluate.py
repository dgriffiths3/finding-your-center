import os
import sys

sys.path.insert(0, './')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pyvista as pv
import tensorflow as tf

from model import LossNetModel
from utils import helpers, tf_utils
from utils.dataset import load_loss_dataset

tf.random.set_seed(42)


def forward_pass(model, dir_loss, ext_loss, clf_acc, pts, labels):

	clf_labels = tf.one_hot(tf.cast(labels[:, 0], tf.int64), depth=2)

	pred = model(pts)

	dir_loss.update_state(labels[:, 2:5], pred[:, :3])
	ext_loss.update_state(labels[:, 5:10], pred[:, 3:8])
	clf_acc.update_state(clf_labels, pred[:, 8:10])

	return dir_loss, ext_loss, clf_acc


def inference():

	model = LossNetModel(config['batch_size'])

	model(tf.zeros((config['batch_size'], config['num_points'], 3)))
	model.load_weights(config['weights'])
	
	print('[info] weights loaded successfully.')

	dataset = load_loss_dataset(config['dataset'], config['batch_size'], config['num_points'], repeat=False)

	dir_loss = tf.keras.metrics.MeanSquaredError()
	ext_loss = tf.keras.metrics.MeanSquaredError()
	clf_acc = tf.keras.metrics.BinaryAccuracy()

	print('[info] running evlaution on test scenes...\n')

	for step, (pts, labels) in enumerate(dataset):

		dir_loss, ext_loss, clf_acc = forward_pass(model, dir_loss, ext_loss, clf_acc, pts, labels)

		if step == config['num_evals']: break

	print('Direction MSE: {:.4f}'.format(dir_loss.result()))
	print('Extent MSE: {:.4f}'.format(ext_loss.result()))
	print('Classifcation Acc.: {:.4f}'.format(clf_acc.result()))


if __name__ == '__main__':

	config = {
		'dataset' : '../data/chairs/chair_crops/loss_1_test.tfrecord',
		'weights' : '../logs/final_loss/chairs/loss_1/models/weights.ckpt',
		'num_evals': 100,
		'batch_size' : 1,
		'num_points' : 4096,
	}

	inference()
