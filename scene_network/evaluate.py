import os
import sys

sys.path.insert(0, './')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pyvista as pv
import tensorflow as tf

from model import ObjDetectorModel
from utils.dataset import load_scene_dataset
from utils import helpers, tf_utils, losses, eval

tf.random.set_seed(42)


def forward_pass(model, pts, cols, labels, metrics, means):

	pred_xyz, pred_dist, pred_ext, _ = model(pts)
	pred_dist = tf.expand_dims(pred_dist[:, :, 1], -1)
	pred_xyz, pred_dist, pred_ext = tf_utils.objectness_mask(pred_xyz, pred_dist, pred_ext, config['score_thresh'])

	boxes = helpers.make_boxes(pred_xyz, pred_ext)
	label_boxes = helpers.make_boxes(labels[:, :, :3], labels[:, :, 3:8])

	if pred_xyz.shape[1] > 1 and config['nms'] == True:
		
		nms_inds = tf_utils.nms(pred_xyz, boxes, pred_dist, 15, 0.25, config['score_thresh'])

		pred_xyz = tf.gather(pred_xyz, nms_inds, batch_dims=1)
		pred_dist = tf.gather(pred_dist, nms_inds, batch_dims=1)
		pred_ext = tf.gather(pred_ext, nms_inds, batch_dims=1)
		boxes = tf.gather(boxes, nms_inds, batch_dims=1)

	chamfer_dist = losses.chamfer_distance(labels[:, :, :3], pred_xyz)
	box_results = eval.box_eval(boxes, labels, label_boxes, config['iou_thresh']).reshape(-1)

	metrics[0].update_state([chamfer_dist])
	metrics[1].update_state([box_results[0]])
	metrics[2].update_state([box_results[1]])
	metrics[3].update_state([box_results[2]])

	return metrics, np.insert(box_results, 0, chamfer_dist)


def inference():

	model = ObjDetectorModel(1, config['n_pred'])

	model(tf.zeros((1, config['num_points'], 3)))
	model.load_weights(config['weights'])
	
	metrics = [tf.keras.metrics.Mean() for _ in range(4)]

	ds_path = os.path.join(config['dataset'])
	dataset = load_scene_dataset(ds_path, 1, config['max_labels'], repeat=False)

	means = []

	for step, (pts, cols, labels) in enumerate(dataset):

		metrics, scene_scores = forward_pass(
			model,
			pts,
			cols,
			labels,
			metrics,
			means
		)

		means.append(scene_scores)

		if step % 10 == 0:
		 	print('[info] step: {}, Chamfer: {:.4f}, Prec: {:.2f}, Rec: {:.2f}, mAP: {:.2f}'.format(
		 		step, metrics[0].result(), metrics[1].result(), metrics[2].result(), metrics[3].result())
		 	)

	np.savetxt(
		os.path.join(os.path.dirname(config['weights']), 'means.csv'),
		np.array(means),
		delimiter=',',
		fmt='%.4f,%.4f,%.4f,%.4f'
	)

	print('---------')
	print('[results] step: {}, Chamfer: {:.4f}, Prec: {:.2f}, Rec: {:.2f}, mAP: {:.2f}'.format(
		step, metrics[0].result(), metrics[1].result(), metrics[2].result(), metrics[3].result())
	)

if __name__ == '__main__':

	config = {
		'dataset' : './data/s3d_scene_test.tfrecord',
		'weights' : './logs/scenenet_1/models/weights.ckpt',
		'iou_thresh' : 0.25,
		'score_thresh' : 0.9,
		'max_labels' : 25,
		'n_pred' : 15,
		'num_points' : 32768,
		'nms': True
	}

	inference()
