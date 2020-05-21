import os
import sys
import glob
import colorsys

sys.path.insert(0, './')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pyvista as pv
import tensorflow as tf

from scene_network.model import ObjDetectorModel
from utils import helpers, tf_utils, losses


def plot_scene(scene, scene_pts, scene_ext, gt_pts, gt_ext):

	pts = scene[:, 0:3]
	cols = scene[:, 3:6]

	cols = cols[pts[:, 2] < np.max(pts[:, 2])-1.5]
	pts = pts[pts[:, 2] < np.max(pts[:, 2])-1.5]

	plot = pv.Plotter()
	plot.set_background('white')
	
	plot.add_points(pts, scalars=cols, rgb=True, opacity=1, render_points_as_spheres=True, point_size=10)
	
	if scene_pts.shape[0] > 0:
	
		ext_hwd = scene_ext[:, :3]
		ext_theta = scene_ext[:, 3:5]
	
		boxes_min = scene_pts - (ext_hwd / 2)
		boxes_max = scene_pts + (ext_hwd / 2)
		boxes = np.hstack((boxes_min, boxes_max))
	
		box_pts = helpers.rotate_boxes(boxes, scene_pts, ext_theta)
	
		classes = np.linspace(0, 1, box_pts.shape[0]+1)
		rgb_classes = np.array([colorsys.hsv_to_rgb(c, 0.8, 0.8) for c in classes])
	
		for i, box in enumerate(box_pts):
			lines = helpers.make_lines(box)
			[plot.add_mesh(l, color=rgb_classes[i], line_width=4) for l in lines]
	
	plot.view_xy()
	plot.show()


def parse_room():

	model = ObjDetectorModel(1, config['n_pred'])
	model(tf.zeros((1, config['n_pts'], 3), tf.float32))
	model.load_weights(config['weights'])

	if config['dataset'] == 's3d':
		room = 'Area_' + str(config['area']) + '_' + config['room'] + '.npy'
		scene = np.load(os.path.join(config['dataset_dir'], 'processed', room))

	scene_extent = [
		np.min(scene[:, 0]), np.min(scene[:, 1]), np.min(scene[:, 2]),
		np.max(scene[:, 0]), np.max(scene[:, 1]), np.max(scene[:, 2])
	]

	object_paths = glob.glob(
		os.path.join(
			config['dataset_dir'],
			'Area_'+str(config['area']),
			config['room'],
			'Annotations',
			'*chair*.npy'
		)
	)

	objects = np.array([np.load(o_f)[:, :3] for o_f in object_paths])
	gt_pts = np.array([np.mean(o, axis=0) for o in objects])
	gt_theta, gt_ext = helpers.get_oabb(objects)
	gt_ext = np.hstack((gt_ext, gt_theta))

	x_stride_len = config['box_size'][0]
	y_stride_len = config['box_size'][1]

	num_xstrides = int(np.ceil((scene_extent[3] - scene_extent[0])/x_stride_len))
	num_ystrides = int(np.ceil((scene_extent[4] - scene_extent[1])/y_stride_len))

	scene_pts = []
	scene_ext = []

	for x_stride in range(num_xstrides):

		for y_stride in range(num_ystrides):

			bbox = [
				scene_extent[0] + (x_stride*x_stride_len),
				scene_extent[1] + (y_stride*y_stride_len),
				-1e10,
				scene_extent[0] + ((x_stride*x_stride_len) + x_stride_len),
				scene_extent[1] + ((y_stride*y_stride_len) + y_stride_len),
				1e10
			]

			scene_crop = helpers.crop_bbox(scene, bbox)
			_, scene_crop = helpers.get_fixed_pts(scene_crop, config['n_pts'])

			pts = scene_crop[:, 0:3]
			cols = scene_crop[:, 3:6] / 256.

			pts_mean = np.mean(pts, axis=0)
			pts -= pts_mean

			xyz, score, ext, _ = model(tf.expand_dims(pts, 0))
			xyz, score, ext = tf_utils.objectness_mask(xyz, score[:, :, 1], ext, config['score_thresh'])

			labels = gt_pts[
				(gt_pts[:, 0] >= bbox[0]) & (gt_pts[:, 0] <= bbox[3]) &
				(gt_pts[:, 1] >= bbox[1]) & (gt_pts[:, 1] <= bbox[4]) &
				(gt_pts[:, 2] >= bbox[2]) & (gt_pts[:, 2] <= bbox[5])
			]

			labels = tf.expand_dims(labels - pts_mean, 0)

			if xyz.shape[1] > 1 and config['nms'] == True:
				boxes = helpers.make_boxes(xyz, ext)
				nms_inds = tf_utils.nms(xyz, boxes, score, 15, 0, config['score_thresh'])

				xyz = tf.gather(xyz, nms_inds, batch_dims=1)
				score = tf.gather(score, nms_inds, batch_dims=1)
				boxes = tf.gather(boxes, nms_inds, batch_dims=1)
				ext = tf.gather(ext, nms_inds, batch_dims=1)

			if xyz.shape[1] > 0:
				for i, pred in enumerate(xyz[0]):
					scene_pts.append((pred.numpy()+pts_mean).tolist())
					scene_ext.append(ext[0, i].numpy().tolist())

	scene_pts = np.array(scene_pts)
	scene_ext = np.array(scene_ext)

	plot_scene(scene, scene_pts, scene_ext, gt_pts, gt_ext)


if __name__ == '__main__':

	config = {
		'dataset' : 's3d',
		'dataset_dir' : './data/Stanford3dDataset',
		'area' : 5,
		'room' : 'office_1',
		'weights' : './logs/scenenet_1/models/weights.ckpt',
		'nms' : False,
		'score_thresh' : 0.9,
		'box_size' : (3, 3),
		'n_pts' : 32768,
		'n_pred' : 15
	}

	parse_room()
