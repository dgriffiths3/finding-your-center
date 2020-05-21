import os
import sys
import json
import colorsys

sys.path.insert(0, './')

import cv2 as cv
import numpy as np
import pyvista as pv
import tensorflow as tf

from utils import helpers, tf_utils
import scannet_utils as su


def rotate_boxes(boxes, centers, theta):

	pts_out = np.zeros((boxes.shape[0], 8, 3), np.float32)

	for i, (b, c, r) in enumerate(zip(boxes, centers, theta)):

		pts_out[i] = helpers.rotate_box(b, c, r)

	return pts_out


def plot(pts, colors, labels):

	labels_mask = labels.astype(np.bool)[:, 0]
	labels = labels[labels_mask]

	centers = labels[:, :3]
	ext = labels[:, 3:6]
	theta = labels[:, 6:8]

	boxes_min = centers - (ext / 2)
	boxes_max = centers + (ext / 2)
	boxes = np.hstack((boxes_min, boxes_max))

	obj_pts = helpers.rotate_boxes(boxes, centers, theta)

	plot = pv.Plotter()
	plot.view_xy()
	plot.show_axes()

	plot.add_points(pts, scalars=colors, rgb=True, render_points_as_spheres=True, point_size=15)
	plot.add_points(labels[:, :3], color=[0, 0, 1], render_points_as_spheres=True, point_size=20)

	classes = np.linspace(0, 1, obj_pts.shape[0]+1)
	rgb_classes = np.array([colorsys.hsv_to_rgb(c, 0.8, 0.8) for c in classes])

	for i, pts in enumerate(obj_pts):
		lines = helpers.make_lines(pts)
		for l in lines:
			plot.add_mesh(l, color=rgb_classes[i], line_width=6)

	plot.show()


def create_example(pts, colors, labels):

	if len(labels.shape) > 0:
		n_inst = labels.shape[0]
	else:
		n_inst = 0

	feature = {
		'points' : tf_utils.float_list_feature(pts.reshape(-1, 1)),
		'colors' : tf_utils.float_list_feature(colors.reshape(-1, 1)),
		'labels' : tf_utils.float_list_feature(labels.reshape(-1, 1)),
		'n_inst' : tf_utils.int64_feature(n_inst)
	}

	return tf.train.Example(features=tf.train.Features(feature=feature))


def crop_scannet(config):

	scan_dirs = [x[0] for x in os.walk(config['in_dir'])]
	box_size = config['box_size']
	overlap = config['overlap']
	max_labels = 0

	with tf.io.TFRecordWriter(config['out_train_file']) as train_writer, tf.io.TFRecordWriter(config['out_test_file']) as test_writer:

		bar = helpers.progbar(len(scan_dirs[1:]))
		bar.start()

		save_count = 0
		train_count = 0
		test_count = 0
		max_labels = 0

		rotations = np.radians(np.array([0, 90, 180, 270])) if config['rotate'] == True else np.array([0.])

		for i, scan_dir in enumerate(scan_dirs[1:]):

			bar.update(i+1)

			scene_id = scan_dir.split('/')[-1]

			pts_path = os.path.join(scan_dir, '{}_vh_clean_2.labels.ply'.format(scene_id))
			agg_path = os.path.join(scan_dir, '{}_vh_clean.aggregation.json'.format(scene_id))
			segmap_path = os.path.join(scan_dir, '{}_vh_clean_2.0.010000.segs.json'.format(scene_id))

			with open(agg_path, 'r') as f: agg = json.load(f)
			segmap = su.load_segmap(segmap_path)
			scene = su.load_pointcloud(pts_path)
			objects, object_means_orig = su.load_objects(scene, agg, segmap, config['label_objects'])

			if object_means_orig.shape[0] == 0: continue

			object_thetas_orig, object_extents = helpers.get_oabb(objects)

			scene_extent = [
				np.min(scene[:, 0]), np.min(scene[:, 1]), np.min(scene[:, 2]),
				np.max(scene[:, 0]), np.max(scene[:, 1]), np.max(scene[:, 2])
			]

			x_stride_len = box_size[0]
			y_stride_len = box_size[1]

			num_xstrides = int(np.ceil((scene_extent[3] - scene_extent[0])/box_size[0]))
			num_ystrides = int(np.ceil((scene_extent[4] - scene_extent[1])/box_size[1]))

			for x_stride in range(num_xstrides):

				for y_stride in range(num_ystrides):

					bbox = [
						scene_extent[0] + (x_stride*x_stride_len) - overlap[0]/2,
						scene_extent[1] + (y_stride*y_stride_len) - overlap[0]/2,
						-1e10,
						scene_extent[0] + ((x_stride*x_stride_len) + x_stride_len) + overlap[0]/2,
						scene_extent[1] + ((y_stride*y_stride_len) + y_stride_len) + overlap[0]/2,
						1e10
					]

					scene_crop_orig = helpers.crop_bbox(scene, bbox)

					if scene_crop_orig.shape[0] < config['n_pts'] / 2: continue

					for angle in rotations:

						_, scene_crop = helpers.get_fixed_pts(scene_crop_orig, config['n_pts'])

						object_means = object_means_orig.copy()
						object_thetas = object_thetas_orig.copy()

						scene_crop[:, :3] = helpers.rotate_euler(scene_crop[:, :3], angle)
						object_means = helpers.rotate_euler(object_means_orig, angle)
						
						radians = np.arctan2(object_thetas[:, 1], object_thetas[:, 0])
						radians -= angle
						object_thetas[:, 0] = np.cos(radians)
						object_thetas[:, 1] = np.sin(radians)
						
						pts = scene_crop[:, :3]
						scene_mean = np.mean(pts, axis=0)
						pts -= scene_mean

						colors = scene_crop[:, 3:6] / 255.

						obj_occ = np.array([helpers.check_occupancy(obj_pts, bbox) for obj_pts in objects])
						obj_occ[obj_occ < 1000] = 0
						obj_occ = obj_occ.astype(np.bool)

						if True in obj_occ:
							labels = object_means[np.where(obj_occ==True)]
							labels -= scene_mean
							labels = np.hstack((labels, object_extents[np.where(obj_occ==True)]))
							labels = np.hstack((labels, object_thetas[np.where(obj_occ==True)]))
							max_labels = labels.shape[0] if labels.shape[0] > max_labels else max_labels
							labels = np.pad(labels, [[0, config['max_labels']-labels.shape[0]],[0, 0]])
						else:
							continue

						# Uncomment to visualise training data
						# plot(pts, colors, labels)

						tf_example = create_example(pts, colors, labels)
						
						if i % 5 != 0:
							train_writer.write(tf_example.SerializeToString())
							train_count += 1
						else:
							test_writer.write(tf_example.SerializeToString())
							test_count += 1
						save_count += 1

		bar.finish()

		print('[info] saved {} examples, {} train, {} test'.format(save_count, train_count, test_count))
		print('[info] max label count: {}'.format(max_labels))


if __name__ == '__main__':

	config = {
		'in_dir' : './data/scannet/scans',
		'out_train_file' : './data/scannet_scene_train.tfrecord',
		'out_test_file' : './data/scannet_scene_test.tfrecord',
		'label_objects' : ['chair', 'office chair'],
		'box_size' : (1.5, 1.5),
		'overlap' : (1.5, 1.5),
		'max_labels' : 23,
		'rotate' : True,
		'n_pts' : 32768,
	}

	crop_scannet(config)
