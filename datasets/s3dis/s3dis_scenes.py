import os
import sys
import colorsys
sys.path.insert(0, './')

import glob
import string
import numpy as np
import pyvista as pv
import tensorflow as tf

from utils import helpers, tf_utils


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

	obj_pts = rotate_boxes(boxes, centers, theta)

	plot = pv.Plotter()
	plot.view_xy()

	# Remove ceiling
	colors = colors[pts[:, 2] < np.max(pts[:, 2])-1.]
	pts = pts[pts[:, 2] < np.max(pts[:, 2])-1.]

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

	n_inst = labels.shape[0] if len(labels.shape) > 0 else 0

	feature = {
		'points' : tf_utils.float_list_feature(pts.reshape(-1, 1)),
		'colors' : tf_utils.float_list_feature(colors.reshape(-1, 1)),
		'labels' : tf_utils.float_list_feature(labels.reshape(-1, 1)),
		'n_inst' : tf_utils.int64_feature(n_inst)
	}

	return tf.train.Example(features=tf.train.Features(feature=feature))


def crop_s3dis():

	filelist = glob.glob(os.path.join(config['in_dir'], '*.npy'))
	box_size = config['box_size']
	overlap = config['overlap']

	saved = 0

	with tf.io.TFRecordWriter(config['out_train_file']) as train_writer, tf.io.TFRecordWriter(config['out_test_file']) as test_writer:

		bar = helpers.progbar(len(filelist))
		bar.start()

		max_labels = 0

		rotations = np.radians(np.array([0, 90, 180, 270])) if config['rotate'] == True else np.array([0.])

		for i, f in enumerate(filelist):

			bar.update(i+1)

			scene = np.load(f)

			area = '_'.join(f.split('/')[-1].split('_')[:2])
			room = '_'.join(f.split('/')[-1].split('.')[0].split('_')[2:])

			area_n = int(f.split('/')[-1].split('_')[1])

			object_paths = glob.glob(os.path.join(config['root_dir'], area, room, 'Annotations', '*{}*.npy'.format(config['label_object'])))
			objects = np.array([np.load(o_f)[:, :3] for o_f in object_paths])
			object_means_orig = np.array([np.mean(o, axis=0) for o in objects])

			if object_means_orig.shape[0] == 0: continue

			object_thetas_orig, object_extents = helpers.get_oabb(objects)

			area = int(f.split('/')[-1].split('_')[1])

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
							try:
								labels = object_means[np.where(obj_occ==True)]
								labels -= scene_mean
								labels = np.hstack((labels, object_extents[np.where(obj_occ==True)]))
								labels = np.hstack((labels, object_thetas[np.where(obj_occ==True)]))
								max_labels = labels.shape[0] if labels.shape[0] > max_labels else max_labels
								labels = np.pad(labels, [[0, config['max_labels']-labels.shape[0]],[0, 0]])
							except:
								print(labels.shape)
								continue
						else:
							continue

						# Uncomment to visualise training data
						# plot(pts, colors, labels)

						tf_example = create_example(pts, colors, labels)

						if area_n != 5:
							train_writer.write(tf_example.SerializeToString())
						else:
							test_writer.write(tf_example.SerializeToString())
						saved += 1
		bar.finish()

		print('[info] total scenes generated: {}'.format(saved))
		print('[info] max label count: {}'.format(max_labels))


if __name__ == '__main__':

	config = {
		'root_dir' : './data/Stanford3dDataset',
		'in_dir': './data/Stanford3dDataset/processed',
		'out_train_file' : './data/s3d_scene_train.tfrecord',
		'out_test_file' : './data/s3d_scene_test.tfrecord',
		'label_object' : 'chair',
		'box_size' : (1.5, 1.5),
		'overlap' : (1.5, 1.5),
		'max_labels' : 25,
		'rotate' : True,
		'n_pts' : 32768
	}

	crop_s3dis()
