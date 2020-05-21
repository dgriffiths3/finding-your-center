import os
import sys

sys.path.insert(0, './')

import glob
import cv2 as cv
import numpy as np
import pyvista as pv
import tensorflow as tf

from utils import helpers, tf_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def plot(pts, colors, dir, label_mean, label):

	colors = colors / 256.

	ext = label[5:8]
	theta = label[8:10]
	center = np.array([0., 0., 0.])

	box_min = label_mean - (ext / 2)
	box_max = label_mean + (ext / 2)
	box = np.hstack((box_min, box_max))

	box_pts = helpers.rotate_box(box, label_mean, theta)

	plot = pv.Plotter()

	plot.add_points(label_mean, color=[0, 1, 0], render_points_as_spheres=True, point_size=30)
	plot.add_points(pts, scalars=colors, rgb=True, render_points_as_spheres=True, point_size=20)
	plot.add_arrows(center, dir, 0.75, color=[0, 0, 1])

	for l in helpers.make_lines(box_pts):
		plot.add_mesh(l, color=[1, 0, 0], line_width=6) 

	plot.show()


def create_example(pts, colors, label):

	feature = {
		'points' : tf_utils.float_list_feature(pts.reshape(-1, 1)),
		'colors' : tf_utils.float_list_feature(pts.reshape(-1, 1)),
		'label' : tf_utils.float_list_feature(label.reshape(-1, 1)),
	}

	return tf.train.Example(features=tf.train.Features(feature=feature))


def generate_random_crops(config):

	crop_size = config['crop_size']
	scene_crop_size = config['scene_crop_size']
	loss_object = tf.keras.losses.MeanSquaredError()

	pos_count = 0
	neg_count = 0
	room_count = 0

	n_rooms = np.array([len(os.listdir(os.path.join(config['root_dir'], 'Area_{}'.format(i+1)))) for i in range(6)]).sum()

	bar = helpers.progbar(n_rooms)
	bar.start()

	with tf.io.TFRecordWriter(config['out_train_file']) as train_writer, tf.io.TFRecordWriter(config['out_test_file']) as test_writer:
		
		for i in range(1, 7):
			
			area_path = os.path.join(config['root_dir'], 'Area_{}'.format(i))
			
			for room_folder in os.listdir(area_path):
				
				room_path = os.path.join(area_path, room_folder)

				if not os.path.isdir(room_path): continue

				room_count += 1
				bar.update(room_count)

				# You can speed this script up by pre loading .txt files and saving as .npy
				room_cloud = np.loadtxt(glob.glob(room_path+'/*.txt')[0], delimiter=' ')

				obj_paths = glob.glob(os.path.join(room_path, 'Annotations', '*{}*.npy'.format(config['object'])))

				objects = np.array([np.load(obj_path) for obj_path in obj_paths])
				object_thetas, object_extents = helpers.get_oabb(objects)
				obj_means = np.array([np.mean(p[:, :3], axis=0) for p in objects])

				for obj_count, obj in enumerate(objects):

					if i != 5 and np.random.rand() > config['train_ratio']: continue # ignore object

					obj_mean = obj_means[obj_count]

					for j in range(config['n_crops_per_object']):

						crop_x = np.random.uniform(obj_mean[0]-(crop_size[0]/2), obj_mean[0]+(crop_size[0]/2))
						crop_y = np.random.uniform(obj_mean[1]-(crop_size[1]/2), obj_mean[1]+(crop_size[1]/2))
						crop_z = np.random.uniform(obj_mean[2]-(crop_size[2]/2), obj_mean[2]+(crop_size[2]/2))

						scene_bbox = np.array([
						crop_x - (scene_crop_size[0]/2.), crop_y - (scene_crop_size[1]/2.), crop_z - 1e10,
						crop_x + (scene_crop_size[0]/2.), crop_y + (scene_crop_size[1]/2.), crop_z + 1e10,
						])

						bbox = np.array([
							crop_x - (crop_size[0]/2.), crop_y - (crop_size[1]/2.), crop_z - (crop_size[2]/2.),
							crop_x + (crop_size[0]/2.), crop_y + (crop_size[1]/2.), crop_z + (crop_size[2]/2.),
						])

						crop_centre = np.array([crop_x, crop_y, crop_z])

						l_dist = [helpers.euc_dist([crop_x, crop_y, crop_z], c) for c in obj_means]

						obj_mean = obj_means[np.argmin(l_dist)]
						obj_ext = object_extents[np.argmin(l_dist)]
						obj_theta = object_thetas[np.argmin(l_dist)]

						curr_obj = objects[np.argmin(l_dist)]

						norm_bbox = np.array([
							bbox[0] - crop_centre[0], bbox[1] - crop_centre[1], bbox[2] - crop_centre[2],
							bbox[3] - crop_centre[0], bbox[4] - crop_centre[1], bbox[5] - crop_centre[2]
						])

						label_mean = obj_mean - crop_centre

						for _ in range(config['n_rotations_per_object']):

							tmp_room_cloud = helpers.crop_bbox(room_cloud, scene_bbox)
							_, tmp_room_cloud = helpers.get_fixed_pts(tmp_room_cloud, config['n_scene_pts'])
							tmp_room_crop = helpers.crop_bbox(tmp_room_cloud, bbox)
							tmp_room_crop[:, :3] -= crop_centre

							theta = np.random.uniform() * 2.0 * np.pi
							tmp_room_crop[:, :3] = helpers.rotate_euler(tmp_room_crop[:, :3], theta)
							label_mean_r = helpers.rotate_euler(label_mean.reshape(1, 3), theta)[0]

							new_angle = np.arctan2(obj_theta[1], obj_theta[0]) - theta
							theta_r = np.array([np.cos(new_angle), np.sin(new_angle)])

							tmp_room_crop = helpers.crop_bbox(tmp_room_crop, norm_bbox)

							check = helpers.check_occupancy(curr_obj[:, :3]-crop_centre, norm_bbox)
							if check == 0: continue

							dist = np.min(l_dist)
							dir = (label_mean_r - np.array([0, 0, 0])) / dist

							pred_dummy = tf.Variable([0., 0., 0.])

							with tf.GradientTape() as tape:
								tape.watch(pred_dummy)
								loss = loss_object(tf.Variable(label_mean_r), pred_dummy)
							grad = tape.gradient(loss, pred_dummy).numpy()

							ret, tmp_room_crop = helpers.get_fixed_pts(tmp_room_crop, config['n_pts'])

							if ret == False: continue

							label = np.array([
								1,
								dist,
								grad[0],
								grad[1],
								grad[2],
								obj_ext[0],
								obj_ext[1],
								obj_ext[2],
								theta_r[0],
								theta_r[1]
							])

							# Uncomment to visualise training data
							# plot(tmp_room_crop[:, :3], tmp_room_crop[:, 3:6], -grad, label_mean_r, label)

							tf_example = create_example(tmp_room_crop[:, :3], tmp_room_crop[:, 3:6], label)

							assert tmp_room_crop.shape[0] == config['n_pts'], '{}'.format(tmp_room_crop.shape)

							if i != 5: train_writer.write(tf_example.SerializeToString())
							if i == 5: test_writer.write(tf_example.SerializeToString())

							pos_count += 1

					for j in range(config['n_negative_samples']):

						for _ in range(1000):

							crop_x = np.random.uniform(np.min(room_cloud[:, 0]), np.max(room_cloud[:, 0]))
							crop_y = np.random.uniform(np.min(room_cloud[:, 1]), np.max(room_cloud[:, 1]))
							crop_z = np.random.uniform(np.min(room_cloud[:, 2]), np.max(room_cloud[:, 2]))
							crop_point = np.array([crop_x, crop_y, crop_z])

							scene_bbox = np.array([
							crop_point[0] - (scene_crop_size[0]/2.), crop_point[1] - (scene_crop_size[1]/2.), crop_point[2] - (scene_crop_size[2]/2.),
							crop_point[0] + (scene_crop_size[0]/2.), crop_point[1] + (scene_crop_size[1]/2.), crop_point[2] + (scene_crop_size[2]/2.),
							])

							bbox = np.array([
								crop_point[0] - (crop_size[0]/2), crop_point[1] - (crop_size[1]/2), crop_point[2] - (crop_size[2]/2),
								crop_point[0] + (crop_size[0]/2), crop_point[1] + (crop_size[1]/2), crop_point[2] + (crop_size[2]/2),
							])

							check = np.array([helpers.check_occupancy(obj_pts, bbox) for obj_pts in objects])
							check = check.astype(np.bool)
							if True in check: continue

							norm_bbox = np.array([
								bbox[0]-crop_point[0], bbox[1]-crop_point[1], bbox[2]-crop_point[2],
								bbox[3]-crop_point[0], bbox[4]-crop_point[1], bbox[5]-crop_point[2]
							])

							for _ in range(config['n_rotations_per_object']):

								tmp_room_cloud = helpers.crop_bbox(room_cloud, scene_bbox)
								ret, tmp_room_cloud, = helpers.get_fixed_pts(tmp_room_cloud, config['n_scene_pts'])
								room_crop = helpers.crop_bbox(tmp_room_cloud, bbox)
								room_crop[:, :3] -= crop_point[:3]

								tmp_room_crop = room_crop.copy()

								theta = np.random.uniform() * 2.0 * np.pi
								tmp_room_crop[:, :3] = helpers.rotate_euler(tmp_room_crop[:, :3], theta)

								tmp_room_crop = helpers.crop_bbox(tmp_room_crop, norm_bbox)
								ret, tmp_room_crop = helpers.get_fixed_pts(tmp_room_crop, config['n_pts'])

								null_value = 0.0

								label = np.array([
									0,
									10,
									null_value,
									null_value,
									null_value,
									null_value,
									null_value,
									null_value,
									null_value,
									null_value
								])

								tf_example = create_example(tmp_room_crop[:, :3], tmp_room_crop[:, 3:6], label)

								assert tmp_room_crop.shape[0] == config['n_pts'], '{}'.format(tmp_room_crop.shape[0].shape)

								if i != 5: train_writer.write(tf_example.SerializeToString())
								if i == 5: test_writer.write(tf_example.SerializeToString())

								neg_count += 1

							break
	bar.finish()

	print('[info] total scenes saved: {}, {} positive, {} negative'.format(
		pos_count+neg_count, pos_count, neg_count)
	)


if __name__ == '__main__':

	config = {
		'root_dir' : './data/Stanford3dDataset',
		'out_train_file' : './data/s3d_patches_train.tfrecord',
		'out_test_file' : './data/s3d_patches_test.tfrecord',
		'object' : 'chair',
		'crop_size' : (1., 1., 1.),
		'n_pts' : 4096,
		'scene_crop_size' : (3., 3., 3.),
		'n_scene_pts' : 32768,
		'train_ratio' : 1.,
		'n_crops_per_object' : 20,
		'n_rotations_per_object' : 20,
		'n_negative_samples' : 20
	}

	generate_random_crops(config)
