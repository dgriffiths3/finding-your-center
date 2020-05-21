import numpy as np
import tensorflow as tf

from loss_network.model import LossNetModel


def float_list_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def fixed_crop3d(pts, boxes, centres, n_pts=4096):

	crops = []

	for b, batch in enumerate(boxes.numpy()):

		batch_crops = []

		for i, bbox in enumerate(batch):

			c = pts.numpy()[b][
				(pts[b][:, 0] >= bbox[0]) & (pts[b][:, 0] <= bbox[3]) &
				(pts[b][:, 1] >= bbox[1]) & (pts[b][:, 1] <= bbox[4]) &
				(pts[b][:, 2] >= bbox[2]) & (pts[b][:, 2] <= bbox[5])
			]

			if len(c) == 0:
				batch_crops.append(np.zeros((n_pts, 3), np.float32))

			elif c.shape[0] < n_pts / 2:
				c -= centres[b, i].numpy()
				dup_idx = np.arange(c.shape[0])
				np.random.shuffle(dup_idx)
				dup_idx = dup_idx[0:n_pts-c.shape[0]]
				c = np.vstack((c, c[dup_idx]))
				batch_crops.append(np.pad(c, [[0, n_pts-c.shape[0]],[0, 0]]))

			elif c.shape[0] < n_pts and c.shape[0] >= n_pts/2:
				c -= centres[b, i].numpy()
				dup_idx = np.arange(c.shape[0])
				np.random.shuffle(dup_idx)
				dup_idx = dup_idx[0:n_pts-c.shape[0]]
				batch_crops.append(np.vstack((c, c[dup_idx])))

			elif c.shape[0] > n_pts:
				c -= centres[b, i].numpy()
				s_idx = np.arange(0, c.shape[0])
				np.random.shuffle(s_idx)
				batch_crops.append(c[s_idx[0:n_pts]])

			else:
				batch_crops.append(c - centres[b, i].numpy())

		crops.append(batch_crops)

	return tf.constant(crops, dtype=tf.float32)


def tb_config(config):

	header_row = 'Attribute | Value'

	s = []

	for key in config.keys():
		s.append('{} | {}\n'.format(key, config[key]))
	for key in config.keys():
		s.append('{} | {}\n'.format(key, config[key]))

	table_rows = tf.strings.join(s)
	table_body = tf.strings.reduce_join(inputs=table_rows, separator='\n')
	table = tf.strings.join([header_row, "---|---", table_body], separator='\n')

	return table


def get_loss_model(config):

	batch_size = config['batch_size'] * config['n_pred']
	obj_size = config['loss_size']
	num_points = config['loss_points']

	loss_model = LossNetModel(batch_size)
	loss_model(
		tf.zeros((batch_size, num_points, 3), dtype=tf.float32)
	)

	loss_model.load_weights(config['loss_weights'])

	return loss_model


def box_offset(boxes, offset=0.0625, sign='minus'):

	if sign == 'add':
		offset_arr = np.zeros((boxes.shape))
		offset_arr[:, :, 0:3] = -offset
		offset_arr[:, :, 3:5] = offset
	elif sign == 'minus':
		offset_arr = np.zeros((boxes.shape))
		offset_arr[:, :, 0:3] = offset
		offset_arr[:, :, 3:5] = -offset
	else:
		raise ValueError("Sign must be either 'add' or 'minus'.")

	return tf.add(boxes, tf.constant(offset_arr, tf.float32))


def print_summary(model, config):

	input_shape = (
		config['batch_size'],
		config['num_points'],
		3
	)

	print('------------------------------------------')
	print('Batch size: {}'.format(input_shape[0]))
	print('Learning rate: {}'.format(config['lr']))
	print('Log directory: {}'.format(config['log_dir']))
	print('Log name: {}'.format(config['log_code']))
	print('------------------------------------------') 


def nms(batch_points, batch_boxes, batch_scores, max_out=100, iou_thresh=0.5, scores_thresh=0.5):
	"""
	Code adapted from : https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
	"""

	output_inds = []

	for b, (points, boxes, scores) in enumerate(zip(batch_points.numpy(), batch_boxes.numpy(), batch_scores.numpy())):

		x1 = boxes[:, 0]
		y1 = boxes[:, 1]
		z1 = boxes[:, 2]

		x2 = boxes[:, 3]
		y2 = boxes[:, 4]
		z2 = boxes[:, 5]

		keep_inds = []

		scores = np.expand_dims(scores[scores > scores_thresh], -1)

		if scores.shape[0] > 1:

			order = np.argsort(-scores, axis=0)

			areas = (x2 - x1) * (y2 - y1) * (z2 - z1)

			num_in = 0

			while order.size > 0:

				if num_in == max_out: break

				i = order[0]

				keep_inds.append(i[0])

				num_in += 1

				xx1 = np.maximum(x1[i], x1[order[1:]])
				yy1 = np.maximum(y1[i], y1[order[1:]])
				zz1 = np.maximum(z1[i], z1[order[1:]])

				xx2 = np.minimum(x2[i], x2[order[1:]])
				yy2 = np.minimum(y2[i], y2[order[1:]])
				zz2 = np.minimum(z2[i], z2[order[1:]])

				w = np.maximum(0.0, xx2 - xx1)
				h = np.maximum(0.0, yy2 - yy1)
				d = np.maximum(0.0, zz2 - zz1)

				inter = w * h * d
				ovr = inter / (areas[i] + areas[order[1:]] - inter)

				inds = np.where(ovr <= iou_thresh)[0]
				order = order[inds + 1]

			output_inds.append(np.array(keep_inds))

	return output_inds


def iou_matrix_2d(pred, clip_border):
	"""
	Code adapted from : https://github.com/tensorflow/models/blob/master/research/object_detection/core/post_processing.py
	"""

	boxes = pred_to_boxes(pred, clip_border)

	x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = tf.split(
		value=boxes, num_or_size_splits=6, axis=2)
	x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = tf.split(
		value=boxes, num_or_size_splits=6, axis=2)

	# Calculates the intersection area.
	intersection_xmin = tf.maximum(x1_min, tf.transpose(x2_min, [0, 2, 1]))
	intersection_xmax = tf.minimum(x1_max, tf.transpose(x2_max, [0, 2, 1]))
	intersection_ymin = tf.maximum(y1_min, tf.transpose(y2_min, [0, 2, 1]))
	intersection_ymax = tf.minimum(y1_max, tf.transpose(y2_max, [0, 2, 1]))

	intersection_area = tf.maximum((intersection_xmax - intersection_xmin), 0) * \
						tf.maximum((intersection_ymax - intersection_ymin), 0)

	area1 =  (x1_max - x1_min) * (y1_max - y1_min)
	area2 =  (x2_max - x2_min) * (y2_max - y2_min)
	union_area = area1 + tf.transpose(area2, [0, 2, 1]) - intersection_area + 1e-8

	iou = intersection_area / union_area

	padding_mask = tf.logical_and(
		tf.less(intersection_xmax, 0),
		tf.less(intersection_ymax, 0)
	)

	ones = tf.ones_like(iou)
	zeros = tf.zeros_like(iou)

	lower_mask = tf.linalg.band_part(ones, 0, -1)
	diag_mask = tf.linalg.band_part(ones, 0, 0)
	zero_mask = tf.cast(tf.where(iou > 0, ones, zeros), tf.bool)
	mask = tf.cast(lower_mask - diag_mask, dtype=tf.bool)
	mask = tf.logical_and(mask, zero_mask)
	inds = tf.where(mask)

	loss = tf.maximum(0.0, tf.reduce_sum(iou - tf.eye(boxes.shape[1])))

	return iou, inds, loss


def gen_iou_mask(pred, iou, inds, lossnet_results, mask_dist=.5, conf_thresh=0.9):
	"""
	Returns mask of size [b, n, 3] where loss network gradients should be masked.
	"""
	mask = []

	for i in range(iou.shape[0]):

		batch_mask = np.ones(pred.shape[1], dtype=np.bool)

		arr = tf.gather(inds, tf.reshape(tf.where(inds[:, 0]==i), ([-1])), axis=0)[:, 1:].numpy()

		if len(arr) > 0:
			confs = tf.stack([
				tf.gather(lossnet_results[i, :, -1], arr[:, 0]),
				tf.gather(lossnet_results[i, :, -1], arr[:, 1])
			], axis=1)

			l_and = np.logical_and(confs[:, 0]>conf_thresh, confs[:, 1]>conf_thresh)

			# Any values with conf < conf_thresh automatically added as valid iou's (mask to loss gradient)
			conf_mask = np.unique(
				np.append(
					np.unique(arr[:, 0][(confs[:, 0]<=conf_thresh).numpy()]),
					np.unique(arr[:, 1][(confs[:, 1]<=conf_thresh).numpy()])
				)
			)

			arr = arr[l_and]
			confs = confs[l_and]

			dirs = tf.stack([
				tf.gather(lossnet_results[i, :, :3], arr[:, 0]),
				tf.gather(lossnet_results[i, :, :3], arr[:, 1])
			], axis=-1) * -1

			preds = tf.stack([
				tf.gather(pred[i, :, :3], arr[:, 0]),
				tf.gather(pred[i, :, :3], arr[:, 1])
			], axis=-1)

			# Predicted destination of gradient (not absolute until close)
			pred_dest = preds + dirs

			# Calculate distance between overlapping destinations (2D ONLY)
			dists = tf.sqrt(tf.reduce_sum(tf.square(pred_dest[:, :2, 0] - pred_dest[:, :2, 1]), axis=-1))
			d_mask = tf.where(dists < mask_dist, tf.ones_like(dists, dtype=tf.bool), tf.zeros_like(dists, dtype=tf.bool))

			arr = arr[tf.expand_dims(d_mask, 0)]

			# Get new set of loss network gradients and calculate vector length
			loss_vec = tf.stack([
				tf.gather(lossnet_results[i, :, :3], arr[:, 0]),
				tf.gather(lossnet_results[i, :, :3], arr[:, 1])
			], axis=0)

			lvec_l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.zeros_like(loss_vec) - loss_vec), axis=-1))

			# Compare pairs, and take the larger vector length for masking
			valid = [arr[n, i] for n, i in enumerate(tf.math.argmax(lvec_l2, axis=0))]

			# Set masks, True means allow loss network gradient, False means mask with zeros.
			batch_mask[valid] = False
			batch_mask[conf_mask] = False

		mask.append(batch_mask)

	return tf.reshape(tf.constant(mask), (pred.shape[0], pred.shape[1], 1))


def objectness_mask(xyz, scores, exts, thresh):

	mask = tf.where(scores>=thresh)[:, 1]

	if len(mask) == 0:
		mask = tf.argmax(scores, axis=1)[0]

	xyz = tf.gather(xyz, mask, batch_dims=0, axis=1)
	scores = tf.gather(scores, mask, batch_dims=0, axis=1)
	exts = tf.gather(exts, mask, batch_dims=0, axis=1)

	if tf.rank(xyz) == 2:
		xyz = tf.expand_dims(xyz, 0)
		scores = tf.expand_dims(scores, 0)
		exts = tf.expand_dims(exts, 0)

	return xyz, scores, exts


def pred_to_boxes(pred, offset):

	boxes = tf.reshape(pred,(-1, 3))
	min = tf.subtract(boxes, offset/2.)
	max = tf.add(boxes, offset/2.)
	boxes = tf.reshape(tf.stack([min, max], axis=1), (pred.shape[0], pred.shape[1], 6))

	return boxes
