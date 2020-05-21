import numpy as np
import tensorflow as tf

from utils import tf_utils


def mean_squared_error(labels, pred):

	return tf.keras.losses.MeanSquaredError()(labels, pred)


def nn_distance(a, b):

    n = a.shape[1]
    m = b.shape[1]

    a_tile = tf.expand_dims(a, 2)
    b_tile = tf.expand_dims(b, 1)

    a_tile = tf.tile(a_tile, [1, 1, m, 1])
    b_tile = tf.tile(b_tile, [1, n, 1, 1])

    diff = tf.reduce_sum(tf.square(a_tile-b_tile), -1)

    dist1 = tf.reduce_min(diff, 2)
    idx1 = tf.argmin(diff, 2)

    dist2 = tf.reduce_min(diff, 1)
    idx2 = tf.argmin(diff, 1)

    return dist1, idx1, dist2, idx2


def chamfer_distance(labels, pred):

	loss = np.zeros((pred.shape[0]))

	for b in range(labels.shape[0]):

		gt = labels[b]
		pts = tf.expand_dims(pred[b], 0)

		gt_mask = tf.cast(gt, tf.bool)[:, 0]
		gt = tf.boolean_mask(gt, gt_mask)
		gt_pts = tf.expand_dims(gt[:, :3], 0)

		gt_p_dist, gt_p_idx, p_gt_dist, p_gt_idx = nn_distance(gt_pts, pts)

		chamfer_loss = tf.reduce_mean(gt_p_dist) + tf.reduce_mean(p_gt_dist)

		loss[b] = chamfer_loss

	return tf.reduce_mean(loss)


@tf.custom_gradient
def chamfer_loss(labels, pred):

	def grad(dy, *args): 
		return None, grads

	grads = np.zeros_like(pred, np.float32)
	loss = np.zeros((pred.shape[0]))

	for b in range(labels.shape[0]):

		gt = labels[b]
		pts = tf.expand_dims(pred[b], 0)

		gt_mask = tf.cast(gt, tf.bool)[:, 0]
		gt = tf.boolean_mask(gt, gt_mask)
		gt_pts = tf.expand_dims(gt[:, :3], 0)

		with tf.GradientTape(persistent=True) as t:
			t.watch([pts])

			gt_p_dist, gt_p_idx, p_gt_dist, p_gt_idx = nn_distance(gt_pts, pts)

			chamfer_loss = tf.reduce_mean(gt_p_dist) + tf.reduce_mean(p_gt_dist)

		grads[b] = t.gradient(chamfer_loss, pts)[0]
		loss[b] = chamfer_loss

	grads = tf.constant(grads)

	return (tf.reduce_mean(loss)), grad


@tf.custom_gradient
def supervised_loss(labels, pred_points, pred_scores, pred_exts, patch_size=1., supervised_grads=True):

	def grad(dy, *args): 
		return None, grad_points, grad_scores, grad_exts

	batch_size = pred_points.shape[0]
	max_dist = tf.math.sqrt(((patch_size / 2.)**2)*3)

	loss_MSE = tf.keras.losses.MeanSquaredError()
	loss_BCE = tf.keras.losses.BinaryCrossentropy()

	loss_points = np.zeros(batch_size, np.float32)
	loss_scores = np.zeros(batch_size, np.float32)
	loss_exts = np.zeros(batch_size, np.float32)

	grad_points = np.zeros_like(pred_points, np.float32)
	grad_scores = np.zeros_like(pred_scores, np.float32)
	grad_exts = np.zeros_like(pred_exts, np.float32)

	for b in range(batch_size):

		gt = labels[b]
		pts = tf.expand_dims(pred_points[b], 0)
		scores = pred_scores[b]
		exts = pred_exts[b]

		gt_mask = tf.cast(gt, tf.bool)[:, 0]
		gt = tf.boolean_mask(gt, gt_mask)
		gt_pts = tf.expand_dims(gt[:, :3], 0)
		gt_ext = gt[:, 3:8]

		with tf.GradientTape(persistent=True) as t:
			t.watch([pts, scores, exts])

			gt_p_dist, gt_p_idx, p_gt_dist, p_gt_idx = nn_distance(gt_pts, pts)

			if supervised_grads:
				chamfer_loss = (tf.reduce_sum(gt_p_dist) + tf.reduce_sum(p_gt_dist)) * 0.1
			else:
				chamfer_loss = tf.reduce_mean(gt_p_dist)

			euc_dists = tf.sqrt(p_gt_dist + 1e-8)
			score_labels = np.zeros((euc_dists.shape[1]), np.float32)
			score_labels[euc_dists<max_dist] = 1
			score_labels_one_hot = tf.one_hot(tf.cast(score_labels, tf.int64), depth=2)
			loss_score = loss_BCE(score_labels_one_hot, scores)

			ext_labels = tf.gather(gt_ext, p_gt_idx, 1)
			ext_labels = tf.where(tf.cast(tf.expand_dims(score_labels, -1), tf.bool), ext_labels, tf.zeros_like(ext_labels))
			loss_ext = loss_MSE(ext_labels, exts)

		grad_points[b] = t.gradient(chamfer_loss, pts)[0]
		grad_scores[b] = t.gradient(loss_score, scores)
		grad_exts[b] = t.gradient(loss_ext, exts)

		loss_points[b] = chamfer_loss
		loss_scores[b] = loss_score
		loss_exts[b] = loss_ext

	grad_points = tf.constant(grad_points)
	grad_scores = tf.constant(grad_scores)
	grad_exts = tf.constant(grad_exts)

	loss_points = tf.reduce_mean(loss_points)
	loss_scores = tf.reduce_mean(loss_scores)
	loss_exts = tf.reduce_mean(loss_exts)

	return (loss_points, loss_scores, loss_exts, tf.expand_dims(score_labels, 0)), grad


@tf.custom_gradient
def unsupervised_loss(pred_points, pred_scores, pred_exts, lossnet_results, conf_thresh=0.9):

	def grad(dy, variables=None, *args):
		return grads_p, grads_d, grads_e, None

	loss_MSE = tf.keras.losses.MeanSquaredError()
	loss_BCE = tf.keras.losses.BinaryCrossentropy()

	grads = lossnet_results[:, :, 0:3]

	iou, inds, _ = tf_utils.iou_matrix_2d(pred_points, clip_border=0.5)
	iou_mask = tf_utils.gen_iou_mask(pred_points, iou, inds, lossnet_results, mask_dist=0.5, conf_thresh=conf_thresh)
	grads = tf.where(iou_mask == True, grads, tf.zeros_like(pred_points))

	lossnet_results = tf.concat([grads, lossnet_results[:, :, 3:]], -1)
	conf = tf.reshape(lossnet_results[:, :, 9], (pred_points.shape[0], -1, 1))

	score_labels = lossnet_results[:, :, 8:10]
	score_labels = tf.where(score_labels > conf_thresh, tf.ones_like(score_labels), tf.zeros_like(score_labels))
	score_labels = tf.where(iou_mask == True, score_labels, tf.zeros_like(score_labels))
	ext_labels = tf.where(conf > conf_thresh, lossnet_results[:, :, 3:8], tf.zeros_like(pred_exts))

	with tf.GradientTape(persistent=True) as grad_tape:
		grad_tape.watch([pred_scores, pred_exts])
		scores_loss = loss_BCE(score_labels, pred_scores)
		exts_loss = loss_MSE(ext_labels, pred_exts)

	grads_p = tf.where(conf >= conf_thresh, grads, tf.zeros_like(pred_points))
	grads_d = grad_tape.gradient(scores_loss, pred_scores)
	grads_e = grad_tape.gradient(exts_loss, pred_exts)
	grads_e = tf.where(conf > conf_thresh, grads_e, tf.zeros_like(pred_exts))

	return (scores_loss, exts_loss, iou_mask), grad


@tf.custom_gradient
def batch_iou_loss_2d(pred, lossnet_results, iou_mask):

	def grad(dy, variables=None): 
		return grads, None, None

	with tf.GradientTape(persistent=True) as t:

		t.watch([pred])
		# clip_border should match corresponding value in unsupervised_loss()
		iou, inds, loss = tf_utils.iou_matrix_2d(pred, clip_border=0.5)
		
	grads = t.gradient(loss, pred)
	grads = tf.where(iou_mask == False, grads, tf.zeros_like(grads))

	return loss, grad
