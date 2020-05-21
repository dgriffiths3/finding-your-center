import math
import colorsys
import cv2 as cv
import numpy as np
import progressbar
import pyvista as pv
import tensorflow as tf


def progbar(n):

	bar = progressbar.ProgressBar(
		maxval=n,
		widgets=[
			progressbar.Bar('=', '[', ']'), ' ',
			progressbar.Percentage(), ' | ',
			progressbar.SimpleProgress(), ' | ',
			progressbar.AdaptiveETA()
		]
	)

	return bar


def crop_bbox(in_pc, bbox):

	out_pc = in_pc[
		(in_pc[:, 0] >= bbox[0]) & (in_pc[:, 0] <= bbox[3]) &
		(in_pc[:, 1] >= bbox[1]) & (in_pc[:, 1] <= bbox[4]) &
		(in_pc[:, 2] >= bbox[2]) & (in_pc[:, 2] <= bbox[5])
	]

	return out_pc


def check_occupancy(obj_points, bbox):

	occupied_pts = obj_points[
		(obj_points[:, 0]>= bbox[0]) & (obj_points[:, 0]<= bbox[3]) &
		(obj_points[:, 1]>= bbox[1]) & (obj_points[:, 1]<= bbox[4]) &
		(obj_points[:, 2]>= bbox[2]) & (obj_points[:, 2]<= bbox[5])
	]

	return occupied_pts.shape[0]


def euc_dist(a, b):

	return np.sqrt(np.sum((a - b)**2))


def bounding_box(pc):
	""" Calculate bounding box as [xmin, ymin, ..., zmax] """

	bbox = [
		np.min(pc[:, 0]), np.min(pc[:, 1]), np.min(pc[:, 2]),
		np.max(pc[:, 0]), np.max(pc[:, 1]), np.max(pc[:, 2])
	]

	return np.array(bbox)


def bbox_overlap(pc_a, pc_b):

	bbox_a = bounding_box(pc_a)
	bbox_b = bounding_box(pc_b)

	if (
		bbox_a[3] >= bbox_b[0] and bbox_b[3] >= bbox_a[0] and
		bbox_a[4] >= bbox_b[1] and bbox_b[4] >= bbox_a[1] and
		bbox_a[5] >= bbox_b[2] and bbox_b[5] >= bbox_a[2]
		):
		overlap = True

	else:

		overlap = False

	return overlap


def rotate_boxes(boxes, centers, theta):

	pts_out = np.zeros((boxes.shape[0], 8, 3), np.float32)

	for i, (b, c, r) in enumerate(zip(boxes, centers, theta)):

		t = np.arctan2(r[1], r[0])

		pts = np.array([
			[b[0], b[1], b[2]],
			[b[3], b[1], b[2]],
			[b[3], b[4], b[2]],
			[b[0], b[4], b[2]],
			[b[0], b[1], b[5]],
			[b[3], b[1], b[5]],
			[b[3], b[4], b[5]],
			[b[0], b[4], b[5]]
		])

		for j, p in enumerate(pts):
			pts_out[i, j][0] = c[0] + ((p[1]-c[1])*np.sin(t) + (p[0]-c[0])*np.cos(t))
			pts_out[i, j][1] = c[1] + ((p[1]-c[1])*np.cos(t) - (p[0]-c[0])*np.sin(t))
			pts_out[i, j][2] = p[2]

	return pts_out


def rotate_box(b, c, r):

	pts_out = np.zeros((8, 3), np.float32)

	t = np.arctan2(r[1], r[0])

	pts = np.array([
		[b[0], b[1], b[2]],
		[b[3], b[1], b[2]],
		[b[3], b[4], b[2]],
		[b[0], b[4], b[2]],
		[b[0], b[1], b[5]],
		[b[3], b[1], b[5]],
		[b[3], b[4], b[5]],
		[b[0], b[4], b[5]]
	])

	for j, p in enumerate(pts):
		pts_out[j][0] = c[0] + ((p[1]-c[1])*np.sin(t) + (p[0]-c[0])*np.cos(t))
		pts_out[j][1] = c[1] + ((p[1]-c[1])*np.cos(t) - (p[0]-c[0])*np.sin(t))
		pts_out[j][2] = p[2]

	return pts_out


def make_lines(pts):

	lines = [
		pv.Line(pts[0], pts[1]), pv.Line(pts[1], pts[2]), pv.Line(pts[2], pts[3]), pv.Line(pts[3], pts[0]),
		pv.Line(pts[4], pts[5]), pv.Line(pts[5], pts[6]), pv.Line(pts[6], pts[7]), pv.Line(pts[7], pts[4]),
		pv.Line(pts[0], pts[4]), pv.Line(pts[1], pts[5]), pv.Line(pts[2], pts[6]), pv.Line(pts[3], pts[7]),
	]

	return lines


def rotate_euler(in_pc, theta):

	out_pc = np.zeros_like((in_pc))

	cosval = np.cos(theta)
	sinval = np.sin(theta)

	R = np.array([
		[np.cos(theta), -np.sin(theta), 0.0],
		[np.sin(theta), np.cos(theta), 0.0],
		[0.0, 0.0, 1.0]
	])

	for idx, p in enumerate(in_pc):
		out_pc[idx] = np.dot(R, p)

	return out_pc


def get_fixed_pts(in_pts, n_pts):

	out_pts = in_pts.copy()

	ret = True

	if out_pts.shape[0] == 0:
		out_pts = np.zeros((n_pts, 6))
		ret = False
	elif out_pts.shape[0] < n_pts:
		dup_idx = np.arange(out_pts.shape[0])
		np.random.shuffle(dup_idx)
		dup_idx = dup_idx[0:n_pts-out_pts.shape[0]]
		out_pts = np.vstack((
			out_pts,
			out_pts[dup_idx]
		))
		if in_pts.shape[0] < n_pts/2:
			out_pts = np.pad(out_pts, [[0, n_pts-out_pts.shape[0]], [0, 0]])
	else:
		s_idx = np.arange(out_pts.shape[0])
		np.random.shuffle(s_idx)
		out_pts = out_pts[s_idx[0:n_pts]]

	return ret, out_pts


def iou(a, b):

	xx1 = np.maximum(a[0], b[0])
	yy1 = np.maximum(a[1], b[1])
	zz1 = np.maximum(a[2], b[2])

	xx2 = np.minimum(a[3], b[3])
	yy2 = np.minimum(a[4], b[4])
	zz2 = np.minimum(a[5], b[5])

	w = np.maximum(0.0, xx2 - xx1)
	h = np.maximum(0.0, yy2 - yy1)
	d = np.maximum(0.0, zz2 - zz1)

	inter = w * h * d

	area_a = (a[3] - a[0]) * (a[4] - a[1]) * (a[5] - a[2])
	area_b = (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2])

	return inter / float(area_a + area_b - inter)


def make_boxes(xyz, extent):

	out_boxes = np.zeros((extent.shape[0], extent.shape[1], 6))

	for b in range(xyz.shape[0]):

		centers = xyz[b, :, :3]
		hwd = extent[b, :, :3]
		theta = extent[b, :, 3:5]

		boxes_min = centers - (hwd / 2.)
		boxes_max = centers + (hwd / 2.)
		boxes = np.hstack((boxes_min, boxes_max))

		boxes = rotate_boxes(boxes, centers, theta)

		for i, box in enumerate(boxes):
			out_boxes[b, i, 0] = np.min(box[:, 0])
			out_boxes[b, i, 1] = np.min(box[:, 1])
			out_boxes[b, i, 2] = np.min(box[:, 2])
			out_boxes[b, i, 3] = np.max(box[:, 0])
			out_boxes[b, i, 4] = np.max(box[:, 1])
			out_boxes[b, i, 5] = np.max(box[:, 2])

	return tf.constant(out_boxes)


def get_oabb(objects):

	oabb = np.zeros((objects.shape[0], 2), np.float32)
	extents = np.zeros((objects.shape[0], 3), np.float32)

	for i, obj in enumerate(objects):
		xy = (obj[:, :2].reshape(-1, 1, 2) * 1000).astype(np.int)
		mar = cv.minAreaRect(xy)
		xyz_ext = np.array(mar[1]).astype(np.float32) / 1000
		h = np.array([np.max(obj[:, 2]) - np.min(obj[:, 2])])
		xyz_ext = np.append(xyz_ext, h, -1)
		rot_angle = np.radians(-mar[2])
		oabb[i] = [np.cos(rot_angle), np.sin(rot_angle)]
		extents[i] = xyz_ext

	return oabb, extents

