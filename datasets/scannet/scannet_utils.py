import json
import plyfile

import numpy as np


def load_segmap(path):

	with open(path) as jsondata:
		d = json.load(jsondata)
		seg = d['segIndices']

	obj_to_segmap = {}

	for i in range(len(seg)):
		if seg[i] not in obj_to_segmap:
			obj_to_segmap[seg[i]] = []
		obj_to_segmap[seg[i]].append(i)

	return obj_to_segmap


def load_pointcloud(path):

	ply = plyfile.PlyData.read(path)

	pc = np.zeros((ply.elements[0].data.shape[0], 7))
	pc[:, 0] = ply.elements[0].data['x']
	pc[:, 1] = ply.elements[0].data['y']
	pc[:, 2] = ply.elements[0].data['z']
	pc[:, 3] = ply.elements[0].data['red']
	pc[:, 4] = ply.elements[0].data['green']
	pc[:, 5] = ply.elements[0].data['blue']
	pc[:, 6] = ply.elements[0].data['label']

	return pc


def load_objects(scene, agg, segmap, label_objects):

	objects = []
	object_means = []
	bounds = []

	for seg in agg['segGroups']:
		if seg['label'] in label_objects:
			obj = []
			for segment in seg['segments']:
				for id in segmap[segment]:
					obj.append(scene[id])
			obj = np.array(obj)
			objects.append(obj)
			bounds.append([
				np.min(obj[:, 0]),
				np.max(obj[:, 0]),
				np.min(obj[:, 1]),
				np.max(obj[:, 1]),
				np.min(obj[:, 2]),
				np.max(obj[:, 2]),
			])
			object_means.append(np.mean(obj[:, :3], axis=0))

	objects = np.array(objects)
	object_means = np.array(object_means)

	return objects, object_means
