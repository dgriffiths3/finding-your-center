import numpy as np
import tensorflow as tf

from utils import losses, helpers


def calculate_ap(rec, prec):

	mrec = np.concatenate(([0.], rec, [1.]))
	mpre = np.concatenate(([0.], prec, [0.]))

	for i in range(mpre.size - 1, 0, -1):
		mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

	i = np.where(mrec[1:] != mrec[:-1])[0]

	ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

	return ap


def box_eval(pred_boxes, labels, labels_boxes, iou_thresh=0.25):

	results = np.zeros((pred_boxes.shape[0], 3))

	for b, (pred_batch, label_xyz_batch, label_batch) in enumerate(zip(pred_boxes.numpy(), labels.numpy(), labels_boxes.numpy())):

		label_batch = label_batch[label_xyz_batch.astype(np.bool)[:, 0]]

		tp = np.zeros(pred_batch.shape[0], dtype=np.float32)
		fp = np.zeros(pred_batch.shape[0], dtype=np.float32)

		labels_used = []

		npos = label_batch.shape[0]

		for p_idx, pred in enumerate(pred_batch):

			pos = False

			for l_idx, label in enumerate(label_batch):

				if l_idx in labels_used: continue

				iou_score = helpers.iou(label, pred)

				if iou_score >= iou_thresh:
					labels_used.append(l_idx)
					pos = True
					tp[p_idx] = 1.
					break

			if pos == False: fp[p_idx] = 1.
		
		tp = np.cumsum(tp)
		fp = np.cumsum(fp)
		rec = tp / float(npos)
		prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

		ap = calculate_ap(rec, prec)

		m_prec = prec[-1]
		m_rec = rec[-1]

		results[b] = [m_prec, m_rec, ap]

	return results