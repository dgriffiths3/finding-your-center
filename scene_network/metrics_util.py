import tensorflow as tf


def get_metrics():

	total_m = tf.keras.metrics.Mean() # 0
	ext_m = tf.keras.metrics.Mean() # 1
	score_m = tf.keras.metrics.Mean() # 2
	iou_m = tf.keras.metrics.Mean() # 3

	sup_xyz_m = tf.keras.metrics.Mean() # 4
	sup_ext_m = tf.keras.metrics.Mean() # 5
	sup_score_m = tf.keras.metrics.Mean() # 6

	test_xyz_m = tf.keras.metrics.Mean() # 0
	test_ext_m = tf.keras.metrics.Mean() # 1
	test_score_m = tf.keras.metrics.Mean() # 2

	train_metrics = [
		total_m,
		ext_m,
		score_m,
		iou_m,
		sup_xyz_m,
		sup_ext_m,
		sup_score_m
	]

	test_metrics = [
		test_xyz_m,
		test_ext_m,
		test_score_m
	]

	return train_metrics, test_metrics


def update_train_metrics(metrics, *argv):

	for i in range(len(argv)):
		if argv[i] != 0: metrics[i].update_state([argv[i]])

	return metrics


def update_test_metrics(metrics, *argv):

	for i in range(len(argv)):
		if argv[i] != 0: metrics[i].update_state([argv[i]])

	return metrics


def write_summaries(summary_writer, pred, metrics, step, train_test='train'):


	with summary_writer.as_default():

		if train_test == 'train':

			tf.summary.scalar('total loss', metrics[0].result(), step=step)
			tf.summary.scalar('ext loss', metrics[1].result(), step=step)
			tf.summary.scalar('clf loss', metrics[2].result(), step=step)
			tf.summary.scalar('iou loss', metrics[3].result(), step=step)
			tf.summary.scalar('supervised xyz', metrics[4].result(), step=step)
			tf.summary.scalar('supervised ext', metrics[5].result(), step=step)
			tf.summary.scalar('supervised clf', metrics[6].result(), step=step)

			tf.summary.histogram('x', pred[:, :, 0], step=step)
			tf.summary.histogram('y', pred[:, :, 1], step=step)
			tf.summary.histogram('z', pred[:, :, 2], step=step)

		elif train_test == 'test':

			tf.summary.scalar('supervised xyz', metrics[0].result(), step=step)
			tf.summary.scalar('supervised ext', metrics[1].result(), step=step)
			tf.summary.scalar('supervised obj', metrics[2].result(), step=step)
