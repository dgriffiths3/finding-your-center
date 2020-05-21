import os
import tensorflow as tf


def load_loss_dataset(in_file, batch_size, n_points=4096, n_labels=10, repeat=True):

    assert os.path.isfile(in_file), '[error] dataset path not found'
    
    shuffle_buffer = 1000

    def _extract_fn(data_record):

        in_features = {
            'points': tf.io.FixedLenFeature([n_points * 3], tf.float32),
            'colors': tf.io.FixedLenFeature([n_points * 3], tf.float32),
            'label': tf.io.FixedLenFeature([n_labels], tf.float32)
        }

        return tf.io.parse_single_example(data_record, in_features)

    def _preprocess_fn(sample):

        pts = sample['points']
        label = sample['label']

        pts = tf.reshape(pts, (n_points, 3))
        pts = tf.random.shuffle(pts)

        return pts, label

    dataset = tf.data.TFRecordDataset(in_file)
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(_extract_fn)
    dataset = dataset.map(_preprocess_fn)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if repeat == True: dataset = dataset.repeat()

    return dataset


def load_scene_dataset(in_file, batch_size, max_labels=25, n_points=32768, n_labels=8, repeat=True):

    assert os.path.isfile(in_file), '[error] dataset path not found'

    shuffle_buffer = 1000

    def _extract_fn(data_record):

        in_features = {
            'points': tf.io.FixedLenFeature([n_points * 3], tf.float32),
            'colors': tf.io.FixedLenFeature([n_points * 3], tf.float32),
            'labels': tf.io.FixedLenFeature([max_labels * n_labels], tf.float32),
            'n_inst': tf.io.FixedLenFeature([], tf.int64)
        }

        return tf.io.parse_single_example(data_record, in_features)

    def _preprocess_fn(sample):
         
        pts = sample['points']
        cols = sample['colors']
        labels = sample['labels']

        pts = tf.reshape(pts, (n_points, 3))
        cols = tf.reshape(cols, (n_points, 3))
        labels = tf.reshape(labels, (max_labels, n_labels))

        shuffle_idx = tf.range(n_points)
        shuffle_idx = tf.random.shuffle(shuffle_idx)

        pts = tf.gather(pts, shuffle_idx)
        cols = tf.gather(cols, shuffle_idx)

        return pts, cols, labels

    dataset = tf.data.TFRecordDataset(in_file)
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(_extract_fn)
    dataset = dataset.map(_preprocess_fn)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if repeat == True: dataset = dataset.repeat()

    return dataset
