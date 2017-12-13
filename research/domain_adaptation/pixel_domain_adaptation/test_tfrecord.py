import tensorflow as tf
import numpy as np

data_path = '/home/wanlu/data/mnist_m.tfrecord'

with tf.Session() as sess:
	feature = {
		'train/images/': tf.FixedLenFeature([], tf.string),
		'train/labels/': tf.FixedLenFeature([], tf.int64)
	}

	filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)

	features = tf.parse_single_example(serialized_example, features=feature)

	image = tf.decode_raw(features['train/images/'], tf.float32)
	label = tf.cast(features['train/labels/'], tf.int32)

#	image = tf.reshape(image, [224,224,3])
	print(label.shape)
	print(image.shape)
	print("******Finish******")
