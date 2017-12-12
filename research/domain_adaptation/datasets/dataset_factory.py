# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A factory-pattern class which returns image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf
import os
from slim.datasets import mnist
import mnist_m
from slim.datasets import dataset_utils
slim = tf.contrib.slim


def get_dataset(dataset_name,
                split_name,
                dataset_dir,
                file_pattern=None,
                reader=None):
  """Given a dataset name and a split_name returns a Dataset.

  Args:
    dataset_name: String, the name of the dataset.
    split_name: A train/test split name.
    dataset_dir: The directory where the dataset files are stored.
    file_pattern: The file pattern to use for matching the dataset source files.
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      reader defined by each dataset is used.

  Returns:
    A tf-slim `Dataset` class.

  Raises:
    ValueError: if `dataset_name` isn't recognized.
  """

# test model for office dataset
  if dataset_name in ['amazon', 'webcam', 'dslr']:    
    if dataset_utils.has_labels(dataset_dir, dataset_name+'_labels.txt'):
      labels_to_names = dataset_utils.read_label_file(dataset_dir, dataset_name+'_labels.txt')

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
      reader = tf.TFRecordReader

    _FILE_PATTERN = dataset_name+'_'+split_name+'.tfrecord'
    _NUM_CLASSES = 31

    if dataset_name == 'amazon':
      _SPLITS_TO_SIZES = {
        'train': 1972,
        'validation': 845
      }
    elif dataset_name == 'webcam':
      _SPLITS_TO_SIZES = {
          'train': 557,
          'validation': 238
        }
    
    _ITEMS_TO_DESCRIPTIONS = {
      'image': 'A colored image of varying height and width.',
      'label': 'The label id of the image, integer between 0 and 30.',
    }

    keys_to_features = {
      'image/encoded':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
          tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/class/label':
          tf.FixedLenFeature(
              [], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
    }

    items_to_handlers = {
      'image': slim.tfexample_decoder.Image(shape=[224,224,3], channels=3),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
      data_sources=os.path.join(dataset_dir,_FILE_PATTERN),
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      num_classes=_NUM_CLASSES,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      labels_to_names=labels_to_names)

  dataset_name_to_module = {'mnist': mnist, 'mnist_m': mnist_m}
  
  if dataset_name not in dataset_name_to_module:
    raise ValueError('Name of dataset unknown %s.' % dataset_name)
  
  return dataset_name_to_module[dataset_name].get_split(split_name, dataset_dir,
                                                        file_pattern, reader)


def provide_batch(dataset_name, split_name, dataset_dir, num_readers,
                  batch_size, num_preprocessing_threads):
  """Provides a batch of images and corresponding labels.

    Args:
    dataset_name: String, the name of the dataset.
    split_name: A train/test split name.
    dataset_dir: The directory where the dataset files are stored.
    num_readers: The number of readers used by DatasetDataProvider.
    batch_size: The size of the batch requested.
    num_preprocessing_threads: The number of preprocessing threads for
      tf.train.batch.
    file_pattern: The file pattern to use for matching the dataset source files.
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      reader defined by each dataset is used.

  Returns:
    A batch of
      images: tensor of [batch_size, height, width, channels].
      labels: dictionary of labels.
  """
  dataset = get_dataset(dataset_name, split_name, dataset_dir)
  print("####@@@@@@@________dataset_factory " + dataset_name + '_' + split_name)
  provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=num_readers,
      common_queue_capacity=20 * batch_size,
      common_queue_min=10 * batch_size)
  [image, label] = provider.get(['image', 'label'])

  # Convert images to float32
  image = tf.image.convert_image_dtype(image, tf.float32)
  image -= 0.5
  image *= 2

  print(image.shape)
  # Load the data.
  labels = {}
  images, labels['classes'] = tf.train.batch(
      [image, label],
      batch_size=batch_size,
      num_threads=num_preprocessing_threads,
      capacity=5 * batch_size)
  labels['classes'] = slim.one_hot_encoding(labels['classes'],
                                            dataset.num_classes)
  if dataset_name in ['amazon', 'webcam', 'dslr']:
    images = tf.image.resize_images(images, [224,224])

  # Convert mnist to RGB and 32x32 so that it can match mnist_m.
  if dataset_name == 'mnist':
    images = tf.image.grayscale_to_rgb(images)
    images = tf.image.resize_images(images, [32, 32])
  return images, labels
