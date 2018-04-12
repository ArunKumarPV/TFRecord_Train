from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import glob
import _pickle as cPickle

import numpy as np
import tensorflow as tf

def readFromTFRecords(filename, batch_size, num_epochs, img_shape, num_threads=2, min_after_dequeue=1000):
    """
    Args:
        filename: the .tfrecords file we are going to load
        batch_size: batch size
        num_epoch: number of epochs, 0 means train forever
        img_shape: image shape: [height, width, channels]
        num_threads: number of threads
        min_after_dequeue: defines how big a buffer we will randomly sample from,
            bigger means better shuffling but slower start up and more memory used.
            (capacity is usually min_after_dequeue + (num_threads + eta) * batch_size)
    
    Return:
        images: (batch_size, height, width, channels)
        labels: (batch_size)
    """
    def read_and_decode(filename_queue, img_shape):
        """Return a single example for queue"""
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
                serialized_example,
                features={
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64),
                }
        )
        # some essential steps
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, img_shape)    # THIS IS IMPORTANT
        image.set_shape(img_shape)
        image = tf.cast(image, tf.float32) * (1 / 255.0)  # set to [0, 1]

        sparse_label = tf.cast(features['label'], tf.int32)
        #print(image[0,:,:].shape)
        #print(sparse_label[0].shape)
        return image, sparse_label

    filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

    image, sparse_label = read_and_decode(filename_queue, img_shape) # share filename_queue with multiple threads
    
    # tf.train.shuffle_batch internally uses a RandomShuffleQueue
    images, sparse_labels = tf.train.shuffle_batch(
            [image, sparse_label], batch_size=batch_size, num_threads=num_threads,
            min_after_dequeue=min_after_dequeue,
            capacity=min_after_dequeue + (num_threads + 1) * batch_size
    )
    #print(images[:,:,:].shape)
    #print(sparse_labels[:].shape)
    return images, sparse_labels    

def convertToTFRecords(images, labels, num_examples, filename):
    """
    Args:
        images: (num_examples, height, width, channels) np.int64 nparray (0~255)
        labels: (num_examples) np.int64 nparray
        num_examples: number of examples
        filename: the tfrecords' name to be saved

    Return: None, but store a .tfrecords file to data_log/
    """
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    writer = tf.python_io.TFRecordWriter(os.path.join('data_log', filename))
    for index in xrange(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[rows])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[cols])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[depth])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[index]])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

def read_data_batches(train=True):
    """Currently read cifar-10 as numpy image files to simulate ordinary file reading operation
    read pixels in np.uint8: 0~255, labels in np.int32
    THE DTYPE IS IMPORTANT FOR read_and_decode() ABOVE!!
    Return:
        images: (50000, 32, 32, 3) for train and (10000, 32, 32, 3) for test
        labels: (50000) for train and (10000) for test
    need to download and extract cifar10 to data_log/ folder

    You can modify this to load any data you want, as long as the return shape is:
        images: (num_examples, height, width, channels)
        labels: (num_examples)
    """
    if train:
        print ('Reading training batches')
        batches = [cPickle.load(open('data_log/cifar-10-batches-py/data_batch_'+str(b+1))) for b in xrange(5)]
    else:
        print ('Reading test batch')
        batches = [cPickle.load(open('data_log/cifar-10-batches-py/test_batch', 'rb'))]

    images = np.zeros((50000, 32, 32, 3), dtype=np.uint8) if train else np.zeros((10000, 32, 32, 3), dtype=np.uint8)
    labels = np.zeros((50000), dtype=np.int32) if train else np.zeros((10000), dtype=np.int32)
    for i, b in enumerate(batches):
        for j, l in enumerate(b['labels']):
            images[i*10000 + j] = b['data'][j].reshape([3, 32, 32]).transpose([2, 1, 0]).transpose(1, 0, 2)
            labels[i*10000 + j] = l

    return images, labels


class Queue_loader():
    # This queue loader use cifar10 as example data
    def __init__(self, batch_size, num_epochs, num_threads=2, min_after_dequeue=1000, train=True):
        if train:
            
            filename = '/home/ai/BVe-G/NewTf_rcrd/bv2_15_data_train.tfrecords'
        else:
            filename = '/home/ai/BVe-G/Backup/TensorFlowCode/TFRecord/bv2_15_data_val.tfrecords'

        # First, we are going to generate a single file which contains both training images and labels 
        #  in standard tensorflow file format (TFRecords), this is simple
        #if not os.path.exists(os.path.join('data_log', filename)):
         #   images, labels = read_data_batches(train)
          #  convertToTFRecords(images, labels, len(images), filename)

        img_shape = [256, 256, 1]
        self.num_examples = 119226 if train else 9019
        # the above 2 lines are set manually assuming we have already generated .tfrecords file
        self.num_batches = int(self.num_examples / batch_size)
        
        # Second, we are going to read from .tfrecords file, this contains several steps
        self.images, self.labels = readFromTFRecords(os.path.join(filename), batch_size, num_epochs,
                img_shape, num_threads, min_after_dequeue)

        # done
