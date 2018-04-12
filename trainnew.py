import yaml
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import time
import os
import sys
from datetime import timedelta
import skimage.io as io
from queue_loader import Queue_loader
from tensorflow.python.framework.graph_util import convert_variables_to_constants


with open('model_bv2_15.yaml') as info:
    model_info = yaml.load(info)

img_height = img_width = 256
# for gray images
num_channels = 1
# num of output
num_labels = model_info['num_output']
# model name
model_name = model_info['model_name']




def init_conv_weights(shape, name):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())

def init_fc_weights(shape, name):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

def init_biases(shape):
    return tf.Variable(tf.constant(0.0,shape=shape))
    
def conv_layer(input_tensor,    # The input or previous layer
                filter_size,    # Width and height of each filter
                in_channels,    # Number of channels in previous layer
                num_filters,    # Number of filters
                layer_name,     # Layer name
                relu=True,           # Use Relu?
                pooling=False,
                pooling_size=4, 
                pooling_stride=4):       # Use 2x2 max-pooling?
    
    # Add layer name scopes for better graph visualization
    with tf.name_scope(layer_name):
    
        # Shape of the filter-weights for the convolution
        shape = [filter_size, filter_size, in_channels, num_filters]

        # Create weights and biases
        weights = init_conv_weights(shape, layer_name + '/weights')
        biases = init_biases([num_filters])
        
        # Add histogram summaries for weights
        tf.summary.histogram(layer_name + '/weights', weights)
        
        # Create the TensorFlow operation for convolution, with S=1 and zero padding
        activations = tf.nn.conv2d(input_tensor, weights, [1, 1, 1, 1], 'VALID') + biases

        # Rectified Linear Unit (ReLU)?
        if relu:
            activations = tf.nn.relu(activations)

        # Do we insert a pooling layer?
        if pooling:
            if pooling_size <= 0:
                pooling_size = 4
            if pooling_stride  <= 0:
                pooling_stride = 4
            # Create a pooling layer with F=2, S=1 and zero padding
            activations = tf.nn.max_pool(activations, [1, pooling_size, pooling_size, 1], [1, pooling_stride, pooling_stride, 1], 'VALID')

        # Return the resulting layer
        return activations

def flatten_tensor(input_tensor):
    """ Helper function for transforming a 4D tensor to 2D
    """
    # Get the shape of the input_tensor.
    input_tensor_shape = input_tensor.get_shape()

    # Calculate the volume of the input tensor
    num_activations = input_tensor_shape[1:4].num_elements()
    
    # Reshape the input_tensor to 2D: (?, num_activations)
    input_tensor_flat = tf.reshape(input_tensor, [-1, num_activations])

    # Return the flattened input_tensor and the number of activations
    return input_tensor_flat, num_activations

def fc_layer(input_tensor,  # The previous layer,         
             input_dim,     # Num. inputs from prev. layer
             output_dim,    # Num. outputs
             layer_name,    # The layer name
             relu=False):         # Use ReLU?

    # Add layer name scopes for better graph visualization
    with tf.name_scope(layer_name):
    
        # Create new weights and biases.
        weights = init_fc_weights([input_dim, output_dim], layer_name + '/weights')
        biases = init_biases([output_dim])
        
        # Add histogram summaries for weights
        tf.summary.histogram(layer_name + '/weights', weights)

        # Calculate the layer activation
        activations = tf.matmul(input_tensor, weights) + biases

        # Use ReLU?
        if relu:
            activations = tf.nn.relu(activations)

        return activations

with tf.name_scope("input"):
    
    # Placeholders for feeding input images
    x = tf.placeholder(tf.float32, shape=(None, img_height, img_width, num_channels), name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, num_labels], name='y_')
    y_cls = tf.argmax(y_, axis=1)
    
with tf.name_scope("augmentation"):
    is_augmentation = tf.placeholder(tf.bool, name='is_on')
    is_true = 0
    is_true = tf.cond(is_augmentation, lambda: 1, lambda: 0)
    if is_true == 1:
        x = augment(x, horizontal_flip=True, vertical_flip=True)

with tf.name_scope("dropout"):
    # Dropout rate applied to the input layer
    p_keep_1 = tf.placeholder(tf.float32)
    tf.summary.scalar('input_keep_probability', p_keep_1)





i = 0
idx_layer = 1
layer = "_layer"
#x = images
#y_ = labels
input_to_next_data = x
input_to_next_size = 1
last_layer = ''

for l_name, value in model_info.items():
    if i > 1:
        l_type = 'type' + layer + str(idx_layer)
        
        # if conv layer:
        if 'conv' in value[l_type]:
            # read whole this conv layer
            l_kernel_size = value['kernel_size' + layer + str(idx_layer)]
            l_num_output = value['num_output' + layer + str(idx_layer)]
            l_is_relu = value['is_relu' + layer + str(idx_layer)]
            l_is_pooling = value['is_pooling' + layer + str(idx_layer)]
            l_pooling_size = 0
            l_pooling_stride = 0
            if l_is_pooling:
                if ('pooling_stride' + layer + str(idx_layer)) in value:
                    l_pooling_stride = value['pooling_stride' + layer + str(idx_layer)]
                if ('pooling_size' + layer + str(idx_layer)) in value:
                    l_pooling_size = value['pooling_size' + layer + str(idx_layer)]
            # define model
            input_to_next_data = conv_layer(input_to_next_data, l_kernel_size, input_to_next_size, 
                                            l_num_output, layer_name = l_name, relu=l_is_relu, pooling=l_is_pooling,
                                            pooling_size=l_pooling_size, pooling_stride=l_pooling_stride)
            input_to_next_size = l_num_output
            last_layer = 'conv'
            
        # if full-connected layer:
        if 'fc' in value[l_type]:
            if 'conv' in last_layer:
                # flatten tensor for full connected layer
                input_to_next_data, input_to_next_size = flatten_tensor(input_to_next_data)
                
            l_num_output = value['num_output' + layer + str(idx_layer)]
            l_is_relu = value['is_relu' + layer + str(idx_layer)]
            input_to_next_data = fc_layer(input_to_next_data, input_to_next_size, l_num_output, 
                                          layer_name = l_name, relu=l_is_relu)
            input_to_next_size = l_num_output
            last_layer = 'fc'
            
        # if drop-out layer:
        if 'dropout' in value[l_type]:
            input_to_next_data = tf.nn.dropout(input_to_next_data, p_keep_1)
            # no need to set last layer for drop-out layer
            # last_layer = 'dropout'
            
        # if calculate loss layer:
        if 'loss' in value[l_type]:
            # Type of loss
            l_type_loss = value['type_loss' + layer + str(idx_layer)]
            
            # Softmax before calculate the loss
            y_pred = tf.nn.softmax(input_to_next_data, name = 'softmax')

            # The class-number is the index of the largest element.
            y_pred_cls = tf.argmax(y_pred, axis=1, name = 'output')
            
            print(y_pred)
            print(y_)
            if l_type_loss in 'SoftmaxCrossEntropyLogits':
                input_to_next_data = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_)
                
            if l_type_loss in 'SigmoidCrossEntropyLogits':
                input_to_next_data = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_)
            
            input_to_next_data = tf.reduce_mean(input_to_next_data, name="cost")
        
            last_layer = 'loss'
            
        # if optimizer layer:
        if 'optimizer' in value[l_type]:
            # Type of optimizer
            l_type_optimizer = value['type_optimizer' + layer + str(idx_layer)]
            l_type_lr = value['type_lr' + layer + str(idx_layer)]
            l_lr = value['lr' + layer + str(idx_layer)]
            global_step = tf.Variable(0)
            learning_rate = 0.0001
            if l_type_lr in 'fix':
                learning_rate = l_lr
            input_to_next_data = tf.train.AdamOptimizer(learning_rate).minimize(input_to_next_data, global_step=global_step, name="train")
            
            last_layer = 'optimizer'
            
        idx_layer = idx_layer + 1
    i = i + 1




correct_prediction = tf.equal(y_pred_cls, y_cls)

# Cast predictions to float and calculate the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")





batch_size=64
num_epochs=100
queue_loader = Queue_loader(batch_size=batch_size, num_epochs=num_epochs)
#print(queue_loader.images)
#print(queue_loader.labels)
labels = tf.one_hot(indices=queue_loader.labels, depth=15, name="arun")
labels = tf.cast(labels, tf.float32)
images = queue_loader.images
#dummy = tf.zeros(images.shape, dtype=images.dtype)
#images = tf.add(images, dummy, name="manu")
print(images)
print(labels)
num_batches = queue_loader.num_batches
saver = tf.train.Saver()
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, model_name)
init = tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')
#saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
sess = tf.Session(config=config)
sess.run(init)
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
print ('Start training')
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#tf.train.write_graph(sess.graph_def, "./checkpoints", model_name + ".pb", True)
#tf.train.write_graph(sess.graph_def, "./checkpoints", model_name + ".txt", True)
try:
    ep = 0
    step = 1
    while not coord.should_stop():
        
        images_, labels_ = sess.run([images, labels])
        #print(images_.shape)
        #print(labels_.shape)
        #images1 = images_
        #print(sum(images1 - images))
        
        #a, b, loss, acc = sess.run([input_to_next_data, opt, loss_, accuracy], feed_dict = {x:images_, y_:labels_, p_keep_1:0.4})
        a, b, loss, acc = sess.run(["softmax:0", "train:0", "cost:0", "accuracy:0"] , {"input/x:0": images_, "input/y_:0": labels_, 'dropout/Placeholder:0': 0.4})
        #print(a)
        #print(b)
        #print(acc)
        if step % 10 == 0:
            print ('epoch: %2d, step: %2d, loss: %.4f, accuracy: %.5f' % (ep+1, step, loss, acc))
        if step % num_batches == 0:
            print ('epoch: %2d, step: %2d, loss: %.4f, epoch %2d done.' % (ep+1, step, loss, ep+1))
            #checkpoint_path = os.path.join('data_log1', 'bv2_15Class')
            #saver.save(sess, checkpoint_path, global_step=ep+1)
            #tf.train.write_graph(sess.graph_def, "./data_log1", 'bv2_15Class-' + str(ep) + ".pb", False)
            minimal_graph = convert_variables_to_constants(sess, sess.graph_def, ["softmax"])
            saver.save(sess, './checkpoint/model' + str(ep) + '.ckpt')
            #saver.export_meta_graph('./checkpoint/my-model-10000.meta')
            tf.train.write_graph(minimal_graph, "./checkpoint", model_name + str(ep) + ".pb", True)
            tf.train.write_graph(sess.graph_def, "./checkpoint", model_name + str(ep) + ".txt", True)
            step = 1
            ep += 1
        else:
            step += 1
except tf.errors.OutOfRangeError:
    print ('\nDone training, epoch limit: %d reached.' % (ep))
finally:
    coord.request_stop()

coord.join(threads)
sess.close()
print ('Done')




