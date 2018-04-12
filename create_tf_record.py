import tensorflow as tf
from PIL import Image
import numpy as np



with open('bmppath.txt') as f:
    array = []
    for line in f:
        if len(line) > 10:
            #print(len(line))
            array.append(line)
        else:
            print(line)
print(len(array))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


classes = ['GoodTrain', 'WaterEdgeTrain', 'WetEdgeTrain', 'DryEdgeTrain', 'PinchTrain', 'FoldLensTrain', 'NoLensTrain', 'NoLensScratchTrain', 'FovLensTrain', 'MultipleLensTrain', 'GapLensTrain', 'PunchOutTrain', 'UnderdoseTrain', 'InnerTearTrain', 'OuterTearTrain']
tfrecords_train_filename = "/home/ai/BVe-G/bv2_15_data_train.tfrecords"
writer = tf.python_io.TFRecordWriter(tfrecords_train_filename)
i =0
for imgname in array:
    #print(imgname)
    imgname = imgname.replace("\n", "")
    if ' ' in imgname:
        imgname = 'notconsider'
    if imgname.endswith('.bmp'):
        img = np.array(Image.open(imgname).resize((256,256), Image.ANTIALIAS))
        height = img.shape[0]
        width = img.shape[1]
        depth = 1
        #print(height,width,depth)    
        img_raw = img.tostring()
                    
        # extract label base on file name:
        #label = np.zeros((1, len(classes)))
        label = -1
        for l in range(len(classes)):
            if classes[l] in imgname:
                label = classes.index(classes[l]) 
                #print(label)
                print("%s %s" % (imgname, label))   

        if label != -1:
            example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'depth': _int64_feature(depth),
                    'image_raw': _bytes_feature(img_raw),
                    'label': _int64_feature(label)}))
                    
            writer.write(example.SerializeToString())
        else:
            print ("No expected label, error reading image: " + img_path)   