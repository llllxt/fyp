import sys
import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sklearn.decomposition
from data.datasets import get_dataset
from sklearn.metrics import classification_report
from utils.utils import display_progression_epoch
from utils.evaluations import plot_confusion_matrix

def tf_extract_features(images, network_name, trainable=False, layer=None, cropping = False):
    print('entering tf_extract_features')
    url = "https://tfhub.dev/google/imagenet/{}/feature_vector/1".format(network_name)
    print("Using", url, layer)
    with tf.variable_scope('image_features'):
        try:
            module = hub.Module(url, trainable=trainable)
        except:
            raise(Exception("Please choose an available model"))
        height, width = hub.get_expected_image_size(module)
        print('expected height, width', height, width)
        ##Preprocessing
        if images.shape[1]!=height:
            images = tf.image.resize_images(images, (height, width))
        if images.shape[3]==1:
            images = tf.image.grayscale_to_rgb(images)
        img = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 3])
        module_features = module(dict(images=images), as_dict=True, signature='image_feature_vector')
        list_layers = []
        for l in module_features.keys():
            list_layers.append(l)
        print(list_layers)

        ## Features extraction
        if not cropping:

             if images.shape[1]!=height:
                 images = tf.image.resize_images(images, (height, width))

             feature_maps = module(dict(images=images), signature="image_feature_vector",
                                  as_dict=True)  # Features with shape [batch_size, num_features].
            
             print('feature_maps', feature_maps.items())
        if cropping:
            print('CROPPING')
            images = tf.image.resize_images(images, (320, 320))
            dict_feature_maps = get_feature_maps_cropped_images(images, module)       
        
        if   layer == 'concat':
            print('entering the concat condition')
            for i,layer in enumerate(list_layers):
                if not i :
                    if cropping:
                        interm = cropped_images_features(layer, dict_feature_maps)
                    else :
                        interm = feature_maps[layer]
                    if len(interm.get_shape())==4:
                        interm_mean = tf.reduce_mean(interm, axis=[1, 2])
                    else:
                        interm_mean = interm
                    concat_tensor_mean = interm_mean
                if i:
                    if cropping:
                        interm = cropped_images_features(layer, dict_feature_maps)
                    else:
                        interm = feature_maps[layer] 
                    if len(interm.get_shape())==4:
                        interm_mean = tf.reduce_mean(interm, axis=[1, 2])
                    else:
                        interm_mean = interm
                    concat_tensor_mean = tf.concat([concat_tensor_mean, interm_mean], axis = 1)

            return 'buffer', concat_tensor_mean

        else:
            print('entering the else condition')
            if layer is not None:
                if cropping:
                    v = cropped_images_features(layer, dict_feature_maps)
                else:
                    v = feature_maps[layer]
            else:
                if cropping:
                    v = cropped_images_features('default', dict_feature_maps)
                else:
                    v = feature_maps["default"]
            if len(v.get_shape())==4:
                features_mean = tf.reduce_mean(v, axis=[1, 2])
            else:
                features_mean = v

            if cropping:
                features = cropped_images_features('default', dict_feature_maps)
            else:
                features = feature_maps["default"]

            return features, features_mean


def extract_features(listx, input_shape, network_name, retrain_with=None, rd=42, layer=None, cropping=False):
    batch_size = 50
    list_feat = []

    if retrain_with is None:
        logdir = os.path.join("train_logs", network_name, str(rd))
    else:
        print("Retraining {} with {}".format(network_name, retrain_with))
        return retrain_features(listx, retrain_with, network_name, rd, layer)
    print('calling tf_extract_features with image size {}'.format(input_shape))
    print('Tensorflow preprocessing...')
    x_pl = tf.placeholder(tf.float32, shape=input_shape)
    features, features_int = tf_extract_features(x_pl, network_name, layer=layer, cropping=cropping)
    config = tf.ConfigProto(device_count = {'GPU': 2})
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for x in listx:
            n_batches = int(np.ceil(x.shape[0]/batch_size))
            print('n_batches', n_batches)
            for i in range(n_batches):
                display_progression_epoch(i, n_batches)
                #每次取一个batch的feature
                x_features_batches = sess.run(features_int, feed_dict={x_pl:x[i*batch_size:min((i+1)*batch_size, x.shape[0])]})
                if i:
                    #在i不是0的时候，把一个batch的feature concate到x_features
                    x_features = np.concatenate([x_features, x_features_batches], axis=0)
                else:
                    #在i是0时，初始化x_features
                    x_features = x_features_batches
            list_feat.append(x_features)
    return list_feat
