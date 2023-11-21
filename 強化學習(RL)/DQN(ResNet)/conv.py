import tensorflow as tf, datetime
from tensorflow import keras
from keras import layers, datasets, optimizers, Sequential
from keras.callbacks import Callback, LearningRateScheduler

def bulid_net(input_shape,nactions):

    conv_layer = [
        #input shape [b,32,32,3]
        
        layers.Conv2D(64,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
        layers.Conv2D(64,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2,2],strides=2,padding="same"),
        
        #shape [b,16,16,64]
        
        layers.Conv2D(128,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
        layers.Conv2D(128,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2,2],strides=2,padding="same"),
        
        #shape [b,8,8,128]
        
        layers.Conv2D(256,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
        layers.Conv2D(256,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2,2],strides=2,padding="same"),
        
        #shape [b,4,4,256]
        
        layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
        layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2,2],strides=2,padding="same"),
        
        #shape [b,2,2,512]
        
        layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
        layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2,2],strides=2,padding="same"),
        
        #shape [b,1,1,512]
        
        layers.Flatten(),
        
        layers.Dense(256,activation=tf.nn.relu),
        layers.Dense(128,activation=tf.nn.relu),
        layers.Dense(nactions)
    ]

    net = Sequential(conv_layer)
    net.build(input_shape=input_shape)
    return net