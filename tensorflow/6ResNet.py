import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock,self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3,3), strides=stride, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3,3), strides=1, padding="same")
        self.bn2 = layers.BatchNormalization()
        if stride!=1:
          self.downsample = Sequential()
          self.downsample.add(layers.Conv2D(filter_num,(1,1),strides=stride))
        else:
          self.downsample = lambda x:x
    
    def call(self, inputs, training=None):
        
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        identity = self.downsample(inputs)

        output = layers.add([out,identity])
        output = self.relu(output)

        return output
