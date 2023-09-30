import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock,self).__init__()

        self.conv1 = layers.Conv2D(filter_num, [3,3], strides=stride, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, [3,3], strides=1, padding="same")
        self.bn2 = layers.BatchNormalization()
        if stride!=1:
          self.downsample = Sequential()
          self.downsample.add(layers.Conv2D(filter_num,[1,1],strides=stride))
        else:
          self.downsample = lambda x:x
    
    def call(self, inputs, training=None):
        
        out = self.conv1(inputs)
        out = self.bn1(out,training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out,training=training)
        
        identity = self.downsample(inputs)

        output = layers.add([out,identity])
        output = self.relu(output)

        return output

class ResNet(keras.Model):
  def __init__(self, layer_dims, num_classes):
    super(ResNet, self).__init__()

    self.stem = Sequential([
      layers.Conv2D(64, [3,3],strides=(1,1)),
      layers.BatchNormalization(),
      layers.Activation('relu'),
      layers.MaxPool2D(pool_size=[2,2],strides=(1,1),padding="same")
    ])
    
    self.layer1 = self.bulid_ResBlock(64,layer_dims[0])
    self.layer2 = self.bulid_ResBlock(128,layer_dims[1],stride=2)
    self.layer3 = self.bulid_ResBlock(256,layer_dims[2],stride=2)
    self.layer4 = self.bulid_ResBlock(512,layer_dims[3],stride=2)
    
    self.avgpool = layers.GlobalAveragePooling2D()
    self.fc = layers.Dense(num_classes)

  def call(self, inputs, training=None):
    x = self.stem(inputs)
    
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    
    x = self.avgpool(x)

    x = self.fc(x)
    
    return x
  
  def bulid_ResBlock(self, filter_num, blocks, stride=1):
    resblock = Sequential()
    
    resblock.add(BasicBlock(filter_num, stride))
    
    for _ in range(1,blocks):
      resblock.add(BasicBlock(filter_num, stride=1))
    return resblock
      
def resent18(num_classes):
  return ResNet([2,2,2,2],num_classes)