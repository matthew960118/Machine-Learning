import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, optimizers
import numpy as np

(xs_train, ys_train), (xs_test, ys_test) = datasets.mnist.load_data()

xs_train = tf.convert_to_tensor(xs_train,dtype=tf.float32)/255
ys_train = tf.convert_to_tensor(ys_train,dtype=tf.int32)
ys_train = tf.one_hot(ys_train,depth=10)

db = tf.data.Dataset.from_tensor_slices((xs_train,ys_train)).batch(255)

model = keras.Sequential([
    layers.Dense(512,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(10)
])

optimizer = optimizers.SGD(learning_rate=0.001)

def train_epoch(epoch):
  for step, (x,y) in enumerate(db):
    with tf.GradientTape() as tape:
        x = tf.reshape(x,(-1,28*28))

        out = model(x)

        loss = tf.reduce_sum(tf.square(out-y))/x.shape[0]

    grads = tape.gradient(loss,model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if step %100 == 0:
       print(epoch,step,"loss:",loss.numpy())

def train():
    for epoch in range(30):
      train_epoch(epoch)
    print("save weights...")

    model.save_weights('./checkpoints/weights.ckpt')

train()

