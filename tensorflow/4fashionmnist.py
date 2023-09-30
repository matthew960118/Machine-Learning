import tensorflow as tf , datetime
from tensorflow import keras
from keras import datasets, layers, optimizers, Sequential, metrics, Input
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# log_dir = '4logs/'+ current_time
# summary_writer = tf.summary.create_file_writer(log_dir)



(x_train,y_train), (x_test,y_test) = datasets.fashion_mnist.load_data()

def preprossing(x, y):
    x = tf.reshape(tf.cast(x,dtype=tf.float32)/255,[-1,28*28])
    y = tf.cast(y,dtype=tf.int32)
    return x, y

db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
db_train = db_train.map(preprossing).batch(128)
db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.map(preprossing).batch(128)

model = Sequential([
    Input(shape=(784,)),
    layers.Dense(256,activation=tf.nn.relu),
    layers.Dense(128,activation=tf.nn.relu),
    layers.Dense(64,activation=tf.nn.relu),
    layers.Dense(32,activation=tf.nn.relu),
    layers.Dense(10,activation=tf.nn.relu),
])
model.build([None,28*28])
optimizer = optimizers.Adam(0.001)
model.compile(optimizer=optimizer,loss=tf.losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(x=tf.reshape(x_train,[-1,28*28]), 
          y=tf.one_hot(y_train,depth=10), 
          epochs=5, 
          validation_data=(tf.reshape(x_test,[-1,28*28]), tf.one_hot(y_test,depth=10)), 
          callbacks=[tensorboard_callback])

def main():
    for epoch in range(20):
        # for step, (x, y) in enumerate(db_train):
        #     x = tf.reshape(x, [-1,28*28])
            
        #     y_onehot = tf.one_hot(y,depth=10)
        #     with tf.GradientTape() as tape:
        #       logits = model(x)
        #       loss = tf.reduce_mean(tf.losses.MSE(y_onehot,logits))
        #     grads = tape.gradient(loss,model.trainable_variables)
        #     optimizer.apply_gradients(zip(grads,model.trainable_variables))

        #     if step % 100 == 0:
        #         print(epoch, step,"loss:", loss)
        
        correct_sum = 0
        data_count = 0
        for step, (x, y) in enumerate(db_test):
            x = tf.reshape(x, [-1,28*28])
            
            logits = model(x)

            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1,output_type=tf.int32)

            correct = tf.reduce_sum(tf.cast(tf.equal(y,pred),dtype=tf.int32))
            correct_sum +=int(correct)
            data_count += x.shape[0]
        acc = correct_sum/data_count
        print(epoch, 'acc: ',acc)

main()
