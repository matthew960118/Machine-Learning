import tensorflow as tf, datetime
from tensorflow import keras
from keras import layers, datasets, optimizers, Sequential
from keras.callbacks import Callback, LearningRateScheduler
from ResNet import resent18

classes = 100

class AccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # 在每個 epoch 結束時計算並顯示準確率
        accuracy = logs.get('val_accuracy')
        print(f'Epoch {epoch + 1} - Validation Accuracy: {accuracy:.4f}')
        

def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32)/255
    y = tf.cast(y, dtype=tf.int32)
    # y = tf.squeeze(y,axis=1)
    
    return x, y

(x, y), (x_test, y_test) = datasets.cifar100.load_data()
# (x, y), (x_test, y_test) = datasets.mnist.load_data()

y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test,axis=1)

x = 2*tf.cast(x,dtype=tf.float32)/255 - 1
x_test = 2*tf.cast(x_test,dtype=tf.float32)/255 - 1
y = tf.cast(y, dtype=tf.int32)
y_test = tf.cast(y_test, dtype=tf.int32)

y = tf.one_hot(y,depth=classes)
y_test = tf.one_hot(y_test,depth=classes)

accuracy_callback = AccuracyCallback()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model = resent18(classes)
model.build(input_shape=[None,28,28,3])
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),loss=tf.losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

#print(model.summary())

model.fit(x,y,validation_data=(x_test,y_test),batch_size=32,epochs=10,callbacks=[tensorboard_callback])
