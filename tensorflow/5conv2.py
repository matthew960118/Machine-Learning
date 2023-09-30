import tensorflow as tf, datetime
from tensorflow import keras
from keras import layers, datasets, optimizers, Sequential
from keras.callbacks import Callback, LearningRateScheduler

class AccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # 在每個 epoch 結束時計算並顯示準確率
        accuracy = logs.get('val_accuracy')
        print(f'Epoch {epoch + 1} - Validation Accuracy: {accuracy:.4f}')
        

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
    layers.Dense(100)
]

# fc_layer = [
#     layers.Dense(256,activation=tf.nn.relu),
#     layers.Dense(128,activation=tf.nn.relu),
#     layers.Dense(100),
# ]


def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32)/255
    y = tf.cast(y, dtype=tf.int32)
    # y = tf.squeeze(y,axis=1)
    
    return x, y

def lr_schedule(epoch, lr):
    if epoch == 25:
        return lr * 0.1  # 在第 50 個 epoch 將學習率減少到原來的 0.1
    else:
        return lr

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test,axis=1)

x = tf.cast(x,dtype=tf.float32)/255
x_test = tf.cast(x_test,dtype=tf.float32)/255
y = tf.cast(y, dtype=tf.int32)
y_test = tf.cast(y_test, dtype=tf.int32)

y = tf.one_hot(y,depth=100)
y_test = tf.one_hot(y_test,depth=100)

accuracy_callback = AccuracyCallback()

lr_scheduler = LearningRateScheduler(lr_schedule)

# train_db = tf.data.Dataset.from_tensor_slices((x,y))
# train_db = train_db.shuffle(1000).map(preprocess).batch(128)

# test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
# test_db = test_db.map(preprocess).batch(128)

net = Sequential(conv_layer)
net.build(input_shape=[None,32,32,3])
net.compile(
    optimizer= optimizers.Adam(learning_rate=1e-4),
    loss= tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy',]
    )


net.fit(x,y,validation_data=(x_test, y_test),epochs=100,callbacks=[tensorboard_callback,lr_scheduler],batch_size=64)