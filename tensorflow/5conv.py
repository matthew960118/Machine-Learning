import tensorflow as tf
from tensorflow import keras
from keras import layers, datasets, optimizers, Sequential

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
]

fc_layer = [
    layers.Dense(256,activation=tf.nn.relu),
    layers.Dense(128,activation=tf.nn.relu),
    layers.Dense(100),
]


def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32)/255
    y = tf.cast(y, dtype=tf.int32)
    # y = tf.squeeze(y,axis=1)
    
    return x, y

(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test,axis=1)


train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(1000).map(preprocess).batch(128)

test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(128)


def main():
    lr = 1e-4
    
    conv_net = Sequential(conv_layer)
    conv_net.build(input_shape=[None,32,32,3])
    
    fc_net = Sequential(fc_layer)
    fc_net.build(input_shape=[None,512])
    
    variables = conv_net.trainable_variables + fc_net.trainable_variables
    optimizer = optimizers.Adam(lr)
    
    for epoch in range(50):
        if epoch%20 == 0:
            lr = lr/10
            optimizer.l
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = conv_net(x)
                out = tf.reshape(out, [-1,512])
                logits = fc_net(out)
                
                y = tf.one_hot(y,depth=100)
                
                loss = tf.losses.categorical_crossentropy(y,logits,from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads,variables))

            if step%100 == 0:
                print(epoch, step, "loss: ", float(loss))
                
        total_num = 0
        total_correct = 0
        for step, (x, y) in enumerate(test_db):
            out = conv_net(x)
            out = tf.reshape(out,[-1,512])
            logits = fc_net(out)
            
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1,output_type=tf.int32)
            
            correct = tf.cast(tf.equal(y,pred), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_correct+=int(correct)
            total_num += x.shape[0]
        acc = total_correct/total_num
        print(epoch, "acc: ", acc)
    
main()