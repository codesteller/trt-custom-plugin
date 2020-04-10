import tensorflow as tf
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import InputLayer, Dense, Flatten, Activation, Conv2D, MaxPooling2D, LeakyReLU


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def test_tensorflow():
    weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    }
    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
    }


    in_shape = (150, 150, 3)
    a = np.random.rand(1, 150, 150, 3)
    x = tf.placeholder(tf.float32, [None, in_shape[0], in_shape[1], in_shape[2]])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        out = sess.run(conv1, feed_dict={x: a})

    print(out.shape)

def test_keras():
    in_shape = (150, 150, 3)
    a = np.random.rand(1, 150, 150, 3)

    model = Sequential()
    model.add(InputLayer(input_shape=in_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='linear',
                              input_shape=in_shape, padding='same'))

    out = model.predict(a)
    print(out.shape)
    


if __name__ == "__main__":
    # test_tensorflow()
    test_keras()
