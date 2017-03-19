# -*- encoding: utf-8 -*-

"""
Keras LeNet implementation.
"""

__author__ = 'Edward Wong <lernanto.wong@gmail.com>'


import sys
import logging
import keras.models
import keras.layers

from . import util

logger = logging.getLogger('LeNet')


def train(dir, output, batch_size=128, epochs=20):
    # load MNIST dataset from files
    x_train = util.load_images(dir + '/train-images-idx3-ubyte.gz')
    y_train = util.load_labels(dir + '/train-labels-idx1-ubyte.gz')
    x_test = util.load_images(dir + '/t10k-images-idx3-ubyte.gz')
    y_test = util.load_labels(dir + '/t10k-labels-idx1-ubyte.gz')

    # build up a 2D convolution network
    lenet = keras.models.Sequential()
    lenet.add(keras.layers.Conv2D(20, (5, 5), padding='same', activation='relu', input_shape=(1, 28, 28)))
    lenet.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    lenet.add(keras.layers.Conv2D(50, (5, 5), padding='same', activation='relu'))
    lenet.add(keras.layers.MaxPooling2D((2, 2)))
    lenet.add(keras.layers.Flatten())
    lenet.add(keras.layers.Dense(500, activation='relu'))
    lenet.add(keras.layers.Dense(10, activation='softmax'))

    lenet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # do the train
    lenet.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    score = lenet.evaluate(x_test, y_test, verbose=0)
    logger.info('evaluation score = %r' % score)

    return lenet


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)6s] %(asctime)s %(message)s'
    )

    input_dir = sys.argv[1]
    output = sys.argv[2]

    lenet = train(input_dir, output)

    # save model as HDF5 format
    lenet.save(output)