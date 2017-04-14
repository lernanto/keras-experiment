#!env python3 -O
# -*- encoding: utf-8 -*-

"""
Keras Siamese network implementation.
"""

__author__ = 'Edward Wong <lernanto.wong@gmail.com>'


import sys
import logging
import numpy
import scipy.linalg
import keras.models
import keras.layers
import matplotlib.pyplot

from . import util

logger = logging.getLogger('Siamese')


def euclidean_distance(inputs):
    """
    Compute Euclidean distance of input tensors.
    """

    x, y = inputs
    return keras.backend.sqrt(keras.backend.sum(keras.backend.square(x - y), axis=-1, keepdims=True))


def contrastive_loss(label, distance):
    """
    Contrastive loss.
    """

    margin = 1.0
    return keras.backend.mean(label * keras.backend.square(distance) +
            (1 - label) * keras.backend.square(keras.backend.maximum(0.0, margin - distance)))


def create_encoder():
    """
    Create convolutional encoder for Siamese network.
    """

    encoder = keras.models.Sequential()
    encoder.add(keras.layers.Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(28, 28, 1)))
    encoder.add(keras.layers.MaxPooling2D((2, 2), padding='valid'))
    encoder.add(keras.layers.Dropout(0.1))
    encoder.add(keras.layers.Conv2D(16, (3, 3), padding='valid', activation='relu'))
    encoder.add(keras.layers.MaxPooling2D((2, 2), padding='valid'))
    encoder.add(keras.layers.Dropout(0.1))
    encoder.add(keras.layers.Flatten())
    encoder.add(keras.layers.Dense(8, activation='relu'))

    return encoder


def generate_pairs(images, labels):
    """
    Generate positive/negative images pairs from labeled data.
    """

    images1 = []
    images2 = []
    new_labels = []
    for i, img in enumerate(images):
        # find next same digit
        for j in range(i, len(images)):
            if numpy.array_equal(labels[i], labels[j]):
                images1.append(img)
                images2.append(images[j])
                new_labels.append([1])
                break

        for j in range(i, len(images)):
            if not numpy.array_equal(labels[i], labels[j]):
                images1.append(img)
                images2.append(images[j])
                new_labels.append([0])
                break

    return numpy.asarray(images1), numpy.asarray(images2), numpy.asarray(new_labels)


def train(image_train, label_train, image_test, label_test, batch_size=128, epochs=20):
    """
    Train Siamese network.
    """

    encoder = create_encoder()

    input1 = keras.layers.Input(shape=(28, 28, 1))
    encode1 = encoder(input1)

    input2 = keras.layers.Input(shape=(28, 28, 1))
    encode2 = encoder(input2)

    distance = keras.layers.Lambda(euclidean_distance, output_shape=(1,))([encode1, encode2])

    siamese = keras.models.Model([input1, input2], distance)
    siamese.compile(loss=contrastive_loss, optimizer='rmsprop')

    siamese.fit(
        image_train,
        label_train,
        validation_data=[image_test, label_test],
        batch_size=batch_size,
        epochs=epochs
    )

    return encoder


def accuracy(encoder, inputs, labels):
    """
    Compute classification accuracy.

    2 images with disatnce less than 0.5 are considered the same digit.
    """

    image1, image2 = inputs
    emb1 = encoder.predict(image1)
    emb2 = encoder.predict(image2)
    distances = numpy.sqrt(numpy.sum(numpy.square(emb1 - emb2), axis=-1))
    acc = numpy.mean(numpy.round(numpy.abs(labels - distances)))
    return acc


def visualize(encoder, images, labels):
    """
    Visualize.
    """

    points = encoder.predict(images)
    u, s, vt = scipy.linalg.svd(points)
    u = u[:, :2]
    s = s[:2]
    points = numpy.dot(u, numpy.diag(s))

    for i, color in enumerate(['red', 'green', 'blue', 'yellow', 'purple', 'olive', 'maroon', 'navy', 'lime', 'aqua']):
        x = points[numpy.where(labels[:, 0] == i)]
        matplotlib.pyplot.scatter(x[:, 0], x[:, 1], marker='.', color=color)

    matplotlib.pyplot.show()


if __name__ == '__main__':
    input_dir = sys.argv[1]
    output = sys.argv[2]

    # load MNIST dataset from files
    x_train = util.load_images(input_dir + '/train-images-idx3-ubyte.gz')
    y_train = util.load_labels(input_dir + '/train-labels-idx1-ubyte.gz')
    x_test = util.load_images(input_dir + '/t10k-images-idx3-ubyte.gz')
    y_test = util.load_labels(input_dir + '/t10k-labels-idx1-ubyte.gz')

    # prepair train and test data
    image_train1, image_train2, label_train = generate_pairs(x_train, y_train)
    image_test1, image_test2, label_test = generate_pairs(x_test, y_test)

    encoder = train([image_train1, image_train2], label_train, [image_test1, image_test2], label_test, epochs=1)
    encoder.save(output)

    # acc_train = accuracy(encoder, [image_train1, image_train2], label_train)
    acc_test = accuracy(encoder, [image_test1, image_test2], label_test)
    # print('train accuracy = %r, test accuracy = %r' % (acc_train, acc_test))
    print('test accuracy = %r' % acc_test)

    visualize(encoder, x_test, y_test)
