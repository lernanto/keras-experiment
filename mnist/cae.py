#!env python3 -O
# -*- encoding: utf-8 -*-

"""
Keras convolutional autoencoder (CAE) for MNIST images.
"""

__author__ = 'Edward Wong <lernanto.wong@gmail.com>'


import sys
import keras.models
import keras.layers
import matplotlib.pyplot

from . import util


def create_encoder():
    """
    Create encoder model.
    """

    encoder = keras.models.Sequential()
    encoder.add(keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
    encoder.add(keras.layers.MaxPooling2D((2, 2), padding='same'))
    encoder.add(keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu'))
    encoder.add(keras.layers.MaxPooling2D((2, 2), padding='same'))
    encoder.add(keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu'))
    encoder.add(keras.layers.MaxPooling2D((2, 2), padding='same'))
    encoder.add(keras.layers.Flatten())

    return encoder


def create_decoder():
    """
    Create decoder model.
    """

    decoder =keras.models.Sequential()
    decoder.add(keras.layers.Reshape((4, 4, 8), input_shape=(4 * 4 * 8,)))
    decoder.add(keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu'))
    decoder.add(keras.layers.UpSampling2D((2, 2)))
    decoder.add(keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu'))
    decoder.add(keras.layers.UpSampling2D((2, 2)))
    decoder.add(keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
    decoder.add(keras.layers.UpSampling2D((2, 2)))
    decoder.add(keras.layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid'))
    # output shape here will be (32, 32, 1), slice it to fit original image
    decoder.add(keras.layers.Lambda(lambda x: x[:, 2:-2, 2:-2, :], output_shape=(28, 28, 1)))

    return decoder


def create_cae(encoder, decoder):
    """
    Create convolutional autoencoder model.
    """

    cae = keras.models.Sequential()
    cae.add(encoder)
    cae.add(decoder)
    cae.compile(optimizer='adadelta', loss='binary_crossentropy')

    return cae


def train(x_train, x_test, batch_size=128, epochs=50):
    """
    Train autoencoder from input images.
    """

    encoder = create_encoder()
    decoder = create_decoder()
    cae = create_cae(encoder, decoder)

    cae.fit(
        x_train,
        x_train,
        validation_data=(x_test, x_test),
        batch_size=batch_size,
        epochs=epochs
    )

    return cae, encoder, decoder


def show(cae, images):
    """
    Show some test images and their reconstructions.
    """

    recon = cae.predict(images)

    matplotlib.pyplot.figure(figsize=(20, 4))
    for i, img in enumerate(images):
        # display original images
        ax = matplotlib.pyplot.subplot(2, len(images), i + 1)
        matplotlib.pyplot.imshow(img.reshape(28, 28))
        matplotlib.pyplot.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstructions
        ax = matplotlib.pyplot.subplot(2, len(images), i + 1 + len(images))
        matplotlib.pyplot.imshow(recon[i].reshape(28, 28))
        matplotlib.pyplot.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    matplotlib.pyplot.show()


if __name__ == '__main__':
    input_dir = sys.argv[1]
    encoder_output = sys.argv[2]
    decoder_output = sys.argv[3]

    # load MNIST dataset from files
    x_train = util.load_images(input_dir + '/train-images-idx3-ubyte.gz')
    x_test = util.load_images(input_dir + '/t10k-images-idx3-ubyte.gz')

    cae, encoder, decoder = train(x_train, x_test)
    encoder.save(encoder_output)
    decoder.save(decoder_output)

    show(cae, x_test[:10])
