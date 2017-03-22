# -*- encoding: utf-8 -*-

"""
Convolutional autoencoder (CAE) for kaomoji.
"""

__author__ = 'Edward Wong <lernanto.wong@gmail.com>'


import sys
import numpy
import numpy.random
import keras.models
import keras.layers
import keras.preprocessing.sequence
import keras.backend

from . import util


def load_data(fname, dic):
    corpus = util.Corpus(fname)
    data = [[dic.word_to_id(c) for c in line] for line in corpus]
    return keras.preprocessing.sequence.pad_sequences(data, maxlen=16)


def create_encoder(dict_size):
    """
    Create encoder part of CAE.
    """

    encoder = keras.models.Sequential()
    encoder.add(keras.layers.Embedding(dict_size, 64, input_length=16))
    encoder.add(keras.layers.Conv1D(64, 3, padding='same', activation='relu'))
    encoder.add(keras.layers.MaxPooling1D(2))
    encoder.add(keras.layers.Conv1D(128, 3, padding='same', activation='relu'))
    encoder.add(keras.layers.MaxPooling1D(2))
    encoder.add(keras.layers.Flatten())
    encoder.add(keras.layers.Dense(256))

    return encoder


def create_decoder(dict_size):
    """
    Create decoder part of CAE.
    """

    decoder = keras.models.Sequential()
    decoder.add(keras.layers.Dense(4 * 128, activation='relu', input_shape=(256,)))
    decoder.add(keras.layers.Reshape((4, 128)))
    decoder.add(keras.layers.UpSampling1D(2))
    decoder.add(keras.layers.Conv1D(128, 3, padding='same', activation='relu'))
    decoder.add(keras.layers.UpSampling1D(2))
    decoder.add(keras.layers.Conv1D(64, 3, padding='same', activation='relu'))
    decoder.add(keras.layers.Dense(dict_size, activation='softmax'))

    return decoder


def train(x_train, x_test, dict_size, batch_size=128, epochs=100):
    """
    Train convolutional autoencoder.
    """

    encoder = create_encoder(dict_size)
    decoder = create_decoder(dict_size)

    cae = keras.models.Sequential()
    cae.add(encoder)
    cae.add(decoder)

    cae.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta')
    cae.fit(
        x_train,
        numpy.expand_dims(x_train, -1),
        validation_data=(x_test, numpy.expand_dims(x_test, -1)),
        batch_size=batch_size,
        epochs=epochs
    )

    return cae, encoder, decoder


def test(cae, dic, texts, batch_size=128):
    """
    Show some reconstructions.
    """

    x = [[dic.word_to_id(c) for c in line] for line in texts]
    x = keras.preprocessing.sequence.pad_sequences(x, maxlen=16)
    recon = cae.predict_classes(x)

    for i, t in enumerate(texts):
        print '\t'.join((t, ''.join([dic.id_to_word(id) for id in recon[i]])))


if __name__ == '__main__':
    data_file = sys.argv[1]
    dict_output = sys.argv[2]
    encoder_output = sys.argv[3]
    decoder_output = sys.argv[4]

    # prepare data and dictionary
    corpus = util.Corpus(data_file)
    dic = util.Dictionary()
    dic.build(corpus, min_freq=3, max_size=1000)
    data = load_data(data_file, dic)
    numpy.random.shuffle(data)

    cut = int(0.9 * len(data))
    x_train = data[:cut]
    x_test = data[cut:]

    cae, encoder, decoder = train(x_train, x_test, len(dic))

    dic.save(dict_output)
    encoder.save(encoder_output)
    decoder.save(decoder_output)

    texts = []
    for i, t in enumerate(corpus):
        if i == 20:
            break
        texts.append(t)

    test(cae, dic, texts)
