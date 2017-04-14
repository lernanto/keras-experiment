# -*- encoding: utf-8 -*-

"""
Utils dealing with MNIST dataset.
"""

__author__ = 'Edward Wong <lernanto.wong@gmail.com>'


import gzip
import struct
import numpy
import keras.utils


def load_images(fname):
    """
    Read MNIST images from idx3-ubypte.gz file.
    """

    with gzip.open(fname) as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = numpy.fromstring(f.read(), dtype=numpy.uint8)
        images = images.astype(numpy.float32).reshape(len(images) // rows // cols, rows, cols, 1) / 255
        return images


def load_labels(fname):
    """
    Read MNIST labels from idx1-ubyte.gz file.
    """

    with gzip.open(fname) as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = numpy.fromstring(f.read(), dtype=numpy.uint8)
        return numpy.expand_dims(labels, axis=-1)
