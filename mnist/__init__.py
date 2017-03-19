# -*- encoding: utf-8 -*-

"""
keras experiments on MNIST dataset.
"""

__author__ = "Edward Wong <lernanto.wong@gmail.com>"


import keras.backend

# use theano image dimension ordering
keras.backend.set_image_dim_ordering('th')