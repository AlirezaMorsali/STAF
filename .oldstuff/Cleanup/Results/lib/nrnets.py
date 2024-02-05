#!/usr/bin/env jupyter
import tensorflow as tf
import numpy as np
import math
from math import log10, sqrt


def make_coordiante(shape):
    x_coordinates = np.linspace(-1, +1, shape[0])
    y_coordinates = np.linspace(-1, +1, shape[1])
    x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
    x_coordinates = x_coordinates.flatten()
    y_coordinates = y_coordinates.flatten()
    Coordinates = np.stack([x_coordinates, y_coordinates]).T
    return Coordinates


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


class SineLayer(tf.keras.layers.Layer):
    def __init__(self, in_features, units, bias=True, is_first=False, omega_0=30.):
        super(SineLayer, self).__init__()
        self.in_features = in_features
        self.units = units
        self.is_first = is_first
        self.omega_0 = omega_0

        self.dense = tf.keras.layers.Dense(self.units,
                                           use_bias=bias,
                                           kernel_initializer=self.init_weights(),
                                           input_shape=(self.in_features,))


    def init_weights(self):
        if self.is_first:
            return tf.keras.initializers.RandomUniform(minval=-1 / self.in_features,
                                                       maxval= 1 / self.in_features)
        else:
            return tf.keras.initializers.RandomUniform(minval=-np.sqrt(6. / self.in_features) / self.omega_0,
                                                       maxval= np.sqrt(6. / self.in_features) / self.omega_0)


    def call(self, input_tensor):
        befor_activation = self.dense(input_tensor)
        after_activation = tf.sin(self.omega_0 * befor_activation)
        #after_activation = befor_activation
        return after_activation


class Siren(tf.keras.Model):
    def __init__(self, in_features,
                    units,
                    out_features,
                    outermost_linear=True,
                    first_omega_0=30.,
                    hidden_omega_0=30.):
        super(Siren, self).__init__()
        self.in_features = in_features
        self.units = units
        self.out_features = out_features
        self.outermost_linear = outermost_linear
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0

        self.net = []

        self.net.append(SineLayer(self.in_features,
                                  self.units[0],
                                  is_first=True,
                                  omega_0=self.first_omega_0))

        for counter in range(1, len(units)):
            self.net.append(SineLayer(self.units[counter-1],
                                      self.units[counter],
                                      is_first=False,
                                      omega_0=self.hidden_omega_0))

        if self.outermost_linear:
            self.net.append(tf.keras.layers.Dense(self.out_features,
                                                  kernel_initializer=self.init_weights()))
        else:
            self.net.append(SineLayer(self.units[counter],
                                      self.out_features,
                                      is_first=False,
                                      omega_0=self.hidden_omega_0))

    def init_weights(self):
        return tf.keras.initializers.RandomUniform(minval=-np.sqrt(6. / self.units[-1]) / self.hidden_omega_0,
                                                   maxval= np.sqrt(6. / self.units[-1]) / self.hidden_omega_0)

    def call(self, input_tensor):
        x = input_tensor
        for layer in self.net:
            x = layer(x)


class ParaSineLayer(tf.keras.layers.Layer):
    def __init__(self, in_features, units, bias=True, is_first=False, omega_0=30.):
        super(ParaSineLayer, self).__init__()
        self.in_features = in_features
        self.units = units
        self.is_first = is_first
        self.omega_0 = omega_0

        self.dense = tf.keras.layers.Dense(self.units,
                                           use_bias=bias,
                                           kernel_initializer=self.init_weights(),
                                           input_shape=(self.in_features,))

    def cal_limit(self):
        X = 2 * np.random.rand(1000, self.in_features) - 1
        W = 2 * np.random.rand(self.in_features, self.units) - 1
        A = np.matmul(X, W)
        return np.std(A)

    def init_weights(self):
        if self.is_first:
            return tf.keras.initializers.RandomUniform(minval=-1 / (self.cal_limit()),
                                                       maxval= 1 / (self.cal_limit()))

        else:
            return tf.keras.initializers.RandomUniform(minval=-np.sqrt(6. / self.in_features) / self.omega_0,
                                                       maxval= np.sqrt(6. / self.in_features) / self.omega_0)

    def build(self, input_shape):

        self.a_1 = self.add_weight(
            name='a_1',
            shape=(1,),
            initializer='zeros',
            trainable=True)

        self.a0 = self.add_weight(
            name='a0',
            shape=(1,),
            initializer='ones',
            trainable=True)
        self.w0 = self.add_weight(
            name='w0',
            shape=(1,),
            initializer='ones',
            trainable=True)
        self.shift0 = self.add_weight(
            name='shift0',
            shape=(1,),
            initializer='zeros',
            trainable=True)

        self.a1 = self.add_weight(
            name='a1',
            shape=(1,),
            initializer='ones',
            trainable=True)
        self.w1 = self.add_weight(
            name='w1',
            shape=(1,),
            initializer='ones',
            trainable=True)
        self.shift1 = self.add_weight(
            name='shift1',
            shape=(1,),
            initializer='zeros',
            trainable=True)

        self.a2 = self.add_weight(
            name='a2',
            shape=(1,),
            initializer=tf.keras.initializers.constant(1/20.),
            trainable=True)
        self.w2 = self.add_weight(
            name='w2',
            shape=(1,),
            initializer=tf.keras.initializers.constant(2.),
            trainable=True)
        self.shift2 = self.add_weight(
            name='shift2',
            shape=(1,),
            initializer='zeros',
            trainable=True)

        self.a3 = self.add_weight(
            name='a3',
            shape=(1,),
            initializer=tf.keras.initializers.constant(0.),
            trainable=True)
        self.w3 = self.add_weight(
            name='w3',
            shape=(1,),
            initializer=tf.keras.initializers.constant(2.),
            trainable=True)
        self.shift3 = self.add_weight(
            name='shift3',
            shape=(1,),
            initializer='zeros',
            trainable=True)

        self.a4 = self.add_weight(
            name='a4',
            shape=(1,),
            initializer=tf.keras.initializers.constant(0.),
            trainable=True)
        self.w4 = self.add_weight(
            name='w4',
            shape=(1,),
            initializer=tf.keras.initializers.constant(3.),
            trainable=True)
        self.shift4 = self.add_weight(
            name='shift4',
            shape=(1,),
            initializer='zeros',
            trainable=True)

        self.a5 = self.add_weight(
            name='a5',
            shape=(1,),
            initializer=tf.keras.initializers.constant(1/30.),
            trainable=True)
        self.w5 = self.add_weight(
            name='w5',
            shape=(1,),
            initializer=tf.keras.initializers.constant(3.),
            trainable=True)
        self.shift5 = self.add_weight(
            name='shift5',
            shape=(1,),
            initializer='zeros',
            trainable=True)

        super(ParaSineLayer, self).build(input_shape)

    def call(self, input_tensor):
        befor_activation = self.dense(input_tensor)
        after_activation = self.a_1 * self.omega_0 * befor_activation + \
                           self.a0 * tf.sin(self.w0 * self.omega_0 * befor_activation + self.shift0) + \
                           self.a1 * tf.cos(self.w1 * self.omega_0 * befor_activation + self.shift1) + \
                           self.a2 * tf.sin(self.w2 * self.omega_0 * befor_activation + self.shift2) + \
                           self.a3 * tf.cos(self.w3 * self.omega_0 * befor_activation + self.shift3) + \
                           self.a4 * tf.sin(self.w4 * self.omega_0 * befor_activation + self.shift4) + \
                           self.a5 * tf.cos(self.w5 * self.omega_0 * befor_activation + self.shift5)
        return after_activation

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self, image_shape, step_show, batch_size, is_fourier=False, B=None):
        super(CustomSaver, self).__init__()
        self.image_shape = image_shape
        self.step = step_show
        self.data = make_coordiante(self.image_shape)
        if is_fourier:
            self.data = self.data @ B
            self.data = np.concatenate([np.sin(2 * math.pi * self.data), np.cos(2 * math.pi * self.data)], axis=-1)
            self.data = tf.data.Dataset.from_tensor_slices(self.data).batch(batch_size)
        else:
            self.data = tf.data.Dataset.from_tensor_slices(self.data).batch(batch_size)
        self.images = []

        def on_epoch_end(self, epoch, logs={}):
            if epoch % self.step == 0:
                self.images.append(self.model.predict(self.data).reshape(self.image_shape))
