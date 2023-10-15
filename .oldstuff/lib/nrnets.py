#!/usr/bin/env jupyter
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import skimage
import math
import time
from math import log10, sqrt
from tensorflow import keras
from tensorflow.keras import layers


class SineLayer(tf.keras.layers.Layer):
    def __init__(self, in_features, units, bias=True, is_first=False, omega_0=30.0):
        super(SineLayer, self).__init__()
        self.in_features = in_features
        self.units = units
        self.is_first = is_first
        self.omega_0 = omega_0

        self.dense = tf.keras.layers.Dense(
            self.units,
            use_bias=bias,
            kernel_initializer=self.init_weights(),
            input_shape=(self.in_features,),
        )

    def init_weights(self):
        if self.is_first:
            return tf.keras.initializers.RandomUniform(
                minval=-1 / self.in_features, maxval=1 / self.in_features
            )
        else:
            return tf.keras.initializers.RandomUniform(
                minval=-np.sqrt(6.0 / self.in_features) / self.omega_0,
                maxval=np.sqrt(6.0 / self.in_features) / self.omega_0,
            )

    def call(self, input_tensor):
        befor_activation = self.dense(input_tensor)
        after_activation = tf.sin(self.omega_0 * befor_activation)
        # after_activation = befor_activation
        return after_activation
# The one that had a bug
# class Siren(tf.keras.Model):
#     def __init__(
#         self,
#         in_features,
#         units,
#         out_features,
#         outermost_linear=True,
#         first_omega_0=30.0,
#         hidden_omega_0=30.0,
#     ):
#         super(Siren, self).__init__()
#         self.in_features = in_features
#         self.units = units
#         self.out_features = out_features
#         self.outermost_linear = outermost_linear
#         self.first_omega_0 = first_omega_0
#         self.hidden_omega_0 = hidden_omega_0

#         self.net = []
#         self.net.append(
#             SineLayer(
#                 self.in_features,
#                 self.units[0],
#                 is_first=True,
#                 omega_0=self.first_omega_0,
#             )
#         )

#         for counter in range(1, len(units)):
#             self.net.append(
#                 SineLayer(
#                     self.units[counter - 1],
#                     self.units[counter],
#                     is_first=False,
#                     omega_0=self.hidden_omega_0,
#                 )
#             )

#         if self.outermost_linear:
#             self.net.append(
#                 tf.keras.layers.Dense(
#                     self.out_features, kernel_initializer=self.init_weights()
#                 )
#             )
#         else:
#             self.net.append(
#                 SineLayer(
#                     self.units[counter],
#                     self.out_features,
#                     is_first=False,
#                     omega_0=self.hidden_omega_0,
#                 )
#             )

#     def init_weights(self):
#         return tf.keras.initializers.RandomUniform(
#             minval=-np.sqrt(6.0 / self.units[-1]) / self.hidden_omega_0,
#             maxval=np.sqrt(6.0 / self.units[-1]) / self.hidden_omega_0,
#         )

#     def call(self, input_tensor):
#         x = input_tensor
#         for layer in self.net:
#             x = layer(x)


class Siren(tf.keras.Model):
    def __init__(
        self,
        in_features,
        units,
        out_features,
        outermost_linear=True,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
    ):
        super(Siren, self).__init__()
        self.in_features = in_features
        self.units = units
        self.out_features = out_features
        self.outermost_linear = outermost_linear
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0

        self.net = []
        self.net.append(
            SineLayer(
                self.in_features,
                self.units[0],
                is_first=True,
                omega_0=self.first_omega_0,
            )
        )

        for counter in range(1, len(units)):
            self.net.append(
                SineLayer(
                    self.units[counter - 1],
                    self.units[counter],
                    is_first=False,
                    omega_0=self.hidden_omega_0,
                )
            )

        if self.outermost_linear:
            self.net.append(
                tf.keras.layers.Dense(
                    self.out_features, kernel_initializer=self.init_weights()
                )
            )
        else:
            self.net.append(
                SineLayer(
                    self.units[counter],
                    self.out_features,
                    is_first=False,
                    omega_0=self.hidden_omega_0,
                )
            )

    def init_weights(self):
        return tf.keras.initializers.RandomUniform(
            minval=-np.sqrt(6.0 / self.units[-1]) / self.hidden_omega_0,
            maxval=np.sqrt(6.0 / self.units[-1]) / self.hidden_omega_0,
        )

    def call(self, input_tensor):
        x = input_tensor
        for layer in self.net:
            x = layer(x)
        return x


def get_siren(
    in_features,
    out_features,
    units,
    outermost_linear=True,
    first_omega_0=30.0,
    hidden_omega_0=30.0,
):

    inputs = keras.Input(shape=(in_features))
    x = SineLayer(in_features, units[0], is_first=True, omega_0=first_omega_0)(inputs)
    for la in range(1, len(units)):
        x = SineLayer(units[la - 1], units[la], omega_0=hidden_omega_0)(x)
    if outermost_linear:
        outputs = layers.Dense(out_features)(x)
    else:
        outputs = SineLayer(units[la], out_features, omega_0=hidden_omega_0)(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="Siren")


# class SineLayer(tf.keras.layers.Layer):
#     def __init__(self, in_features, units, bias=True, is_first=False, omega_0=30.):
#         super(SineLayer, self).__init__()
#         self.in_features = in_features
#         self.units = units
#         self.is_first = is_first
#         self.omega_0 = omega_0

#         self.dense = tf.keras.layers.Dense(self.units,
#                                            use_bias=bias,
#                                            kernel_initializer=self.init_weights(),
#                                            input_shape=(self.in_features,))


#     def init_weights(self):
#         if self.is_first:
#             return tf.keras.initializers.RandomUniform(minval=-1 / self.in_features,
#                                                        maxval= 1 / self.in_features)
#         else:
#             return tf.keras.initializers.RandomUniform(minval=-np.sqrt(6. / self.in_features) / self.omega_0,
#                                                        maxval= np.sqrt(6. / self.in_features) / self.omega_0)


#     def call(self, input_tensor):
#         befor_activation = self.dense(input_tensor)
#         after_activation = tf.sin(self.omega_0 * befor_activation)
#         #after_activation = befor_activation
#         return after_activation


# class Siren(tf.keras.Model):
#     def __init__(self, in_features,
#                     units,
#                     out_features,
#                     outermost_linear=False,
#                     first_omega_0=30.,
#                     hidden_omega_0=30.):
#         super(Siren, self).__init__()
#         self.in_features = in_features
#         self.units = units
#         self.out_features = out_features
#         self.outermost_linear = outermost_linear
#         self.first_omega_0 = first_omega_0
#         self.hidden_omega_0 = hidden_omega_0


#         self.net = []

#         self.net.append(SineLayer(self.in_features,
#                                   self.units[0],
#                                   is_first=True,
#                                   omega_0=self.first_omega_0))

#         for counter in range(1, len(units)):
#             self.net.append(SineLayer(self.units[counter-1],
#                                       self.units[counter],
#                                       is_first=False,
#                                       omega_0=self.hidden_omega_0))

#         if self.outermost_linear:
#             self.net.append(tf.keras.layers.Dense(self.out_features,
#                                                   kernel_initializer=self.init_weights()))
#         else:
#             self.net.append(SineLayer(self.units[counter],
#                                       self.out_features,
#                                       is_first=False,
#                                       omega_0=self.hidden_omega_0))


#     def init_weights(self):
#         return tf.keras.initializers.RandomUniform(minval=-np.sqrt(6. / self.units[-1]) / self.hidden_omega_0,
#                                                    maxval= np.sqrt(6. / self.units[-1]) / self.hidden_omega_0)


#     def call(self, input_tensor):
#         x = input_tensor
#         for layer in self.net:
#             x = layer(x)

#         return x
