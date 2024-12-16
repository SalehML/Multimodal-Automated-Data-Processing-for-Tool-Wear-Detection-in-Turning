#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:01:03 2024

@author: svalizad
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

num_classes = 3

input_shape = (224, 224, 3)

lr = 0.001
weight_decay = 0.001
batch_size = 12
num_epochs = 100
patch_size = 48
image_size = 224
num_patches = (image_size // patch_size) ** 2

projection_dim = 64
num_heads = 4
transformer_units = [
    
    projection_dim * 2,
    projection_dim,
    
    ]

transformer_layers = 3
mlp_head_units = [128]

def mlp(x, hidden_units, dropout_rate):
    
    for units in hidden_units:
        
        x = layers.Dense(units,activation = tf.nn.gelu)(x)
        
        x = layers.Dropout(dropout_rate)(x)
        
    return x


class Patches(layers.Layer):
    
    def __init__(self,patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        
    def call(self,images):
        
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images = images, 
                                           sizes = [1, self.patch_size, self.patch_size, 1],
                                           strides = [1, self.patch_size, self.patch_size, 1],
                                           rates = [1, 1, 1, 1], 
                                           padding = "VALID",
                                           )
        
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(layers.Layer):
        
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder,self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units = projection_dim)
        self.position_embedding = layers.Embedding(input_dim = num_patches, output_dim = projection_dim)
        
    def call(self, patch):
        positions = tf.range(start = 0, limit = self.num_patches, delta = 1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    

def create_vit_classifier():
    
    inputs = layers.Input(shape = input_shape)
    
    patches =  Patches(patch_size)(inputs)
    
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    for _ in range(transformer_layers):
        
        x1 = layers.LayerNormalization(epsilon = 1e-6)(encoded_patches)
        
        attention_output = layers.MultiHeadAttention(num_heads = num_heads, key_dim = projection_dim, dropout = 0.25)(x1, x1)
        
        x2 = layers.Add()([attention_output, encoded_patches])
        
        x3 = layers.LayerNormalization(epsilon = 1e-6)(x2)
        
        x3 = mlp(x3, hidden_units = transformer_units, dropout_rate=0.25)
        
        encoded_patches = layers.Add()([x3, x2])
        
    representation = layers.LayerNormalization(epsilon = 1e-6)(encoded_patches)
    
    representation = layers.Flatten()(representation)
    
    #representation = layers.Dropout(0.5)(representation)
    
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    
    logits = layers.Dense(num_classes)(features)
    
    model = keras.Model(inputs = inputs, outputs = logits)
    
    model_II = keras.Model(inputs = inputs, outputs = representation)
    
    model_III = keras.Model(inputs = inputs, outputs = encoded_patches)
    
    return model_III

    
        