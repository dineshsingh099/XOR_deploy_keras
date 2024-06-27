import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from keras.models import Model
from src.config import config

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import src.preprocessing.preprocessor as pp
from src.preprocessing.data_management import load_dataset, save_model, load_model

import src.pipeline as pi

functional_nn = pi.functional_mlp()

def binary_cross_entropy_loss(Y_hat,Y_true):
    
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=Y_true,y_pred=Y_hat))

        
def training_loop():
    for e in range(config.epochs):
        for X_train_mb, Y_train_mb in pp.training_data_generator():

            with tf.GradientTape() as tape:

                Y_pred = functional_nn(X_train_mb, training=True)
                loss_func = binary_cross_entropy_loss(Y_pred,Y_train_mb)

            gradients = tape.gradient(loss_func,functional_nn.trainable_weights)
            config.optimizer.apply_gradients(zip(gradients,functional_nn.trainable_weights))

        print("Epoch # {}, Loss Function Value = {}".format(e+1,loss_func))
        if loss_func < 0.0001:
            break
        
if __name__ == "__main__":
    training_loop()
    save_model(functional_nn)