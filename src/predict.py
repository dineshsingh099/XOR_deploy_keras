import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from src.config import config

from src.preprocessing.data_management import load_model
import src.train_pipeline 
import src.pipeline as pi


def predicts(x):
    data = load_model("custom_xor_nn_keras.pkl")
    functional_nn = pi.functional_mlp()

# Load weights and biases into the model
    for layer in functional_nn.layers:
        if layer.name in data:
            layer.set_weights(data[layer.name])
    
    return functional_nn.predict(x)

if __name__=="__main__":
    X_new = np.array([[0, 1], [1, 0],[0,0],[1,1]])    
    prediction = predicts(X_new)
    print(prediction)
    class_labels = (prediction >= 0.5).astype(int)
    print(class_labels)
    
    