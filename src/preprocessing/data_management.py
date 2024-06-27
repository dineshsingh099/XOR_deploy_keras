import os 
import pandas as pd
import pickle

from src.config import config

def load_dataset(file_name):
    
    file_path = os.path.join(config.DATAPATH,file_name)
    data = pd.read_csv(file_path)
    return data

def save_model(functional_nn):
    
    pkl_file_path = os.path.join(config.SAVED_MODEL_PATH,"custom_xor_nn_keras.pkl")
    
    weights_and_biases = {}
    for layer in functional_nn.layers:
        weights_and_biases[layer.name] = layer.get_weights()
    
    with open(pkl_file_path,"wb") as file_handle:
        pickle.dump(weights_and_biases, file_handle)
        
    print("Saved model with file name {} at {}".format("custom_xor_nn_keras.pkl",config.SAVED_MODEL_PATH))
    

def load_model(file_name):
    pkl_file_path = os.path.join(config.SAVED_MODEL_PATH,file_name)
    
    with open(pkl_file_path,"rb") as file_handle:
        loaded_model = pickle.load(file_handle)
               
    return loaded_model