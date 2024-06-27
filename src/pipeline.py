from keras.layers import Input, Dense
from keras.models import Model

import src.preprocessing.preprocessor as pre

def functional_mlp():

    inp = Input(shape=(pre.X_train.shape[1],))
    first_hidden_out = Dense(units=4,activation="relu")(inp)
    second_hidden_out = Dense(units=2,activation="relu")(first_hidden_out)
    nn_out = Dense(units=1,activation="sigmoid")(second_hidden_out)

    return Model(inputs=[inp],outputs=[nn_out])