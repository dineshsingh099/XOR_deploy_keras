import os
import pathlib
import src
from keras.optimizers import RMSprop

epochs = 700

mb_size = 2

optimizer = RMSprop(learning_rate=0.01)

PACKAGE_ROOT = pathlib.Path(src.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

SAVED_MODEL_PATH = os.path.join(PACKAGE_ROOT,"trained_models")

