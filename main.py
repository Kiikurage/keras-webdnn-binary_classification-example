import os

import keras
import numpy as np
from webdnn.backend import generate_descriptor
from webdnn.frontend.keras import KerasConverter

model_path = './model.h5'


def train_model():
    # Multi-layer perceptron for XOR
    x = (np.random.rand(512, 2) - 0.5).astype(np.float32)
    y = keras.utils.to_categorical((np.sign(x[:, 0]) == np.sign(x[:, 1])).astype(np.int32), 2)

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(4, activation='relu', input_shape=(2,)))
    model.add(keras.layers.Dense(4, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))

    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(), metrics=["accuracy"])
    model.fit(x, y, epochs=50, batch_size=16)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    return model


def main():
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)

    else:
        model = train_model()

    graph = KerasConverter(batch_size=1).convert(model)
    generate_descriptor('webgl', graph).save('./output')


if __name__ == '__main__':
    main()
