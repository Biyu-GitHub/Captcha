import os
import cv2
import numpy as np

import keras
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model


def get_model():
    inputs = Input(shape=(60, 160, 1))
    net = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    net = Conv2D(32, (3, 3), activation='relu')(net)
    net = MaxPooling2D((2, 2))(net)
    net = Dropout(0.25)(net)

    net = Conv2D(64, (3, 3), padding='same', activation='relu')(net)
    net = Conv2D(64, (3, 3), activation='relu')(net)
    net = MaxPooling2D((2, 2))(net)
    net = Dropout(0.25)(net)

    net = Flatten()(net)

    net = Dense(512, activation='relu')(net)
    net = Dropout(0.5)(net)

    num1 = Dense(10, activation='softmax', name='num1')(net)
    num2 = Dense(10, activation='softmax', name='num2')(net)
    num3 = Dense(10, activation='softmax', name='num3')(net)
    num4 = Dense(10, activation='softmax', name='num4')(net)

    model = Model(inputs, [num1, num2, num3, num4])

    return model


def get_data(image_path):
    x = []
    y = []
    for f in os.listdir(image_path):
        img = cv2.imread(image_path + f, cv2.IMREAD_GRAYSCALE)
        img = (img - 128) / 128.0
        x.append(img)

        one_hot = []
        for _ in f.split(".")[0]:
            one_hot.append(keras.utils.to_categorical(int(_), 10))

        y.append(one_hot)

    return np.expand_dims(np.array(x), -1), np.array(y)


def train(x_train, y_train, x_test, y_test):
    model = get_model()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', loss_weights=[1., 1., 1., 1., ],
                  metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir="./log/", write_images=True)
    checkpoint = ModelCheckpoint(filepath="./log/checkpoint-{epoch:02d}-{val_loss:.4f}.hdf5", monitor='val_loss', verbose=1, save_best_only=True)

    history = model.fit(x_train, [y_train[:, 0, :], y_train[:, 1, :], y_train[:, 2, :], y_train[:, 3, :]],
              validation_data=(x_test, [y_test[:, 0, :], y_test[:, 1, :], y_test[:, 2, :], y_test[:, 3, :]]),
              epochs=10, batch_size=16,
              callbacks=[checkpoint, tensorboard])


if __name__ == '__main__':
    train_path = "images/train/"
    test_path = "images/test/"
    x_train, y_train = get_data(train_path)
    x_test, y_test = get_data(test_path)

    train(x_train, y_train, x_test, y_test)
