import os
import cv2
import numpy as np

import keras
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model


def get_model():
    inputs = Input(shape=(60, 160, 3))
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
    y = [[], [], [], []]
    for f in os.listdir(image_path):
        img = cv2.imread(image_path + f)
        img = (img - 128) / 128.0
        x.append(img)

        for i, _ in enumerate(f.split(".")[0]):
            y[i].append(keras.utils.to_categorical(int(_), 10))

    return np.array(x), np.array(y)


def train(x_train, y_train, x_test, y_test):
    model = get_model()
    model.load_weights("./log/checkpoint-05-2.4127.hdf5")
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', loss_weights=[.5, 1., 1., .6, ],
                  metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir="./log/", write_images=True)
    checkpoint = ModelCheckpoint(filepath="./log/checkpoint-{epoch:02d}-{val_loss:.4f}.hdf5", monitor='val_loss',
                                 verbose=1, save_best_only=True)

    history = model.fit(x_train, [y for y in y_train],
                        validation_data=(x_test, [y for y in y_test]),
                        epochs=10, batch_size=16,
                        callbacks=[checkpoint, tensorboard])


def test(x_test, y_test):
    label_list = np.argmax(y_test, axis=2)

    model = get_model()
    model.load_weights("./log/checkpoint-05-2.4127.hdf5")

    total_num = 0
    true_num = 0
    false_num = 0

    for i in range(x_test.shape[0]):
        y_pred = model.predict(np.expand_dims(x_test[i], 0))

        label = "".join([str(x) for x in label_list[:, i]])
        pred = "".join([str(x) for x in np.argmax(np.array(y_pred), axis=2)[:, 0]])

        total_num += 1
        if label == pred:
            true_num += 1
        else:
            false_num += 1

        print("label: %s, pred: %s"%(label, pred))

    print("total_num", total_num)
    print("true_num", true_num)
    print("false_num", false_num)
    print("accuracy: %.4f"%(true_num * 1. / total_num))



if __name__ == '__main__':
    train_path = "images/train/"
    test_path = "images/test/"

    # x_train, y_train = get_data(train_path)
    # x_test, y_test = get_data(test_path)
    # np.save("x_train", x_train)
    # np.save("y_train", y_train)
    # np.save("x_test", x_test)
    # np.save("y_test", y_test)

    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")
    # test(x_test, y_test)
    train(x_train, y_train, x_test, y_test)
# 8000/8000 [==============================] - 602s 75ms/step - loss: 7.3397 - num1_loss: 1.6631 - num2_loss: 1.9630 - num3_loss: 1.9488 - num4_loss: 1.7649 - num1_acc: 0.4068 - num2_acc: 0.3076 - num3_acc: 0.3060 - num4_acc: 0.3762 - val_loss: 5.1088 - val_num1_loss: 0.9981 - val_num2_loss: 1.4534 - val_num3_loss: 1.4944 - val_num4_loss: 1.1629 - val_num1_acc: 0.6995 - val_num2_acc: 0.5090 - val_num3_acc: 0.5030 - val_num4_acc: 0.6215
# 8000/8000 [==============================] - 591s 74ms/step - loss: 4.9831 - num1_loss: 0.9413 - num2_loss: 1.4667 - num3_loss: 1.4706 - num4_loss: 1.1045 - num1_acc: 0.6799 - num2_acc: 0.4999 - num3_acc: 0.4983 - num4_acc: 0.6152 - val_loss: 3.4476 - val_num1_loss: 0.5493 - val_num2_loss: 1.0566 - val_num3_loss: 1.1152 - val_num4_loss: 0.7265 - val_num1_acc: 0.8590 - val_num2_acc: 0.6815 - val_num3_acc: 0.6690 - val_num4_acc: 0.7775
# 8000/8000 [==============================] - 591s 74ms/step - loss: 3.8495 - num1_loss: 0.6338 - num2_loss: 1.1808 - num3_loss: 1.2144 - num4_loss: 0.8204 - num1_acc: 0.7898 - num2_acc: 0.5981 - num3_acc: 0.6034 - num4_acc: 0.7299 - val_loss: 2.8211 - val_num1_loss: 0.3839 - val_num2_loss: 0.8741 - val_num3_loss: 0.9902 - val_num4_loss: 0.5729 - val_num1_acc: 0.9005 - val_num2_acc: 0.7360 - val_num3_acc: 0.6995 - val_num4_acc: 0.8220
# 8000/8000 [==============================] - 620s 77ms/step - loss: 3.1858 - num1_loss: 0.5058 - num2_loss: 0.9834 - num3_loss: 1.0306 - num4_loss: 0.6660 - num1_acc: 0.8279 - num2_acc: 0.6728 - num3_acc: 0.6685 - num4_acc: 0.7749 - val_loss: 2.4127 - val_num1_loss: 0.3164 - val_num2_loss: 0.7674 - val_num3_loss: 0.8636 - val_num4_loss: 0.4653 - val_num1_acc: 0.9150 - val_num2_acc: 0.7645 - val_num3_acc: 0.7495 - val_num4_acc: 0.8650
