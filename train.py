'''
@Author: by
@Date: 2019-07-12 11:16:30
@LastEditTime: 2019-07-12 11:26:05
'''

import argparse

import h5py
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

import net


def get_args():
    parser = argparse.ArgumentParser(description="Convert captcha images")

    parser.add_argument("--train", metavar="TRAIN", default=None,
                        help="Captcha images output path.")

    parser.add_argument("--test", metavar="TEST", default=None,
                        help="Captcha images output path.")

    parser.add_argument("--epochs", metavar="EPOCHS", default=16,
                        help="Captcha images output path.")

    parser.add_argument("--batch_size", metavar="BATCH_SIZE", default=256,
                        help="Captcha images output path.")

    args = parser.parse_args()

    return args


def train():
    args = get_args()
    train_input = args.train
    test_input = args.test
    epochs = args.epochs
    batch_size = args.batch_size

    model = net.get_model()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-3, amsgrad=True),
                  metrics=['accuracy'])

    h5f_train = h5py.File(train_input, 'r')
    h5f_test = h5py.File(test_input, 'r')

    x_train = h5f_train['input']
    y_train = h5f_train['label']
    x_test = h5f_test['input']
    y_test = h5f_test['label']

    tensorboard = TensorBoard(log_dir="./log/", write_images=True)
    checkpoint = ModelCheckpoint(filepath="./log/checkpoint-{epoch:02d}-{val_loss:.4f}.hdf5", monitor='val_loss',
                                 verbose=1, save_best_only=True)

    model.fit(x_train, [y_train[0], y_train[1], y_train[2], y_train[3]],
              validation_data=(x_test, [y_test[0], y_test[1], y_test[2], y_test[3]]),
              epochs=epochs, batch_size=batch_size, shuffle='batch',
              callbacks=[checkpoint, tensorboard])


if __name__ == '__main__':
    train()
