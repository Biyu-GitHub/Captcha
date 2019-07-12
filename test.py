'''
@Author: by
@Date: 2019-07-12 12:18:07
@LastEditTime: 2019-07-12 12:18:39
'''

import argparse
import string

import h5py
import numpy as np

import net

characters = string.digits + string.ascii_uppercase


def get_args():
    parser = argparse.ArgumentParser(description="Convert captcha images")

    parser.add_argument("--input", metavar="INPUT", default="./test.h5",
                        help="Captcha images output path.")

    parser.add_argument("--model", metavar="MODEL", default="./log/checkpoint-15-0.8206.hdf5",
                        help="Captcha images output path.")

    args = parser.parse_args()

    return args


def decode_label(y):
    ret = ""
    for index in np.argmax(y, 1):
        ret += characters[index]

    return ret


def decode_pre(y):
    ret = ""
    for index in np.argmax(np.array(y), 2)[:, 0]:
        ret += characters[index]

    return ret


def test():
    total_num = 0
    true_num = 0
    false_num = 0

    args = get_args()

    model = net.get_model()
    model.load_weights(args.model)

    h5f_test = h5py.File(args.input, 'r')
    x_test = h5f_test['input']
    y_test = h5f_test['label']

    for i in range(x_test.shape[0]):
        x = x_test[i]
        y = y_test[:, i, :]

        label = decode_label(y)
        y_pre = decode_pre(model.predict(np.expand_dims(x, 0)))

        print(label, y_pre, end=" ")

        if (label == y_pre):
            true_num += 1
            print("T")
        else:
            false_num += 1
            print("F")

        total_num += 1

    print("total_num", total_num)
    print("true_num", true_num)
    print("false_num", false_num)


if __name__ == '__main__':
    test()
