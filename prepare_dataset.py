'''
@Author: by
@Date: 2019-07-08 09:40:00
@LastEditTime: 2019-07-12 10:26:20
'''

from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import string
import random
import argparse
import os
import shutil
import cv2


def get_args():
    parser = argparse.ArgumentParser(description="Generate captcha images")

    parser.add_argument("--path", metavar="PATH", default="./images/",
                        help="Captcha images output path.")

    parser.add_argument("--nums", metavar="NUM", default=100000,
                        help="Captcha images output path.")

    args = parser.parse_args()

    return args


def generate_captcha(width=170, height=80, char_nums=4):
    '''
    Generate images default is 100k. Then random chose 20% as test.
    :param width: Image width.
    :param height: Image height.
    :param char_num: The number of every captch contains characters.
    :return: None.
    '''

    args = get_args()

    train_output_path = os.path.join(args.path, "train")
    test_output_path = os.path.join(args.path, "test")

    if not os.path.exists(train_output_path):
        os.makedirs(train_output_path)

    if not os.path.exists(test_output_path):
        os.makedirs(test_output_path)

    characters = string.digits + string.ascii_uppercase
    generator = ImageCaptcha(width, height)

    for i in range(int(args.nums)):
        print("\r[INFO] Preparing dataset ... %d/%s" % (i + 1, args.nums), end="")
        label = "".join([random.choice(characters) for j in range(char_nums)])
        generator.write(label, os.path.join(train_output_path, label + ".jpg"), format='jpeg')

    img_list = sorted(os.listdir(train_output_path))
    random.shuffle(img_list)
    train_nums = len(img_list)
    test_nums = int(train_nums * 0.2)

    for i in range(test_nums):
        shutil.move(os.path.join(train_output_path, img_list[i]), test_output_path)

    print("\n[INFO] Finished !")
    print("train_output_path:", os.path.abspath(train_output_path))
    print("test_output_path:", os.path.abspath(test_output_path))
    print("train_nums:", train_nums)
    print("test_nums:", test_nums)


if __name__ == '__main__':
    generate_captcha()
