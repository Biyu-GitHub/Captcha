'''
@Author: by
@Date: 2019-07-08 14:04:11
@LastEditTime: 2019-07-12 11:06:08
'''

import argparse
import os
import string

import cv2
import h5py


def get_args():
    parser = argparse.ArgumentParser(description="Convert captcha images")

    parser.add_argument("--path", metavar="PATH", default=None,
                        help="Captcha images output path.")

    parser.add_argument("--out", metavar="OUT", default=None,
                        help="Captcha images output path.")

    args = parser.parse_args()

    return args


def convert(shape=(80, 170, 3), class_num=36, char_num=4):
    '''
    Convert image and label to .h5py.
    :param input_dir: Images path.
    :param output_file: Output filename.
    :param shape: Input imaghe shape = (height, width, channel).
    :param class_num: Classex numbers.
    :param char_num: The number of every captch contains characters.
    :return: None
    '''
    args = get_args()

    input_dir = args.path
    output_file = args.out

    if os.path.exists(output_file):
        os.remove(output_file)

    input_h, input_w, input_c = shape

    characters = string.digits + string.ascii_uppercase

    with h5py.File(output_file, 'w') as f:
        f.create_dataset("input", (1024, input_h, input_w, input_c),
                         maxshape=(None, input_h, input_w, input_c),
                         dtype='float32')

        f.create_dataset("label", (char_num, 1024, class_num),
                         maxshape=(char_num, None, class_num),
                         dtype='int32')

    img_list = os.listdir(input_dir)
    img_num = len(img_list)

    count = 0
    h5f = h5py.File(output_file, 'a')

    for j, f in enumerate(img_list):
        print("\r%d/%d : %s" % (j, img_num, f), end="")
        if count >= h5f['input'].shape[0]:
            input_nums = h5f['input'].shape[0] * 2
            h5f['input'].resize((input_nums, input_h, input_w, input_c))
            h5f['label'].resize((char_num, input_nums, class_num))

        full_path = os.path.join(input_dir, f)

        img = cv2.imread(full_path)
        img = img / 256.

        label = f.split(".")[0]

        h5f["input"][count] = img

        for i, _ in enumerate(label):
            h5f["label"][i, count, characters.find(_)] = 1

        count += 1

    h5f['input'].resize((count, input_h, input_w, input_c))
    h5f['label'].resize((4, count, 36))
    h5f.close()

    print("\nDone!")


if __name__ == "__main__":
    convert()
