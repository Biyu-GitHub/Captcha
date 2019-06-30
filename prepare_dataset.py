import os
import random
import argparse
import shutil
from captcha.image import ImageCaptcha


def get_args():
    parser = argparse.ArgumentParser(description="Generate captcha images")

    parser.add_argument("--path", metavar="PATH", default="./images/",
                        help="Captcha images output path.")

    args = parser.parse_args()

    return args


def generate_captcha():
    '''
    生成10k张由4位数字组成的验证码，并且划分出20%当作验证集
    :return:
    '''
    train_output_path = os.path.join(args.path, "train")
    test_output_path = os.path.join(args.path, "test")

    if not os.path.exists(train_output_path):
        os.makedirs(train_output_path)

    if not os.path.exists(test_output_path):
        os.makedirs(test_output_path)

    img = ImageCaptcha()

    for i in range(10000):
        captcha_text = str(i).zfill(4)
        output_name = os.path.join(train_output_path, captcha_text + ".jpg")
        img.write(captcha_text, output_name)

        print(output_name)

    img_list = sorted(os.listdir(train_output_path))
    random.shuffle(img_list)
    for i in range(int(len(img_list) * 0.2)):
        shutil.move(os.path.join(train_output_path, img_list[i]), test_output_path)


if __name__ == '__main__':
    args = get_args()
    generate_captcha()
