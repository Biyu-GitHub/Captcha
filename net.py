'''
@Author: by
@Date: 2019-07-08 10:09:10
@LastEditTime: 2019-07-12 10:24:23
'''

from keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D, Flatten
from keras.models import Input, Model
from keras.utils.vis_utils import plot_model
import os

# def get_model(width=170, height=80, channel=3, char_num=4, class_num=36, save_model_img=True):
def get_model(shape=(80, 170, 3), class_num=36, char_num=4, save_model_img=True):
    '''
    Define the network and return a keras model.
    :param shape: Input imaghe shape = (height, width, channel).
    :param class_num: Classex numbers.
    :param char_num: The number of every captch contains characters.
    :param save_model_img: If or not save a net structure to an inage.
    :return: A keras model.
    '''
    
    input_tensor = Input(shape=shape)
    
    x = input_tensor

    for i in range(4):
        x = Conv2D(filters=32 * (i + 1), kernel_size=(3, 3), activation='relu')(x)
        x = Conv2D(32 * (i + 1), (3, 3), activation='relu')(x)
        x = BatchNormalization(axis=1)(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    x = [Dense(units=class_num, activation='softmax', name='c%d' % (i + 1))(x) for i in range(char_num)]

    model = Model(inputs=input_tensor, outputs=x)

    if save_model_img:
        plot_model(model=model, show_shapes=True)

    return model


if __name__ == '__main__':
    model = get_model()
