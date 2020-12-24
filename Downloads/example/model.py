#encoding=utf-8
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.applications import MobileNet, MobileNetV2, VGG16
import tensorflow.keras.backend as K
from numpy.random import randint


def siamese(input_shape=(224, 224, 3)):
    input_1 = Input(shape=input_shape)
    input_2 = Input(shape=input_shape)
    base_model = MobileNet(input_shape=input_shape, weights='imagenet', include_top=False)

    # 两张输入图片用cnn提取出特征
    encoded_image_1 = base_model(input_1)
    encoded_image_2 = base_model(input_2)

    encoded_image_1 = Flatten()(encoded_image_1)
    encoded_image_2 = Flatten()(encoded_image_2)

    # 两组特征的L1距离作为下一层的特征
    l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    l1_distance = l1_distance_layer([encoded_image_1, encoded_image_2])

    # 全连接层计算两张图片是否相同
    out = Dense(512, activation='relu')(l1_distance)
    out = Dense(1, activation='sigmoid')(out)

    model = Model([input_1, input_2], out)

    return model
