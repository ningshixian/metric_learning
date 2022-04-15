import os
import re
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import codecs
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from base.base_model import BaseModel
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, SGD, Nadam
from keras.regularizers import l2
from keras.constraints import unit_norm
from keras import backend as K
# from bert4keras.bert import build_bert_model
# from bert4keras.layers import LayerNormalization
from bert4keras.optimizers import (
    Adam,
    extend_with_weight_decay,
    extend_with_exponential_moving_average,
)
from bert4keras.optimizers import *
from bert4keras.backend import keras, set_gelu
from utils.margin_softmax import sparse_amsoftmax_loss
from utils.utils import get_bert_model

# set_gelu('tanh')  # 切换gelu版本


class SiameseResortModel(BaseModel):
    def __init__(self, config):
        super(SiameseResortModel, self).__init__(config)
        # 优化器选择
        self.opt_dict = {
            "sgd": SGD(lr=self.config.model.learning_rate, decay=1e-5, momentum=0.9, nesterov=True),
            "adam": Adam(lr=self.config.model.learning_rate, clipvalue=1.0),
            "nadam": Nadam(lr=self.config.model.learning_rate, clipvalue=1.0),
            "rmsprop": RMSprop(lr=self.config.model.learning_rate, clipvalue=1.0),
            "adamw": extend_with_weight_decay(Adam, "AdamW")(self.config.model.learning_rate, weight_decay_rate=0.01),
            "adamlr": extend_with_piecewise_linear_lr(Adam, "AdamLR")(learning_rate=self.config.model.learning_rate, lr_schedule={1000: 1.0}),
            "adamga": extend_with_gradient_accumulation(Adam, "AdamGA")(learning_rate=self.config.model.learning_rate, grad_accum_steps=10),
            "adamla": extend_with_lookahead(Adam, "AdamLA")(learning_rate=self.config.model.learning_rate, steps_per_slow_update=5, slow_step_size=0.5),
            "adamlo": extend_with_lazy_optimization(Adam, "AdamLO")(learning_rate=self.config.model.learning_rate, include_in_lazy_optimization=[]),
            "adamwlr":extend_with_piecewise_linear_lr(extend_with_weight_decay(Adam, "AdamW"), "AdamWLR")(
                learning_rate=self.config.model.learning_rate, weight_decay_rate=0.01, lr_schedule={1000: 1.0}
            ),
            "adamema": extend_with_exponential_moving_average(Adam, name="AdamEMA")(self.config.model.learning_rate, ema_momentum=0.9999)
        }
            
        self.build_model()

    """基于曼哈顿空间距离计算两个字符串语义空间表示相似度计算"""

    def exponent_neg_manhattan_distance(self, inputX):
        (sent_left, sent_right) = inputX
        return K.exp(-K.sum(K.abs(sent_left - sent_right), axis=1, keepdims=True))

    """基于欧式距离的字符串相似度计算"""

    def euclidean_distance(self, vects):
        x, y = vects
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    """余弦距离"""

    def cosine_distance(self, vests):
        x, y = vests
        x = K.l2_normalize(x, axis=-1)
        y = K.l2_normalize(y, axis=-1)
        return -K.mean(x * y, axis=-1, keepdims=True)

    def cos_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0],1)

    """对比损失"""

    def contrastive_loss(self, y_true, y_pred, margin=1):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    """搭建编码层网络,用于权重共享"""

    def create_base_network(self, input_shape):
        input = Input(shape=input_shape)
        # output = Dense(128, activation='relu')(input)  # 降维，减少距离度量的计算量（效果下降）
        # output = Dropout(DROPOUT_RATE)(output)
        # output = Dense(128, activation='relu')(output)
        output = input
        return Model(input, output)

    """搭建网络"""

    def build_model(self):
        left_input = Input(shape=(768,))
        right_input = Input(shape=(768,))

        shared_module = self.create_base_network(input_shape=(768,))
        left_output = shared_module(left_input)
        right_output = shared_module(right_input)

        # 2分类任务：语义匹配任务: 相似、不相似 
        sub_a = Lambda(lambda x: x[0] - x[1])([left_output, right_output])
        mul_a = Lambda(lambda x: x[0] * x[1])([left_output, right_output])
        x = keras.layers.concatenate([left_output, right_output, sub_a, mul_a])
        # x = keras.layers.Dropout(DROPOUT_RATE)(x)
        # x = Lambda(lambda x: K.l2_normalize(x, 1))(x)  # l2正则
        # x = BatchNormalization()(x)  # 加速收敛、防止过拟合
        x = Dense(units=256, activation='relu')(x)
        x = Dense(units=128, activation='relu')(x)
        distance = Dense(1, activation="sigmoid")(x)

        # # 度量学习
        # # distance = Lambda(self.cosine_distance, output_shape=self.cos_dist_output_shape)([left_output, right_output])
        # distance = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([left_output, right_output])

        self.train_model = Model([left_input, right_input], distance)
        self.train_model.compile(
            # loss=contrastive_loss,
            loss="binary_crossentropy",
            optimizer=self.opt_dict[self.config.model.optimizer],
            metrics=["acc", "sparse_categorical_accuracy"],
        )
        self.train_model.summary()
