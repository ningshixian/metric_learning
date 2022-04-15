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


class AmsoftmaxNluRecallModel(BaseModel):
    def __init__(self, config):
        super(AmsoftmaxNluRecallModel, self).__init__(config)
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
        cls_num_path = os.path.join(
            self.config.callbacks.checkpoint_dir, 
            self.config.data_loader.cls_num_file,
        )
        with open(cls_num_path, "r", encoding="utf-8") as f:
            self.cls_num = int(f.read())
            
        self.build_model()

    def build_model(self):
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        r_out = Input(shape=(None,), dtype="int32")

        bert_model = get_bert_model(self.config.model.root)
        x = bert_model([x1_in, x2_in])
        first_token = Lambda(lambda x: x[:, 0])(x)
        embedding = first_token
        first_token = Dropout(self.config.model.dropout_rate, name="dp1")(first_token)  #防止过拟合
        first_token = Lambda(lambda v: K.l2_normalize(v, 1))(first_token)  # 特征归一化（l2正则）√
        # embedding = first_token
        # first_token = Batchnormalization()(first_token)
        first_out = Dense(
            self.cls_num,
            name="dense_output",
            use_bias=False,  # no bias √
            kernel_constraint=unit_norm(),    # 权重归一化（单位范数（unit_form），限制权值大小为 1.0）√
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        )(first_token)
        p_out = first_out

        # encoder = Model([x1_in, x2_in], [p_out, embedding]) # 最终的目的是要得到一个编码器
        # train_model = Model([x1_in, x2_in, r_out], [p_out, embedding]) # 用分类问题做训练
        self.encoder = Model([x1_in, x2_in], embedding) # 最终的目的是要得到一个编码器
        self.train_model = Model([x1_in, x2_in, r_out], p_out) # 用分类问题做训练

        final_loss = sparse_amsoftmax_loss(r_out, p_out, self.config.model.scale, self.config.model.margin)
        self.train_model.add_loss(final_loss)
        self.train_model.summary()

        # compile the model
        print("[INFO] compiling model...")
        self.train_model.compile(
            optimizer=self.opt_dict[self.config.model.optimizer], 
            metrics=["sparse_categorical_accuracy"]  # "acc"
        )  # 用足够小的学习率
