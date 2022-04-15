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


class BimpmResortModel(BaseModel):
    def __init__(self, config):
        super(BimpmResortModel, self).__init__(config)
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

    def build_model(self):
        a = Input(shape=(None,), name="anchor_input")
        b = Input(shape=(None,))

        c = Input(shape=(None,), name="sample_input")
        d = Input(shape=(None,))

        bert_model = get_bert_model(self.config.model.root)
        embed_left = bert_model([a, b])
        embed_right = bert_model([c, d])
        # TypeError : MultiPerspective layer does not support masking in Keras
        # 解决方案 https://blog.csdn.net/yscoder/article/details/99995102
        embed_left = Lambda(lambda x: x, output_shape=lambda s:s)(embed_left)
        embed_right = Lambda(lambda x: x, output_shape=lambda s:s)(embed_right)

        # ----- Context Representation Layer ----- 
        bilstm = keras.layers.Bidirectional(keras.layers.LSTM(
            self._params['lstm_units'],
            return_sequences=True,
            dropout=self._params['dropout_rate']
        ))
        rep_left = bilstm(embed_left)
        rep_right = bilstm(embed_right)

        # ----- Matching Layer -----
        matching_layer = MultiPerspective(self._params['mp_dim'])
        matching_left = matching_layer([rep_left, rep_right])
        matching_right = matching_layer([rep_right, rep_left])

        # ----- Aggregation Layer -----
        agg_left = keras.layers.Bidirectional(keras.layers.LSTM(
            self._params['lstm_units'],
            return_sequences=False,
            dropout=self._params['dropout_rate']
        ))(matching_left)
        agg_right = keras.layers.Bidirectional(keras.layers.LSTM(
            self._params['lstm_units'],
            return_sequences=False,
            dropout=self._params['dropout_rate']
        ))(matching_right)
        
        # agg_left = keras.layers.GlobalAveragePooling1D()(matching_left)
        # agg_right = keras.layers.GlobalAveragePooling1D()(matching_right)
        
        aggregation = keras.layers.concatenate([agg_left, agg_right])
        aggregation = keras.layers.Dropout(rate=self._params['dropout_rate'])(aggregation)
        
        # ----- Prediction Layer -----
        # output = Dense(
        #     128,
        #     activation="relu",
        #     kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),  # bert_model.initializer,
        #     kernel_regularizer=keras.regularizers.l2(0.001),    # l2正则
        # )(aggregation)
        # output = Dropout(rate=self._params['dropout_rate'])(output)
        x_out = Dense(units=1, activation="sigmoid")(aggregation)  #  原论文activation='linear'

        self.train_model = Model(inputs=[a, b, c, d], outputs=x_out)
        self.train_model.compile(
            # loss=contrastive_loss,
            loss="binary_crossentropy",
            optimizer=self.opt_dict[self.config.model.optimizer],
            metrics=["acc", "sparse_categorical_accuracy"],
        )
        self.train_model.summary()
