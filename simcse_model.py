import os
from base.base_model import BaseModel
from keras.optimizers import RMSprop, SGD, Nadam
from bert4keras.optimizers import *
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from utils.utils import get_bert_model


def simcse_loss(y_true, y_pred):
    """用于SimCSE训练的loss
    y_true只是凑数的，并不起作用。因为真正的y_true是通过batch内数据计算得出的。
    y_pred就是batch内的每句话的embedding，通过bert编码得来
    """
    # 构造标签
    # idxs = [0,1,2,3,4,5]
    idxs = K.arange(0, K.shape(y_pred)[0])
    # 给idxs添加一个维度，idxs_1 = [[0,1,2,3,4,5]]
    idxs_1 = idxs[None, :]
    # 获取每句话的同义句id，即
    # 如果一个句子id为奇数，那么和它同义的句子的id就是它的上一句，如果一个句子id为偶数，那么和它同义的句子的id就是它的下一句
    # idxs_2 = [ [1], [0], [3], [2], [5], [4] ]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    # 生成计算loss时可用的标签
    # y_true = [[0,1,0,0,0,0],[1,0,0,0,0,0],[0,0,0,1,0,0],[0,0,1,0,0,0],[0,0,0,0,0,1],[0,0,0,0,1,0]]
    y_true = K.equal(idxs_1, idxs_2)
    y_true = K.cast(y_true, K.floatx())
    # 计算相似度
    # 首先对句向量各个维度做了一个L2正则，使其变得各项同性，避免下面计算相似度时，某一个维度影响力过大。
    y_pred = K.l2_normalize(y_pred, axis=1)
    # 其次，计算batch内每句话和其他句子的内积相似度(其实就是余弦相似度)
    similarities = K.dot(y_pred, K.transpose(y_pred))
    # 然后，将矩阵的对角线部分变为0，代表每句话和自身的相似性并不参与运算
    similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12
    # 最后，将所有相似度乘以20，这个目的是想计算softmax概率时，更加有区分度
    similarities = similarities * 20
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)


class SimcseModel(BaseModel):
    def __init__(self, config):
        super(SimcseModel, self).__init__(config)
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
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))

        bert_model = get_bert_model(self.config.model.root)
        x = bert_model([x1_in, x2_in])
        first_token = Lambda(lambda x: x[:, 0])(x)
        embedding = first_token

        self.train_model = Model([x1_in, x2_in], embedding) # 用分类问题做训练
        
        self.train_model.summary()
        self.train_model.compile(
            loss=simcse_loss, 
            optimizer=self.opt_dict[self.config.model.optimizer],
            # metrics=["sparse_categorical_accuracy"]  # "acc"
        )
