import os
from base.base_model import BaseModel
from bert4keras.optimizers import *
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.losses import kullback_leibler_divergence as kld
from utils.utils import get_bert_model


def crossentropy_with_rdrop(y_true, y_pred, alpha=4):
    """配合R-Drop的交叉熵损失
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    loss1 = K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))
    loss2 = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
    return loss1 + K.mean(loss2) / 4 * alpha


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
        self.num_classes = 119
        self.build_model()

    def build_model(self):
        # 加载预训练模型
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))

        bert = get_bert_model(self.config.model.root)
        x = bert_model([x1_in, x2_in])
        output = Lambda(lambda x: x[:, 0])(x)
        output = Dense(
            units=self.num_classes,
            activation='softmax',
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            # kernel_initializer=bert.initializer
        )(output)

        self.train_model = keras.models.Model([x1_in, x2_in], output)
        self.train_model.summary()

        self.train_model.compile(
            loss=crossentropy_with_rdrop,
            optimizer=self.opt_dict[self.config.model.optimizer],
            metrics=['sparse_categorical_accuracy'],
        )

