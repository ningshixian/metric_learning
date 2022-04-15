#! -*- coding:utf-8 -*-
# 通过R-Drop增强模型的泛化性能
# 数据集：IFLYTEK' 长文本分类 (https://github.com/CLUEbenchmark/CLUE)
# 博客：https://kexue.fm/archives/8496

import os
import re
import sys
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from keras import backend as K
from keras.layers import Lambda, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.losses import kullback_leibler_divergence as kld
from keras.constraints import unit_norm
from keras.models import Model
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator

sys.path.append("../")
import util
from margin_softmax import sparse_amsoftmax_loss

# sets random seed
seed = 123
random.seed(seed)
np.random.seed(seed)
# set cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# set GPU memory
# 方法1:显存占用会随着epoch的增长而增长,也就是后面的epoch会去申请新的显存,前面已完成的并不会释放,为了防止碎片化
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 按需求增长
sess = tf.Session(config=config)
K.set_session(sess)

# specify the batch size and number of epochs
LR = 2e-5  # [3e-4, 5e-5, 2e-5] 默认学习率是0.001
SGD_LR = 0.001
warmup_proportion = 0.1  # 学习率预热比例
weight_decay = 0.01  # 权重衰减系数，类似模型正则项策略，避免模型过拟合
DROPOUT_RATE = 0.3  # 0.1
batch_size = 32     # 16
EPOCHS = 25
maxlen = 64  # 最大不能超过512, 若出现显存不足，请适当调低这一参数
EB_SIZE = 128
scale, margin = 30, 0.15  # amsoftmax参数 30, 0.35
optimizer = "adam"  # "sgd" "adamW" "adamwlr":带权重衰减和warmup的优化器
kid2label, label2kid = {}, {} # kid转换成递增的id格式
cls_num = None

# BERT base
config_path = '../corpus/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../corpus/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../corpus/chinese_L-12_H-768_A-12/vocab.txt'

domain_dict = {
    "rs": "LXHRSDH",
    "xz": "LXHXZDH",
    "cw": "LXHCWDH",
    "it": "LXHITDH",
    "c2": "C2ZXKF",
    "c3": "C3GYKF",
    "c4": "ZHJKF",
}
gen_train_file = "train_data/train.csv"
model_path = "model/best_model.weights"
# 模型组件初始化
drop_layer = Dropout(DROPOUT_RATE)
bn_layer = BatchNormalization()
lambda_layer = Lambda(lambda x: x[:, 0])
dense_layer = Dense(EB_SIZE)  # activation="relu" --> loss下降快


def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    df = pd.read_csv(
        filename, header=0, sep=",", encoding="utf-8", engine="python"
    )
    # 加了global，则可以在函数内部对函数外的对象进行操作了，也可以改变它的值了
    global kid2label, label2kid
    global cls_num
    for kid in df["knowledge_id"]:
        kid2label.setdefault(kid, len(kid2label)+1)
    label2kid = {v: k for k, v in kid2label.items()}
    cls_num = len(kid2label)

    for index, row in df.iterrows():
        text, label = row['question'], kid2label[row['knowledge_id']]
        D.append((text, int(label)))
    return D

# 加载数据集
train_data = load_data(gen_train_file)  # [:3200]
valid_data = []
# print(train_data[::5][:10])

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            for i in range(2):
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = None

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    dropout_rate=0.3,
    return_keras_model=False,
)
output = Lambda(lambda x: x[:, 0])(bert.model.output)
embedding = output

output = Dropout(DROPOUT_RATE, name="dp1")(output)
output = Lambda(lambda x: K.l2_normalize(x, 1))(output)  # 特征归一化（l2正则）√
output = Dense(
    units=cls_num,
    name="dense_output",
    use_bias=False,  # √
    kernel_constraint=unit_norm(),    # 权重归一化√
    kernel_initializer=bert.initializer
)(output)
# output = Dense(
#     units=cls_num,
#     activation='softmax',
#     kernel_initializer=bert.initializer
# )(output)

encoder = Model(bert.model.input, embedding) # 最终的目的是要得到一个编码器
model = Model(bert.model.input, output) # 用分类问题做训练
model.summary()


# def crossentropy_with_rdrop(y_true, y_pred, alpha=4):
#     """配合R-Drop的交叉熵损失
#     """
#     y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
#     y_true = K.cast(y_true, 'int32')
#     loss1 = K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))
#     loss2 = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
#     return loss1 + K.mean(loss2) / 4 * alpha


def crossentropy_with_rdrop(y_true, y_pred, alpha=4):
    """配合R-Drop的交叉熵损失
    """
    loss1 = K.mean(sparse_amsoftmax_loss(y_true, y_pred, scale, margin))
    loss2 = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
    return loss1 + K.mean(loss2) / 4 * alpha


model.compile(
    loss=crossentropy_with_rdrop,
    optimizer=Adam(lr=LR, clipvalue=1.0),
    metrics=['sparse_categorical_accuracy'], 
)


# def evaluate(data):
#     total, right = 0., 0.
#     for x_true, y_true in data:
#         y_pred = model.predict(x_true).argmax(axis=1)
#         y_true = y_true[:, 0]
#         total += len(y_true)
#         right += (y_true == y_pred).sum()
#     return right / total


# class Evaluator(keras.callbacks.Callback):
#     """评估与保存
#     """
#     def __init__(self):
#         self.best_val_acc = 0.

#     def on_epoch_end(self, epoch, logs=None):
#         val_acc = 0
#         for domain, valid_data in valid_generator.items():
#             val_acc += evaluate(valid_data)
#         val_acc = val_acc/len(valid_generator)

#         if val_acc > self.best_val_acc:
#             self.best_val_acc = val_acc
#             model.save_weights('best_model.weights')
#         print(
#             u'val_acc: %.5f, best_val_acc: %.5f\n' %
#             (val_acc, self.best_val_acc)
#         )


def clean(x):
    """预处理：去除文本的噪声信息"""
    x = re.sub('"', "", x)
    x = re.sub("\s", "", x)  # \s匹配任何空白字符，包括空格、制表符、换页符等
    x = re.sub(",", "，", x)
    return x.lower().strip()


def clean_sim(x):
    x = re.sub(r"(\t\n|\n)", "", x)
    x = x.strip().strip("###").replace("######", "###")
    return x.split("###")


def create_test_pairs(test):
    """Positive and negative pair creation.
    Alternates between positive and negative pairs.
    """
    test_data_dict = {}
    for index, row in test.iterrows():
        query = clean(row["user_input"])
        answer_id = clean_sim(row["answer_id"])
        recall = clean_sim(row["recall"])
        recall_id = clean_sim(row["recall_id"])
        test_data_dict.setdefault(query, {})
        test_data_dict[query]["kid"] = answer_id
        test_data_dict[query]["candidate"] = recall
        test_data_dict[query]["cid"] = recall_id
    return test_data_dict


def seq_padding(ML):
    """将序列padding到同一长度, value=0, mode='post'"""

    def func(X, padding=0):
        return np.array(
            [
                np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x
                for x in X
            ]
        )

    return func


def compute_pair_input_arrays(input_arrays):
    inp1_1, inp1_2 = [], []
    for instance in input_arrays:
        x1, x2 = tokenizer.encode(instance, maxlen=maxlen)
        inp1_1.append(x1)
        inp1_2.append(x2)
    # print(np.array(inp1_1).shape, np.array(inp1_2).shape)
    L = [len(x) for x in inp1_1]
    ML = max(L) if L else 0

    pad_func = seq_padding(ML)
    res = [inp1_1, inp1_2]
    res = list(map(pad_func, res))
    return res


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, encoder=None):
        self.encoder = encoder
        self.f1_avg = 0

    def on_train_begin(self, logs={}):
        self.valid_predictions = []

    def on_epoch_end(self, epoch, logs={}):
        f1_avg = []
        for domain in domain_dict:
            # 获取测试数据
            test_file = "train_data/{}_test.csv".format(domain)
            df_test = pd.read_csv(
                test_file, header=0, sep=",", encoding="utf-8", engine="python"
            )  # 防止乱码
            test_data_dict = create_test_pairs(df_test)
            # 模型预测和评估
            n, n1, n2 = 0, 0, 0  # 问题总数、回复的问题数、回复的正确问题数
            nb_kid_not_in_cid = 0
            for query, info_dict in tqdm(test_data_dict.items()):
                n += 1
                kids = info_dict["kid"]
                all_cand_text = info_dict["candidate"]
                all_cand_index = info_dict["cid"]
                if all_cand_index:
                    n1 += 1
                # print([query])
                # print(all_cand_text)
                all_cand_text_ids = compute_pair_input_arrays(all_cand_text)
                test_x = compute_pair_input_arrays([query])
                test_query_vec = self.encoder.predict(test_x)
                all_cand_vecs = self.encoder.predict(all_cand_text_ids)
                # print(test_query_vec.shape)  # (1, 768)
                # print(all_cand_vecs.shape)  # (10, 768)
                dot_list = cosine_similarity(all_cand_vecs, test_query_vec)
                dot_list = [x[0] for x in dot_list]

                # top1预测结果
                max_idx = np.argmax(dot_list)
                pred_one = str(int(all_cand_index[max_idx]))  # 预测的kid
                if pred_one in kids:
                    n2 += 1
                else:
                    if not set(kids) & set(all_cand_index):
                        nb_kid_not_in_cid += 1

            acc = round(n2 / n, 4)
            p = round(n2 / n1, 4)
            r = round(n1 / n, 4)
            f1 = round(2 * p * r / (p + r), 4)
            f1_avg.append(f1)
            print("\ndomain: {}".format(domain))
            print("问题总数: {}、回复的问题数: {}、回复的正确问题数: {}".format(n, n1, n2))
            print("how much kid not in candidate set: {}".format(nb_kid_not_in_cid))
            print("acc=n2/n= {}".format(acc))
            print("precision=n2/n1= {}".format(p))
            print("recall=n1/n= {}".format(r))
            print("f1=2pr/(p+r)= {}".format(f1))

        f1_avg = sum(f1_avg) / len(f1_avg)
        print("\nf1_avg=sum/len= {}".format(f1_avg))
        if f1_avg > self.f1_avg:
            print("epoch:{} 当前最佳average f1-score！".format(epoch+1))
            self.f1_avg = f1_avg
            self.encoder.save_weights(model_path)
        print('\n')


def predict_to_file(in_file, out_file):
    """输出预测结果到文件
    结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    fw = open(out_file, 'w')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            text = l['sentence']
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            label = model.predict([[token_ids], [segment_ids]])[0].argmax()
            l = json.dumps({'id': str(l['id']), 'label': str(label)})
            fw.write(l + '\n')
    fw.close()


if __name__ == '__main__':

    # evaluator = Evaluator()
    evaluator = CustomCallback(
        encoder=encoder,
    )

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model.weights')
    # predict_to_file('/root/CLUE-master/baselines/CLUEdataset/iflytek/test.json', 'iflytek_predict.json')
