#! -*- coding: utf-8 -*-
# SimCSE 中文测试

import sys
import pandas as pd
from tqdm import tqdm
import re
import os
import random
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.constraints import unit_norm
from keras.losses import kullback_leibler_divergence as kld
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.snippets import sequence_padding
import jieba
import string
from zhon.hanzi import punctuation

sys.path.append(r"./")
from margin_softmax import sparse_amsoftmax_loss

# from train_data.data import clean, clean_sim
sys.path.append(r"SimCSE/")
from utils import *

jieba.initialize()

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
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)
K.set_session(sess)

"""
python SimCSE/eval2.py LongforBERT cls LXH 0.3
nohup python SimCSE/eval2.py LongforBERT cls LXH 0.3 > logs/simcse.log 2>&1 &
"""

LR = 2e-5  # 1e-3
batch_size = 32  # 尽可能大
epochs = 25
t = 0.05  # 温度系数τ
scale, margin = 30, 0.35  # amsoftmax参数 0.15 0.25 0.35
domain_dict = {
    "it": "LXHITDH",
    "rs": "LXHRSDH",
    "xz": "LXHXZDH",
    "cw": "LXHCWDH",
    "c2": "C2ZXKF",
    "c3": "C3GYKF",
    "c4": "ZHJKF",
}

# 基本参数
if len(sys.argv[1:]) == 4:
    model_type, pooling, task_name, dropout_rate = sys.argv[1:]
else:
    model_type, pooling, task_name, dropout_rate = "BERT", "cls", "LXH", 0.3
assert model_type in [
    "BERT",
    "RoBERTa",
    "NEZHA",
    "WoBERT",
    "RoFormer",
    "BERT-large",
    "RoBERTa-large",
    "NEZHA-large",
    "SimBERT",
    "SimBERT-tiny",
    "SimBERT-small",
    "LongforBERT",
]
assert pooling in ["first-last-avg", "last-avg", "cls", "pooler"]
assert task_name in ["ATEC", "BQ", "LCQMC", "PAWSX", "STS-B", "LXH"]
dropout_rate = float(dropout_rate)

if task_name == "PAWSX":
    maxlen = 128
else:
    maxlen = 64

# # 加载数据集
# data_path = 'senteval_cn/'
# datasets = {
#     '%s-%s' % (task_name, f):
#     load_data('%s%s/%s.%s.data' % (data_path, task_name, task_name, f))
#     for f in ['train', 'valid', 'test']
# }


df = pd.read_csv(
    "SimCSE/LXH/train.csv", header=0, sep=",", encoding="utf-8", engine="python"
)
kid2label = {}  # kid转换成递增的id格式
for kid in df["knowledge_id"]:
    kid2label.setdefault(kid, len(kid2label))
    # kid2label.setdefault(kid, len(kid2label)+1)
datasets = df["question"]
# step, rows = 2, len(df["question"])
# step, rows = 2, 2000
# datasets = {
#     "LXH-train": [
#         list(df["question"])[i : i + step] + list(map(lambda x: kid2label[x], df["knowledge_id"][i : i + step]))
#         for i in tqdm(range(0, rows, step))
#     ]
# }
# print(datasets["LXH-train"][:5])

# bert配置
model_name = {
    "BERT": "chinese_L-12_H-768_A-12",
    "RoBERTa": "chinese_roberta_wwm_ext_L-12_H-768_A-12",
    "WoBERT": "chinese_wobert_plus_L-12_H-768_A-12",
    "NEZHA": "nezha_base_wwm",
    "RoFormer": "chinese_roformer_L-12_H-768_A-12",
    "BERT-large": "uer/mixed_corpus_bert_large_model",
    "RoBERTa-large": "chinese_roberta_wwm_large_ext_L-24_H-1024_A-16",
    "NEZHA-large": "nezha_large_wwm",
    "SimBERT": "chinese_simbert_L-12_H-768_A-12",
    "SimBERT-tiny": "chinese_simbert_L-4_H-312_A-12",
    "SimBERT-small": "chinese_simbert_L-6_H-384_A-12",
    "LongforBERT": "longforBERT_v4.1",
}[model_type]

config_path = "../corpus/%s/bert_config.json" % model_name
if model_type == "NEZHA":
    checkpoint_path = "../corpus/%s/model.ckpt-691689" % model_name
elif model_type == "NEZHA-large":
    checkpoint_path = "../corpus/%s/model.ckpt-346400" % model_name
else:
    checkpoint_path = "../corpus/%s/bert_model.ckpt" % model_name
dict_path = "../corpus/%s/vocab.txt" % model_name

# 建立分词器
if model_type in ["WoBERT", "RoFormer"]:
    tokenizer = get_tokenizer(
        dict_path, pre_tokenize=lambda s: jieba.lcut(s, HMM=False)
    )
else:
    tokenizer = get_tokenizer(dict_path)

# 建立 encoder 模型
if model_type == "RoFormer":
    pretrained_encoder = get_encoder(
        config_path,
        checkpoint_path,
        model="roformer",
        pooling=pooling,
        dropout_rate=dropout_rate,
    )
elif "NEZHA" in model_type:
    pretrained_encoder = get_encoder(
        config_path,
        checkpoint_path,
        model="nezha",
        pooling=pooling,
        dropout_rate=dropout_rate,
    )
else:
    pretrained_encoder = get_encoder(
        config_path, checkpoint_path, pooling=pooling, dropout_rate=dropout_rate
    )


with open("model/cls_num.txt", "r", encoding="utf-8") as f:  # 分类数cls_num的值保存到文件
    cls_num = int(f.read())


# 语料id化
all_token_ids, all_segment_ids, all_labels = [], [], []
all_labels = list(map(lambda x: kid2label[x], df["knowledge_id"]))  # 标签序列化
for d in tqdm(datasets):
    token_ids, segment_ids = tokenizer.encode(d, maxlen=maxlen)
    all_token_ids.append(token_ids)
    all_segment_ids.append(segment_ids)

train_token_ids = all_token_ids
train_segment_ids = all_segment_ids
train_labels = all_labels
# train_token_ids = sequence_padding(all_token_ids)
# train_segment_ids = sequence_padding(all_segment_ids)


if task_name != "PAWSX":
    train_token_ids = list(zip(train_token_ids, train_segment_ids, train_labels))
    np.random.shuffle(train_token_ids)
    # train_token_ids = train_token_ids[:10000]


class data_generator(DataGenerator):
    """训练语料生成器"""
    def __init__(self, data, batch_size=32):
        super().__init__(data, batch_size)

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (token_ids, segment_ids, labels) in self.sample(random):
            # token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            # 每个batch内，每一句话都重复一次
            for i in range(2):
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([labels])
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids, batch_labels], None
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def clean(x):
    """预处理：去除文本的噪声信息"""
    x = re.sub('"', "", str(x))
    x = re.sub("\s", "", x)  # \s匹配任何空白字符，包括空格、制表符、换页符等
    x = re.sub(",", "，", x)  # 方便存储为csv文件
    x = re.sub(
        "[{}]+$".format(string.punctuation + punctuation + " "), "", x
    )  # 干掉字符串结尾的中英文标点符号
    return x


def clean_sim(x):
    """预处理：切分相似问"""
    x = re.sub(r"(\t\n|\n)", "", x)
    x = x.strip().strip("###").replace("######", "###")
    return x.split("###")


def custom_ce_loss(from_logits):
    # 采用了闭包的方式，将参数传给 sparse_amsoftmax_loss，再调用 inner
    def inner(y_true, y_pred):
        return K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=from_logits
        )
    return inner


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
    # 温度系数τ=0.05
    similarities = similarities / t
    # from_logits=True的交叉熵自带softmax激活函数
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)


# From: https://github.com/bojone/r-drop
def rdrop_loss(y_true, y_pred, alpha=4):
    """loss从300多开始，需要epoch=50让其下降
    """
    loss = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
    return K.mean(loss) / 4 * alpha


def crossentropy_with_rdrop(y_true, y_pred, alpha=4):
    """配合R-Drop的交叉熵损失
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    loss1 = K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))
    # loss1 = K.mean(sparse_amsoftmax_loss(y_true, y_pred, scale, margin))
    loss2 = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
    return loss1 + K.mean(loss2) / 4 * alpha


# 模型构建
x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))
target = Input(shape=(None,), dtype="int32")
emb = pretrained_encoder([x1_in, x2_in])   # pooling='cls'
emb_norm = Lambda(lambda v: K.l2_normalize(v, 1))(emb)  # 特征归一化（l2正则,专供 amsoftmax 使用）√
# emb_norm = Dropout(DROPOUT_RATE, name="dp1")(emb_norm)   #防止过拟合
output = Dense(
    cls_num,
    # activation='softmax',
    use_bias=False,  # no bias √
    kernel_constraint=unit_norm(),  # 权重归一化（单位范数（unit_form），限制权值大小为 1.0）√
    kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
)(emb_norm)
r_output = Dense(
    units=cls_num,
    activation='softmax',
    kernel_constraint=unit_norm(),
    kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
)(emb)

encoder = Model([x1_in, x2_in], emb)  # 最终的目的是要得到一个编码器


# 自定义的loss
# 重点：把自定义的loss添加进层使其生效，同时加入metric方便在KERAS的进度条上实时追踪
am_loss = sparse_amsoftmax_loss(target, output, scale, margin)
sim_loss = simcse_loss(target, emb)
rdrop_loss = rdrop_loss(target, r_output)
# 配合R-Drop的交叉熵损失
ce_rdrop_loss = crossentropy_with_rdrop(target, r_output)
# 配合R-Drop的amsoftmax损失
am_rdrop_loss = K.mean(am_loss) + rdrop_loss
# 配合SimCSE的amsoftmax损失
am_simcse_loss = K.mean(am_loss) + sim_loss
# All Three Loss 加权和
am_simcse_rdrop_loss = K.mean(am_loss) + sim_loss + rdrop_loss


# 自定义 metrics
def sparse_categorical_accuracy(y_true, y_pred):
    # return tf.metrics.SparseCategoricalAccuracy(y_true, y_pred[0])
    return K.metrics.sparse_categorical_accuracy(y_true, y_pred[0])


def train_rdrop():
    # 数据生成器
    train_generator = data_generator(train_token_ids, batch_size)
    # model build
    train_model = Model([x1_in, x2_in, target], [output, r_output])  # 用分类问题做训练

    # 联合训练 amsoftmax+RDrop 
    train_model.add_loss(am_rdrop_loss)
    train_model.add_metric(K.mean(am_loss), name="am_loss")
    train_model.add_metric(rdrop_loss, name="rdrop_loss")
    train_model.compile(
        optimizer=Adam(lr=LR),
        metrics=[sparse_categorical_accuracy],
        )
    custom_callback = CustomCallback(
        # valid_data=valid_data,  # (input, [kid])
        # test_data=test_data,  # (primary, kid)
        batch_size=batch_size,
        encoder=encoder,
    )
    train_model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[custom_callback],
        # shuffle=True,
    )


def train_cse():
    # 数据生成器
    train_generator = data_generator(train_token_ids, batch_size)
    # model build
    train_model = Model([x1_in, x2_in, target], [output, emb]) 
    train_cl_model = Model([x1_in, x2_in, target], emb) 

    # 联合训练 amsoftmax+simcse
    train_model.add_loss(am_simcse_loss)
    train_model.add_metric(K.mean(am_loss), name="am_loss")
    train_model.add_metric(sim_loss, name="sim_loss")
    train_model.compile(
        optimizer=Adam(lr=LR),
        metrics=[sparse_categorical_accuracy],
        )
    custom_callback = CustomCallback(
        # valid_data=valid_data,  # (input, [kid])
        # test_data=test_data,  # (primary, kid)
        batch_size=batch_size,
        encoder=encoder,
    )
    train_model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[custom_callback],
        # shuffle=True,
    )

    # # 数据生成器
    # train_generator = data_generator(train_token_ids, batch_size)

    # 单独 CL，缓解 bert 语义坍塌
    train_cl_model.add_loss(sim_loss)
    train_cl_model.add_metric(sim_loss, name="sim_loss")
    train_cl_model.compile(optimizer=Adam(lr=LR))
    custom_callback = CustomCallback(
        # valid_data=valid_data,  # (input, [kid])
        # test_data=test_data,  # (primary, kid)
        batch_size=batch_size,
        encoder=encoder,
    )
    train_cl_model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[custom_callback],
        # shuffle=True,
    )

    # # 保存权重
    # encoder.save_weights("model/chinese_bert_simcse.h5")
    # # 加载权重测试
    # encoder.load_weights("model/chinese_bert_simcse.h5")


# # 语料向量化
# all_vecs = []
# for a_token_ids, b_token_ids in all_token_ids:
#     a_vecs = encoder.predict([a_token_ids,
#                               np.zeros_like(a_token_ids)],
#                              verbose=True)
#     b_vecs = encoder.predict([b_token_ids,
#                               np.zeros_like(b_token_ids)],
#                              verbose=True)
#     all_vecs.append((a_vecs, b_vecs))

# # 标准化，相似度，相关系数
# all_corrcoefs = []
# for (a_vecs, b_vecs), labels in zip(all_vecs, all_labels):
#     a_vecs = l2_normalize(a_vecs)
#     b_vecs = l2_normalize(b_vecs)
#     sims = (a_vecs * b_vecs).sum(axis=1)
#     corrcoef = compute_corrcoef(labels, sims)
#     all_corrcoefs.append(corrcoef)

# all_corrcoefs.extend([
#     np.average(all_corrcoefs),
#     np.average(all_corrcoefs, weights=all_weights)
# ])

# for name, corrcoef in zip(all_names + ['avg', 'w-avg'], all_corrcoefs):
#     print('%s: %s' % (name, corrcoef))


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
        # try:
        #     assert len(recall) == len(recall_id) == 10
        # except:
        #     print("len(recall) ≠ len(recall_id) 比如query='拔牙'")
        #     print(index,query,recall,recall_id)
        #     continue
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


def compute_pair_input_arrays(input_arrays, maxlen, tokenizer):
    inp1_1, inp1_2 = [], []
    for instance in input_arrays:
        x1, x2 = tokenizer.encode(instance, maxlen=maxlen)
        inp1_1.append(x1)
        inp1_2.append(x2)

    L = [len(x) for x in inp1_1]
    ML = max(L) if L else 0

    pad_func = seq_padding(ML)
    res = [inp1_1, inp1_2]
    res = list(map(pad_func, res))
    return res


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, batch_size=16, encoder=None):
        # self.valid_inputs = valid_data  # (input, [kid])
        # self.valid_outputs = []  # valid_data[-1]
        # self.test_inputs = test_data  # (primary, kid)
        self.batch_size = batch_size
        self.encoder = encoder
        self.f1_avg = 0

    def on_train_begin(self, logs={}):
        self.valid_predictions = []

    def on_epoch_end(self, epoch, logs={}):
        f1_avg = []
        acc_f1_list = []
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
            pred_1_list, val_1_list = [], []
            pred_3_list, val_3_list = [], []
            for query, info_dict in tqdm(test_data_dict.items()):
                n += 1
                kids = info_dict["kid"]
                all_cand_text = info_dict["candidate"]
                all_cand_index = info_dict["cid"]
                all_cand_text_ids = compute_pair_input_arrays(
                    all_cand_text, maxlen, tokenizer=tokenizer
                )
                if all_cand_index:
                    n1 += 1
                test_x = compute_pair_input_arrays([query], maxlen, tokenizer=tokenizer)
                test_query_vec = self.encoder.predict(
                    test_x, batch_size=self.batch_size * 10
                )
                all_cand_vecs = self.encoder.predict(
                    all_cand_text_ids, batch_size=self.batch_size * 10
                )
                
                # 计算相似度
                # dot_list = np.dot(all_cand_vecs, test_query_vec[0])   # l2之后embedding
                dot_list = cosine_similarity(all_cand_vecs, test_query_vec)
                dot_list = [x[0] for x in dot_list]

                # top1预测结果
                max_idx = np.argmax(dot_list)
                pred_one = str(int(all_cand_index[max_idx]))  # 预测的kid
                # f.write("\n".join(list(map(lambda x:str(x), zip([query]*len(all_cand_text), all_cand_text, all_cand_index)))))
                # f.write("\nmax_idx：{}\tpred_id: {}\tkids: {}".format(max_idx, pred_one, kids))
                if pred_one in kids:
                    n2 += 1
                    val_1_list.append(pred_one)
                    # f.write("\n√√√\n\n")
                else:
                    if not set(kids) & set(all_cand_index):
                        nb_kid_not_in_cid += 1
                    val_1_list.append(kids[0])
                    # f.write("\n×××\n\n")
                pred_1_list.append(pred_one)

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
            acc_f1_list.extend([acc, f1])

        f1_avg = sum(f1_avg) / len(f1_avg)
        print("\nf1_avg=sum/len= {}".format(f1_avg))
        if f1_avg > self.f1_avg:
            print("epoch:{} 当前最佳！".format(epoch+1))
            self.f1_avg = f1_avg
            # self.encoder.save_weights(model_path)
        print("\t".join(list(map(lambda x: str(x), acc_f1_list))))
        print("\n")


if __name__ == "__main__":
    train_cse()
