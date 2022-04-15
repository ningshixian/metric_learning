#! -*- coding:utf-8 -*-
import math
import os
import json
import re
from tqdm import tqdm
import numpy as np
import random
import argparse
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold

import keras
from keras.backend import tensorflow_backend
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from bert4keras.backend import search_layer
from bert4keras.models import build_transformer_model as build_bert_model
from bert4keras.tokenizers import Tokenizer
import tensorflow as tf

import data
from util import get_model_by_name

"""
cuda3: nohup python q_q_encoder_using_cls_training_nsx.py -m bert -e 20 -d 0.5 -f 0 -am 0 > log1.txt 2>&1 &
python q_q_encoder_using_cls_training_nsx.py -m bert -e 1 -d 0.5 -f 0 -am 1 -l on
"""

# from util import read_tsv_data
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# set GPU memory
# 方法1:显存占用会随着epoch的增长而增长,也就是后面的epoch会去申请新的显存,前面已完成的并不会释放,为了防止碎片化
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 按需求增长
sess = tf.Session(config=config)
K.set_session(sess)

# 选择测试集用的
CAT = ["HR-π", "差旅系统", "HALO-BIGROOM", "成本管理系统", "场景化费用系统", "供应商关系管理平台", "C2资管系统知识库"]
cat2sheet = {
    "HALO-BIGROOM": 0,
    "差旅系统": 1,
    "成本管理系统": 3,
    "HR-π": 2,
    "场景化费用系统": 4,
    "供应商关系管理平台": 5,
    "C2资管系统知识库": 6,
}

# pseg = ""
# Threshold = 1  # gradient_penalty
# dropout=0.5
scale, margin = 30, 0.15  # amsoftmax参数 30, 0.35
min_learning_rate = 2e-5
maxlen = 512
batch_size = 16
# line = "off"


def sparse_categorical_crossentropy(y_true, y_pred):
    """自定义稀疏交叉熵
    这主要是因为keras自带的sparse_categorical_crossentropy不支持求二阶梯度。
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, "int32")
    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
    return K.categorical_crossentropy(y_true, y_pred)


def loss_with_gradient_penalty(y_true, y_pred, epsilon=1):
    """带梯度惩罚的loss
    """
    loss = K.mean(sparse_categorical_crossentropy(y_true, y_pred))
    embeddings = search_layer(y_pred, "Embedding-Token").embeddings
    gp = K.sum(K.gradients(loss, [embeddings])[0].values ** 2)
    return loss + 0.5 * epsilon * gp


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L) if L else 0
    return np.array(
        [
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x
            for x in X
        ]
    )


def bert_model(params, model_config):
    orig_bert_model = build_bert_model(
        model_config.config_path,
        model_config.checkpoint_path,
        model=model_config.model_type,
    )

    # for l in orig_bert_model.layers:
    #    l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    r_out = Input(shape=(None,), dtype="int32")

    x = orig_bert_model([x1_in, x2_in])

    first_token = Lambda(lambda x: x[:, 0])(x)
    embedding = first_token
    first_token = Dropout(params[0], name="dp1")(first_token)
    first_out = Dense(
        cls_num,
        name="dense_output",
        activation="softmax",
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
    )(first_token)

    p_out = first_out
    model = Model([x1_in, x2_in], [p_out, embedding])

    train_model = Model([x1_in, x2_in, r_out], [p_out, embedding])
    final_loss = loss_with_gradient_penalty(r_out, p_out, Threshold)

    train_model.add_loss(final_loss)
    train_model.compile(optimizer=Adam(min_learning_rate),)  # 用足够小的学习率
    train_model.summary()

    return train_model, model


def bert_model_amsoftmax(params, model_config):
    orig_bert_model = build_bert_model(
        model_config.config_path,
        model_config.checkpoint_path,
        model=model_config.model_type,
    )

    # for l in orig_bert_model.layers:
    #    l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    r_out = Input(shape=(None,), dtype="int32")

    x = orig_bert_model([x1_in, x2_in])

    first_token = Lambda(lambda x: x[:, 0])(x)
    embedding = first_token
    first_token = Dropout(params[0], name="dp1")(first_token)
    first_token = Lambda(lambda x: K.l2_normalize(x, 1))(first_token)  # l2正则
    first_out = Dense(
        cls_num,
        name="dense_output",
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
    )(first_token)

    p_out = first_out
    model = Model([x1_in, x2_in], [p_out, embedding])

    train_model = Model([x1_in, x2_in, r_out], [p_out, embedding])

    from margin_softmax import sparse_amsoftmax_loss

    final_loss = sparse_amsoftmax_loss(r_out, p_out, scale, margin)

    train_model.add_loss(final_loss)
    train_model.compile(optimizer=Adam(min_learning_rate),)  # 用足够小的学习率
    train_model.summary()

    return train_model, model


# cur_config = get_model_by_name("bert")
# tokenizer = Tokenizer(cur_config.spm_path)  # 建立分词器
# use_postag = False


def predict_emb(sent_array, nn_model):
    X1 = []
    X2 = []
    for sent in sent_array:
        text = sent[:maxlen]
        x1, x2 = tokenizer.encode(text)
        X1.append(x1)
        X2.append(x2)
    # X1 = [x1]
    # X2 = [x2]
    X1 = seq_padding(X1)
    X2 = seq_padding(X2)
    _, emb = nn_model.predict([X1, X2], batch_size=200)
    # print(np.array(emb).shape)
    return emb


def predict_vec(d, nn_model):
    text = d
    if use_postag:
        # s_nv, s_v = mask_sent_by_verb(text)
        x1, x2 = tokenizer.encode(text, text)
    else:
        x1, x2 = tokenizer.encode(text)
    X1 = [x1]
    X2 = [x2]
    X1 = seq_padding(X1)
    X2 = seq_padding(X2)
    R = nn_model.predict([X1, X2])[0]
    return R


def predict_one_sent(d, model):
    text = d
    lab_array = predict_vec(text, model)
    # lab_array = [str(ele) for ele in lab_array]
    return lab_array


# def comput_f1_by_label(trues, preds):
#     # print(trues.shape, preds.shape)
#     for index, (col_trues, col_pred) in enumerate(zip(trues.T, preds.T)):
#         # print(col_trues.shape, col_pred.shape)
#         col_pred = [1 if one_pred >= 0.5 else 0 for one_pred in col_pred]
#         print(index, classification_report(col_trues, col_pred))


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, valid_data, test_data, batch_size=16, encoder=None):
        self.valid_inputs = valid_data  # (input, [kid])
        self.valid_outputs = []  # valid_data[-1]
        self.test_inputs = test_data  # (primary, kid)

        self.batch_size = batch_size
        self.f1 = 0
        self.encoder = encoder

    def on_train_begin(self, logs={}):
        self.valid_predictions = []
        self.test_predictions = []

    def on_epoch_begin(self, epoch, logs={}):
        all_pred_1_list, all_val_1_list = [], []
        all_pred_3_list, all_val_3_list = [], []
        all_pred_5_list, all_val_5_list = [], []
        print(self.valid_inputs.keys()-self.test_inputs.keys())
        for cat, valid_input in self.valid_inputs.items():
            if cat=="ITBASE":
                continue
            querys = [one[0] for one in valid_input]
            valid_inputs, _ = compute_input_arrays(querys, querys)
            self.valid_outputs = [one[1] for one in valid_input]
            self.valid_predictions.append(
                self.encoder.predict(valid_inputs, batch_size=self.batch_size)
            )
            print("on epoch {} end".format(epoch))
            print(len(self.valid_predictions[-1]))  # 2
            print(len(self.valid_outputs))
            all_cand_text = [one[0] for one in self.test_inputs[cat]]
            all_cand_index = [str(int(one[1])) for one in self.test_inputs[cat]]
            all_cand_ids, _ = compute_input_arrays(all_cand_text, all_cand_text)
            all_cand_vecs = self.encoder.predict(
                all_cand_ids, batch_size=self.batch_size
            )[
                1
            ]  # primary embeddings

            pred_1_list, val_1_list = [], []
            pred_3_list, val_3_list = [], []
            pred_5_list, val_5_list = [], []
            for idx, one in enumerate(
                self.valid_predictions[-1][1]
            ):  # one:input embedding
                # 正确kid列表
                one_anwser_list = self.valid_outputs[idx]
                one_anwser_list = [str(int(one_a)) for one_a in one_anwser_list]
                # print(one.shape, all_cand_vecs.shape)
                dot_list = np.dot(all_cand_vecs, one)

                # top1预测结果
                max_idx = np.argmax(dot_list)
                pred_one = str(int(all_cand_index[max_idx]))  # 预测的kid
                if one_anwser_list[0] not in all_cand_index:
                    print(one_anwser_list, " not in candidate set")
                    continue
                if pred_one in one_anwser_list:
                    val_1_list.append(pred_one)
                else:
                    val_1_list.append(one_anwser_list[0])
                    print(
                        "wrong {} pid:{} kid:{}".format(
                            querys[idx], pred_one, one_anwser_list,
                        )
                    )
                pred_1_list.append(pred_one)

                # top3
                max_idx = np.argpartition(dot_list, -3)[-3:]
                pred_three = list(
                    map(lambda x: str(int(x)), [all_cand_index[idx] for idx in max_idx])
                )
                gold_one = list(set(pred_three) & set(one_anwser_list))
                if gold_one:
                    pred_3_list.append(gold_one[0])
                    val_3_list.append(gold_one[0])
                else:
                    pred_3_list.append(pred_three[0])
                    val_3_list.append(one_anwser_list[0])
                # top5
                max_idx = np.argpartition(dot_list, -5)[-5:]
                pred_five = list(
                    map(lambda x: str(int(x)), [all_cand_index[idx] for idx in max_idx])
                )
                gold_one = list(set(pred_five) & set(one_anwser_list))
                if gold_one:
                    pred_5_list.append(gold_one[0])
                    val_5_list.append(gold_one[0])
                else:
                    pred_5_list.append(pred_five[0])
                    val_5_list.append(one_anwser_list[0])

            print("cat: ", cat)
            # top1 report 版本0.21.x以上
            report = classification_report(
                val_1_list, pred_1_list, digits=4, output_dict=True
            )
            print("Top1 micro avg", report["accuracy"])
            print("Top1 macro avg", report["macro avg"])
            print("Top1 weighted avg", report["weighted avg"])
            # top3 report
            report3 = classification_report(
                val_3_list, pred_3_list, digits=4, output_dict=True
            )
            print("Top3 micro avg", report3["accuracy"])
            print("Top3 macro avg", report3["macro avg"])
            print("Top3 weighted avg", report3["weighted avg"])
            # top5 report
            report5 = classification_report(
                val_5_list, pred_5_list, digits=4, output_dict=True
            )
            print("Top5 micro avg", report5["accuracy"])
            print("Top5 macro avg", report5["macro avg"])
            print("Top5 weighted avg", report5["weighted avg"])

            all_pred_1_list.extend(pred_1_list)
            all_val_1_list.extend(val_1_list)
            all_pred_3_list.extend(pred_3_list)
            all_val_3_list.extend(val_3_list)
            all_pred_5_list.extend(pred_5_list)
            all_val_5_list.extend(val_5_list)

        print("ALL")
        # top1 report 版本0.21.x以上
        report = classification_report(
            all_val_1_list, all_pred_1_list, digits=4, output_dict=True
        )
        print("Top1 micro avg", report["accuracy"])
        print("Top1 macro avg", report["macro avg"])
        print("Top1 weighted avg", report["weighted avg"])
        f1 = f1_score(all_val_1_list, all_pred_1_list, average="weighted")
        if f1 > self.f1:
            print("epoch:{} 当前最佳f1-score！".format(epoch))
            self.f1 = f1
            self.model.save_weights("it_nlu-{name}.h5".format(name=epoch))
        # top3 report
        report3 = classification_report(
            all_val_3_list, all_pred_3_list, digits=4, output_dict=True
        )
        print("Top3 micro avg", report3["accuracy"])
        print("Top3 macro avg", report3["macro avg"])
        print("Top3 weighted avg", report3["weighted avg"])
        # top5 report
        report5 = classification_report(
            all_val_5_list, all_pred_5_list, digits=4, output_dict=True
        )
        print("Top5 micro avg", report5["accuracy"])
        print("Top5 macro avg", report5["macro avg"])
        print("Top5 weighted avg", report5["weighted avg"])


def slice(x, index):
    return x[:, :, index]


def train_and_predict(
    model, train_data, valid_data, test_data, epochs, batch_size
    ):
    custom_callback = CustomCallback(
        valid_data=valid_data,  # (input, [kid])
        test_data=test_data,  # (primary, kid)
        batch_size=batch_size,
        encoder=model[1],
    )

    if line=="off":
        model[0].fit(
            train_data, epochs=epochs, batch_size=batch_size, callbacks=[custom_callback],
        )
    elif line=="on":
        model[0].fit(
            train_data, epochs=epochs, batch_size=batch_size, callbacks=[custom_callback],
        )
        model[1].save_weights("it_nlu-best-online.h5")

    return custom_callback


# def data_processor():
#     pass


def compute_output_arrays(df, columns):
    return np.asarray(df).reshape((-1, 1))


# def mask_sent_by_verb(long_text):
#     line = long_text.strip()
#     words = pseg.cut(line)
#     none_verb_mask = "XXX"
#     verb_mask = "VVV"
#     without_verb = list()
#     only_verb = list()
#     for word, flag in words:
#         if flag[0].lower() == "v":
#             only_verb.append(word)
#             without_verb.append(verb_mask)
#         else:
#             only_verb.append(none_verb_mask)
#             without_verb.append(word)
#     with_v_sent = "".join(only_verb)
#     without_v_sent = "".join(without_verb)
#     return without_v_sent, with_v_sent


def _trim_input(
    title,
    question,
    answer,
    max_sequence_length,
    t_max_len=30,
    q_max_len=239,
    a_max_len=239,
    ):
    if use_postag:
        # masked_sent_without_verb, masked_sent_only_verb = mask_sent_by_verb(title)
        masked_sent_without_verb = title
        masked_sent_only_verb = title
        x1, x2 = tokenizer.encode(masked_sent_without_verb, masked_sent_only_verb)
    else:
        x1, x2 = tokenizer.encode(title,)
    if len(x1) > maxlen:
        if use_postag:
            t0_x1, t0_x2 = tokenizer.encode(
                masked_sent_without_verb, masked_sent_only_verb
            )
        else:
            t0_x1, t0_x2 = tokenizer.encode(title,)
        q0_x1, q0_x2 = tokenizer.encode("")
        a0_x1, a0_x2 = tokenizer.encode("")

        head_tag_x1, head_tag_x2 = [t0_x1[0]], [t0_x2[0]]
        tail_tag_x1, tail_tag_x2 = [t0_x1[-1]], [t0_x2[-1]]

        t_x1, t_x2 = t0_x1[1:-1], t0_x2[1:-1]
        q_x1, q_x2 = q0_x1[1:-1], q0_x2[1:-1]
        a_x1, a_x2 = a0_x1[1:-1], a0_x2[1:-1]

        t_len = len(t_x1)
        q_len = len(q_x1)
        a_len = len(a_x1)

        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + math.floor((t_max_len - t_len) / 2)
            q_max_len = q_max_len + math.ceil((t_max_len - t_len) / 2)
        else:
            t_new_len = t_max_len

        if a_max_len > a_len:
            a_new_len = a_len
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len

        x1 = (
            head_tag_x1
            + t_x1[:t_new_len]
            + q_x1[:q_new_len]
            + a_x1[:a_new_len]
            + tail_tag_x1
        )
        x2 = (
            head_tag_x2
            + t_x2[:t_new_len]
            + q_x2[:q_new_len]
            + a_x2[:a_new_len]
            + tail_tag_x2
        )

    if len(x1) > maxlen:
        print("x1!!!!", len(x1))
    if len(x2) > maxlen:
        print("x2!!!!", len(x2))
    return x1, x2


def compute_input_arrays(df, columns):
    x1_, x2_ = [], []
    for instance in tqdm(df):
        t, q, a, cat = instance, instance, instance, instance
        x1, x2 = _trim_input(t, q, a, maxlen)
        x1_.append(x1)
        x2_.append(x2)
    x1_ = seq_padding(x1_)
    x2_ = seq_padding(x2_)
    sent_str_dict = {
        str(index_list.tolist()): df[index] for index, index_list in enumerate(x1_)
    }
    return [x1_, x2_], sent_str_dict


# def get_combos(tuple_list, n):
#     res_list = list()
#     while len(res_list) < n:
#         one_tuple = list()
#         for tup in tuple_list:
#             one_ele = random.sample(tup, 1)[0]
#             one_tuple.append(one_ele)
#         res_list.append(one_tuple)
#     return res_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser("传入参数：***.py")
    parser.add_argument("-m", "--model", default="bert")
    parser.add_argument("-e", "--epoch", default="1")
    parser.add_argument("-d", "--dropout", default="0.05")
    parser.add_argument("-f", "--flooding_thres", default="0.0")
    parser.add_argument("-p", "--use_postag", default="False")
    parser.add_argument("-r", "--random", default="0")
    parser.add_argument("-am", "--amsoft", default="1")
    parser.add_argument("-l", "--line", default="off")  # online offline

    args = parser.parse_args()
    name_ = args.model
    cur_config = get_model_by_name(name_)  # 获取bert相关配置
    epoch_num = int(args.epoch)
    dropout = float(args.dropout)
    Threshold = float(args.flooding_thres)
    use_postag = bool(args.use_postag)
    amsoft = int(args.amsoft)
    line = str(args.line).strip('\n').strip()

    # 读入数据
    if line=="off":
        # [primary], [kid的索引], {cat:[(用户输入,kid)]}, {cat:[(primary, kid)]}, 分类数
        # df_train, output_categories, valid_data, all_test_anwsers, cls_num = data.preprocessing(CAT, cat2sheet)
        df_train, output_categories, all_test_anwsers, cls_num = data.get_train_data(CAT)
        valid_data = data.get_test_data(CAT, cat2sheet)
    elif line=="on":
        all_test_anwsers = {}
        with open("it_data.txt", mode="r", encoding="utf-8") as fp:
            for item in fp:
                if not len(item.split("\t"))==4:
                    print(item)
                sim, kid, primary, base_code = item.strip("\n").strip().split('\t')
                all_test_anwsers.setdefault(base_code, [])
                all_test_anwsers[base_code].append((primary, kid))
        print(all_test_anwsers.keys())
        df_train, output_categories, cls_num = data.get_train_data_online()
        # valid_data = data.get_test_data(CAT, cat2sheet)
        valid_data = data.get_IT_test_data()

    # tr_len = -1
    # df_train = df_train[:tr_len]
    # output_categories = output_categories[:tr_len]

    # shuffle the data
    x_y_pairs = [
        (one_data, output_categories[i]) for i, one_data in enumerate(df_train)
    ]  # (primary, kid)
    r = random.random
    random.seed(int(args.random))
    random.shuffle(x_y_pairs, random=r)
    df_train = [one[0] for one in x_y_pairs]
    output_categories = [one[1] for one in x_y_pairs]

    tokenizer = Tokenizer(cur_config.spm_path)  # 建立分词器
    outputs = np.array(output_categories)
    inputs, tr_sent_dict = compute_input_arrays(
        df_train, []
    )  # [indices, segments], [句子索引格式映射回字符串格式]

    # ================== 模型训练 ==================== #
    session_global = None
    comb = [dropout, 1, "sigmoid", 9]  # params
    print("test combination ", comb)
    K.clear_session()
    session_global = tf.Session()
    tensorflow_backend.set_session(session_global)
    graph = tf.get_default_graph()
    if amsoft == 0:
        train_model, model = bert_model(comb, cur_config)
    else:
        train_model, model = bert_model_amsoftmax(comb, cur_config)

    train_idx = list(range(len(df_train)))
    list_train_idx = list(train_idx)
    batch_c1 = len(df_train) % batch_size
    if batch_c1 != 0:   # 取整
        train_idx = train_idx[:-batch_c1]

    train_in = []
    train_in.extend(inputs)
    train_in.append(outputs)
    assert len(train_in) == 3
    # x1_in、x2_in、r_out 各取一个batch的数据
    train_inputs = [train_in[i][train_idx] for i in range(len(train_in))]
    print(len(train_inputs[0]), len(train_inputs[1]), len(train_inputs[2]))

    model.load_weights("it_nlu-best-online.h5")  # 直接测试
    history = train_and_predict(
        (train_model, model),
        train_data=train_inputs,
        valid_data=valid_data,  # (input, [kid])
        test_data=all_test_anwsers,  # (primary, kid)
        epochs=epoch_num,
        batch_size=batch_size,
    )
    exit()

