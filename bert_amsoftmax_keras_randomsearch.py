# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import os
import keras
from keras.backend import tensorflow_backend
import keras.backend as K
import tensorflow as tf

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

from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from bert4keras.backend import search_layer
from bert4keras.models import build_transformer_model as build_bert_model
from bert4keras.tokenizers import Tokenizer


"""nohup python bert_amsoftmax_keras_randomsearch.py > log_am.txt 2>&1 &"""

# from util import read_tsv_data
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# set GPU memory
# 方法1:显存占用会随着epoch的增长而增长,也就是后面的epoch会去申请新的显存,前面已完成的并不会释放,为了防止碎片化
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 按需求增长
sess = tf.Session(config=config)
K.set_session(sess)

# 清空之前model占用的内存
K.clear_session()
tf.reset_default_graph()
session_global = tf.Session()  # 创建一个会话，当上下文管理器退出时会话关闭和资源释放自动完成
# session_global = tf.Session().as_default()  # 创建一个默认会话，当上下文管理器退出时会话并没有关闭
tensorflow_backend.set_session(session_global)
graph = tf.get_default_graph()  # 默认的数据流图DAG


# %%
def clean(x):
    x = re.sub('"', "", x)
    x = re.sub("\s", "", x)  # \s匹配任何空白字符，包括空格、制表符、换页符等
    x = re.sub(",", "，", x)
    return x


# %%
import pandas as pd
import os
import csv
import re

with open("DatasetLXH/fewshot_train.csv", "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["knowledge_id", "question", "base_code"])
    for filename in [
        r"DatasetLXH/IT生产知识.xlsx",
        r"DatasetLXH/生产法务数据.xlsx",
        r"DatasetLXH/生产人事数据.xlsx",
    ]:
        df = pd.read_excel(
            filename, usecols=[0, 1, 2, 4], encoding="utf-8", keep_default_na=False
        )
        for index, row in df.iterrows():
            lines = []
            kid, pri, sims, base_code = row
            lines.append([kid, clean(pri), base_code])
            for sim in sims.strip().strip("###").split("###"):
                if sim.strip():
                    lines.append([kid, clean(sim), base_code])
            writer.writerows(lines)

with open("DatasetLXH/fewshot_train.csv", encoding="utf-8") as csvfile:
    for line in csvfile:
        if not len(line.split(",")) == 3 or "" in line.split(","):
            print(line)


# %%
sheet2base = {
    "HALO测试内容": "HALOBIGROOMBASE",
    "差旅测试内容": "CLXTBASE",
    "HR-π": "HRPBASE",
    "成本管理平台": "CBGLXTBASE",
    "场景化费用": "CJHFYXTBASE",
    "供应商": "GYSGXGLPTBASE",
    "商业资产": "C2ZGXTBASE",
}
with open("DatasetLXH/fewshot_test.csv", "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["knowledge_id", "question", "base_code"])
    for filename in [r"DatasetLXH/HALO_差旅_HR-_成本管理测试集123123.xlsx"]:
        for sheet_name, base_code in sheet2base.items():
            df = pd.read_excel(
                filename,
                sheet_name=sheet_name,
                usecols=[0, 3],
                encoding="utf-8",
                keep_default_na=False,
            )
            for index, row in df.iterrows():
                lines = []
                query, kid = row
                if kid:
                    lines.append([clean(str(kid)), clean(query), base_code])
                    writer.writerows(lines)

with open("DatasetLXH/fewshot_test.csv", encoding="utf-8") as csvfile:
    for line in csvfile:
        if not len(line.split(",")) == 3 or "" in line.split(","):
            print(line)


# %%
import pandas as pd
import numpy as np

test = pd.read_csv("DatasetLXH/fewshot_test.csv", engine="python")
train = pd.read_csv("DatasetLXH/fewshot_train.csv", engine="python")  # , nrows=16000
# for index,row in data.iterrows():
# train = train[["knowledge_id", "question", "base_code"]]
# test = test[["knowledge_id", "question", "base_code"]]


# %%


def sparse_categorical_crossentropy(y_true, y_pred):
    """自定义稀疏交叉熵
    这主要是因为keras自带的sparse_categorical_crossentropy不支持求二阶梯度。
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, "int32")
    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
    return K.categorical_crossentropy(y_true, y_pred)


def loss_with_gradient_penalty(y_true, y_pred, epsilon=1):
    """带梯度惩罚的loss"""
    loss = K.mean(sparse_categorical_crossentropy(y_true, y_pred))
    embeddings = search_layer(y_pred, "Embedding-Token").embeddings
    gp = K.sum(K.gradients(loss, [embeddings])[0].values ** 2)
    return loss + 0.5 * epsilon * gp


# 稀疏版AM-Softmax
def sparse_amsoftmax_loss(y_true, y_pred, scale=30, margin=0.45):
    y_true = K.expand_dims(y_true[:, 0], 1)  # 保证y_true的shape=(None, 1)
    y_true = K.cast(y_true, "int32")  # 保证y_true的dtype=int32
    batch_idxs = K.arange(0, K.shape(y_true)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, y_true], 1)
    y_true_pred = tf.gather_nd(y_pred, idxs)  # 目标特征，用tf.gather_nd提取出来
    y_true_pred = K.expand_dims(y_true_pred, 1)
    y_true_pred_margin = y_true_pred - margin  # 减去margin
    _Z = K.concatenate([y_pred, y_true_pred_margin], 1)  # 为计算配分函数
    _Z = _Z * scale  # 缩放结果，主要因为pred是cos值，范围[-1, 1]
    logZ = K.logsumexp(_Z, 1, keepdims=True)  # 用logsumexp，保证梯度不消失
    logZ = logZ + K.log(
        1 - K.exp(scale * y_true_pred - logZ)
    )  # 从Z中减去exp(scale * y_true_pred)
    return -y_true_pred_margin * scale + logZ


# %%
from sklearn.metrics import classification_report, f1_score

unique_test_base = test["base_code"].unique().tolist()
ff = open("result_amsoftmax_optuna.txt", "a+")


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, batch_size=16, encoder=None):
        # self.valid_inputs = valid_data  # (input, [kid])
        # self.valid_outputs = []  # valid_data[-1]
        # self.test_inputs = test_data  # (primary, kid)
        self.batch_size = batch_size
        self.encoder = encoder
        self.f1 = 0

    def on_train_begin(self, logs={}):
        self.valid_predictions = []

    def on_epoch_end(self, epoch, logs={}):

        all_pred_1_list, all_val_1_list = [], []
        for base in tqdm(unique_test_base):
            test_data = test[test["base_code"] == base]  # test
            querys = test_data["question"].tolist()
            labels_test = test_data["knowledge_id"].astype(str).tolist()
            valid_inputs = compute_input_arrays(querys)
            self.valid_predictions.append(
                self.encoder.predict(valid_inputs, batch_size=self.batch_size)[1]
            )
            assert len(self.valid_predictions[-1]) == len(labels_test)

            candidate_data = train[train["base_code"] == base]  # candidate
            querys_cand = candidate_data["question"].tolist()
            labels_cand = candidate_data["knowledge_id"].astype(str).tolist()
            cand_inputs = compute_input_arrays(querys_cand)
            cand_vecs = self.encoder.predict(cand_inputs, batch_size=self.batch_size)[
                1
            ]  # primary embeddings
            assert len(cand_vecs) == len(labels_cand)

            pred_1_list, val_1_list = [], []
            for idx, one in enumerate(self.valid_predictions[-1]):
                # 正确kid列表
                one_anwser_list = labels_test[idx].split("###")
                one_anwser_list = [str(int(one_a)) for one_a in one_anwser_list]
                # 预测的kid
                dot_list = np.dot(cand_vecs, one)  # 点积
                max_idx = np.argmax(dot_list)
                pred_one = str(int(labels_cand[max_idx]))
                # top1预测结果
                if one_anwser_list[0] not in labels_cand:
                    print(one_anwser_list, " not in candidate set")
                    continue
                if pred_one in one_anwser_list:
                    val_1_list.append(pred_one)
                else:
                    try:
                        val_1_list.append(one_anwser_list[0])
                        print(
                            "wrong {} pid:{} kid:{}".format(
                                querys[idx],
                                pred_one,
                                one_anwser_list,
                            )
                        )
                    except:
                        print("error: ", idx)
                pred_1_list.append(pred_one)

            print("base: ", base)
            # top1 report 版本0.21.x以上
            report = classification_report(
                val_1_list, pred_1_list, digits=4, output_dict=True
            )
            print("Top1 micro avg", report["accuracy"])
            print("Top1 macro avg", report["macro avg"])
            print("Top1 weighted avg", report["weighted avg"])
            print("\n")

            all_pred_1_list.extend(pred_1_list)
            all_val_1_list.extend(val_1_list)

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
            self.encoder.save_weights("it_nlu-best.h5")
        print("\n")

        ff.write("ALL" + "\n")
        # top1 report 版本0.21.x以上
        report = classification_report(
            all_val_1_list, all_pred_1_list, digits=4, output_dict=True
        )
        ff.write("Top1 micro avg" + str(report["accuracy"]) + "\n")
        ff.write("Top1 macro avg" + str(report["macro avg"]) + "\n")
        ff.write("Top1 weighted avg" + str(report["weighted avg"]) + "\n")
        f1 = f1_score(all_val_1_list, all_pred_1_list, average="weighted")
        if f1 > self.f1:
            ff.write("epoch:{} 当前最佳f1-score！".format(epoch) + "\n")
            self.f1 = f1
            self.encoder.save_weights("it_nlu-best.h5")
        ff.write("\n" + "\n")


# %%
from bert4keras.backend import search_layer
from bert4keras.models import build_transformer_model as build_bert_model
from bert4keras.tokenizers import Tokenizer
import optuna

# root = r"D:/#Pre-trained_Language_Model/weights/bert/chinese_L-12_H-768_A-12"
root = r"/data/voiceprint2/corpus/chinese_L-12_H-768_A-12"
tokenizer = Tokenizer(os.path.join(root, "vocab.txt"))  # 建立分词器
bert_model = build_bert_model(
    os.path.join(root, "bert_config.json"),
    os.path.join(root, "bert_model.ckpt"),
    model="bert",
)  # embed


def bert_model_amsoftmax(trial):

    # for l in orig_bert_model.layers:
    #    l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    r_out = Input(shape=(None,), dtype="int32")

    x = bert_model([x1_in, x2_in])

    first_token = Lambda(lambda x: x[:, 0])(x)
    embedding = first_token
    first_token = Dropout(
        trial.suggest_discrete_uniform("dropout_rate", 0.1, 0.9, 0.1), name="dp1"
    )(first_token)
    first_token = Lambda(lambda x: K.l2_normalize(x, 1))(first_token)  # l2正则
    first_out = Dense(
        cls_num,
        name="dense_output",
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
    )(first_token)

    p_out = first_out
    model = Model([x1_in, x2_in], [p_out, embedding])

    train_model = Model([x1_in, x2_in, r_out], [p_out, embedding])

    scale = trial.suggest_discrete_uniform("scale", 5, 40, 5)
    margin = trial.suggest_categorical("margin", [0.05, 0.15, 0.25, 0.35, 0.45, 0.5])
    final_loss = sparse_amsoftmax_loss(r_out, p_out, scale, margin)

    train_model.add_loss(final_loss)
    train_model.compile(
        optimizer=Adam(
            trial.suggest_categorical("learning_rate", [1e-4, 3e-4, 1e-3, 3e-3, 2e-5])
        ),
    )  # 用足够小的学习率
    train_model.summary()
    # return train_model, model

    custom_callback = CustomCallback(
        batch_size=batch_size,
        encoder=model,
    )
    train_model.fit(
        x=[train_x[0], train_x[1], train_y],
        y=None,
        epochs=n_epochs,
        batch_size=batch_size,
        callbacks=[custom_callback],
    )

    return custom_callback.f1


# %%
import numpy as np
from tqdm import tqdm

maxlen = 512
use_postag = False

unique_train_label = np.array(train["knowledge_id"].unique().tolist())
labels_train = np.array(train["knowledge_id"].tolist())
map_train_label_indices = {
    label: np.flatnonzero(labels_train == label) for label in unique_train_label
}  # 非零元素的索引
# print("1: ", len(unique_train_label))   # 4234
# print("2: ", map_train_label_indices[33389])    # rows


# %%
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L) if L else 0
    return np.array(
        [
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x
            for x in X
        ]
    )


def _trim_input(title, t_max_len=30, q_max_len=0, a_max_len=0):
    if use_postag:
        # masked_sent_without_verb, masked_sent_only_verb = mask_sent_by_verb(title)
        masked_sent_without_verb = title
        masked_sent_only_verb = title
        x1, x2 = tokenizer.encode(masked_sent_without_verb, masked_sent_only_verb)
    else:
        x1, x2 = tokenizer.encode(title, maxlen=t_max_len)
        # x1, x2 = tokenizer.encode(title,title)
    if len(x1) > maxlen:
        if use_postag:
            t0_x1, t0_x2 = tokenizer.encode(
                masked_sent_without_verb, masked_sent_only_verb
            )
        else:
            t0_x1, t0_x2 = tokenizer.encode(
                title,
            )
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


def compute_input_arrays(input_arrays):
    x1_, x2_ = [], []
    for instance in input_arrays:
        x1, x2 = _trim_input(instance[:maxlen], t_max_len=maxlen)
        x1_.append(x1)
        x2_.append(x2)
    x1_ = seq_padding(x1_)
    x2_ = seq_padding(x2_)
    return [x1_, x2_]


# %%
from keras.optimizers import Adam

n_epochs = 15
batch_size = 64
dropout_rate = 0.5
learning_rate = 2e-5  # [1e-4, 3e-4, 1e-3, 3e-3]
maxlen = 512
steps_per_epoch = 100
scale, margin = 30, 0.15  # amsoftmax参数 30, 0.35

querys_cand = train["question"].tolist()
train_x = compute_input_arrays(querys_cand)

label2id = {}
tmp = train["knowledge_id"].unique().tolist()
for i, label in enumerate(tmp):
    label2id[label] = i + 1
train_y = np.array(list(map(lambda x: label2id[x], train["knowledge_id"].tolist())))
cls_num = len(tmp)


# with session_global.as_default():
#     with session_global.graph.as_default():  # 用with新建一个graph，这样在运行完以及异常退出时就会释放内存
#         train_model, model = bert_model_amsoftmax()
#         custom_callback = CustomCallback(batch_size=batch_size, encoder=model,)
#         train_model.fit(x=[train_x[0], train_x[1], train_y], y=None, epochs=n_epochs, batch_size=batch_size)


# 创建一个学习实例，因为objective返回的评价指标是ndcg，因此目标是最大化
study = optuna.create_study(direction="maximize")
# n_trials代表多少种参数组合，n_jobs是并行搜索的个数，-1代表使用所有的cpu核心
N_TRIALS = 20  # 随机搜索的次数trail
study.optimize(
    bert_model_amsoftmax, n_trials=N_TRIALS, n_jobs=1, gc_after_trial=True
)  # , callbacks=[monitor]
# opt_utils.log_study(study)

print("最优超参: ", study.best_params)
print("最优超参下，objective函数返回的值: ", study.best_value)
print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

ff.write("最优超参: " + str(study.best_params) + "\n")
ff.write("最优超参下，objective函数返回的值: " + str(study.best_value) + "\n")
ff.write("Number of finished trials: {}".format(len(study.trials)) + "\n")
ff.write("Best trial:" + "\n")
trial = study.best_trial
ff.write("  Value: {}".format(trial.value) + "\n")
ff.write("  Params: " + "\n")
for key, value in trial.params.items():
    ff.write("    {}: {}".format(key, value) + "\n")


ff.close()
exit()


# %%
with session_global.as_default():
    with session_global.graph.as_default():  # 用with新建一个graph，这样在运行完以及异常退出时就会释放内存
        # X_train = model.predict(np.array(train['question'].values.tolist()))
        # X_test = model.predict(np.array(test['question'].values.tolist()))

        # y_train = np.array(train['knowledge_id'].values.tolist())
        # y_test = np.array(test['knowledge_id'].values.tolist())

        querys = test["question"].tolist()
        valid_inputs = compute_input_arrays(querys)
        X_test = model.predict(valid_inputs, batch_size=batch_size)
        y_test = np.array(test["knowledge_id"].tolist())

        querys_cand = train["question"].tolist()
        cand_inputs = compute_input_arrays(querys_cand)
        X_train = model.predict(
            cand_inputs, batch_size=batch_size
        )  # primary embeddings
        y_train = np.array(train["knowledge_id"].astype(str).tolist())


# %%
# from sklearn.preprocessing import LabelEncoder
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import LinearSVC

# knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
# svc = LinearSVC()

# knn.fit(X_train, y_train)
# svc.fit(X_train, y_train)

# y_pred_knn = knn.predict(X_test)
# acc_knn = accuracy_score(y_test, y_pred_knn)
# y_pred_svc = svc.predict(X_test)
# acc_svc = accuracy_score(y_test, y_pred_svc)

# print(f'KNN accuracy = {acc_knn}, SVM accuracy = {acc_svc}')


# %%
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X_embedded = TSNE(n_components=2).fit_transform(X_test)

plt.figure(figsize=(10, 10))

for i, t in enumerate(set(y_test)):
    idx = y_test == t
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

plt.legend(bbox_to_anchor=(1, 1))


# %%
from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2).fit_transform(X_train)

plt.figure(figsize=(10, 10))

for i, t in enumerate(set(y_train)):
    idx = y_train == t
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

plt.legend(bbox_to_anchor=(1, 1))
