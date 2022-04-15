# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import os
import keras
from keras.backend import tensorflow_backend
import keras.backend as K
import tensorflow as tf

"""nohup python triplet_offline_keras.py > log_triplet_offline.txt 2>&1 &"""

# from util import read_tsv_data
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
            filename, usecols=[0, 1, 2, 4], keep_default_na=False
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

test = pd.read_csv("DatasetLXH/fewshot_test.csv", engine='python') 
train = pd.read_csv("DatasetLXH/fewshot_train.csv", engine='python')  # , nrows =10240
# for index,row in data.iterrows():
# train = train[["knowledge_id", "question", "base_code"]]
# test = test[["knowledge_id", "question", "base_code"]]


# %%
from bert4keras.backend import search_layer
from bert4keras.models import build_transformer_model as build_bert_model
from bert4keras.tokenizers import Tokenizer

root = r"/data/ningshixian/work/corpus/chinese_L-12_H-768_A-12"
tokenizer = Tokenizer(os.path.join(root, "vocab.txt"))  # 建立分词器
bert_model = build_bert_model(
    os.path.join(root, "bert_config.json"),
    os.path.join(root, "bert_model.ckpt"),
    model="bert",
)  # embed


# %%
import os
import keras
from keras.models import Model
from keras.layers import *

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))
x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)  # first_token
x = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)  # L2正则化

# x = Dropout(0.4)(x)
# x = Dense(128, name="dense_layer")(x)
# x = Dense(256, activation="relu")(x)
# x = Dropout(0.4)(x)
# x = Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
# x = Dropout(0.4)(x)
# dense_layer = Dense(128, name="dense_layer")(x)
# norm_layer = Lambda(lambda x: K.l2_normalize(x, axis=1), name="norm_layer")(dense_layer)

model = Model(inputs=[x1_in, x2_in], outputs=x)
model.summary()


# %%
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Layer

# def triplet_loss(model_anchor, model_positive, model_negative, margin):
#     distance1 = tf.sqrt(
#         tf.reduce_sum(tf.pow(model_anchor - model_positive, 2), 1, keepdims=True)
#     )
#     distance2 = tf.sqrt(
#         tf.reduce_sum(tf.pow(model_anchor - model_negative, 2), 1, keepdims=True)
#     )
#     return tf.reduce_mean(tf.maximum(distance1 - distance2 + margin, 0)) + 1e-9


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        a, p, n = inputs
        p_dist = K.sqrt(K.sum(K.square(a - p), axis=-1))
        n_dist = K.sqrt(K.sum(K.square(a - n), axis=-1))
        return K.mean(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0) + 1e-9

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss


# %%
from keras import backend as K
from keras.backend import tensorflow_backend
from keras.models import Model
from keras.layers import Input, Layer

# Input for anchor, positive and negative images
x1_in = Input(shape=(None,), name="anchor_input")
x2_in = Input(shape=(None,))

x3_in = Input(shape=(None,), name="positive_input")
x4_in = Input(shape=(None,))

x5_in = Input(shape=(None,), name="negative_input")
x6_in = Input(shape=(None,))

# Output for anchor, positive and negative embedding vectors
# The bert_model instance is shared (Siamese network)
emb_a = model([x1_in, x2_in])
emb_p = model([x3_in, x4_in])
emb_n = model([x5_in, x6_in])

# Layer that computes the triplet loss from anchor, positive and negative embedding vectors
triplet_loss_layer = TripletLossLayer(alpha=0.4, name="triplet_loss_layer")(
    [emb_a, emb_p, emb_n]
)
# Model that can be trained with anchor, positive negative images
train_model = Model([x1_in, x2_in, x3_in, x4_in, x5_in, x6_in], triplet_loss_layer)

# # BYJ
# train_model = Model([x1_in, x2_in, x3_in, x4_in, x5_in, x6_in], x_a)
# final_loss = triplet_loss(x, x_p, x_n, 0.1)
# train_model.add_loss(final_loss)

train_model.summary()


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


def compute_input_arrays(input_arrays):
    x1_, x2_ = [], []
    for instance in (input_arrays):
        x1, x2 = _trim_input(instance[:maxlen], t_max_len=maxlen)
        x1_.append(x1)
        x2_.append(x2)
    x1_ = seq_padding(x1_)
    x2_ = seq_padding(x2_)
    return [x1_, x2_]


# # %%

# def get_triplets(unique_train_label, map_train_label_indices):
#     label_l, label_r = np.random.choice(unique_train_label, 2, replace=False)
#     while len(map_train_label_indices[label_l]) < 2:
#         label_l, label_r = np.random.choice(unique_train_label, 2, replace=False)
#     a, p = np.random.choice(map_train_label_indices[label_l], 2, replace=False)

#     n = np.random.choice(map_train_label_indices[label_r])
#     return a, p, n


# def get_triplets_batch(k, train_set, unique_train_label, map_train_label_indices):

#     while True:
#         idxs_a, idxs_p, idxs_n = [], [], []
#         for _ in range(k):
#             a, p, n = get_triplets(unique_train_label, map_train_label_indices)
#             idxs_a.append(a)
#             idxs_p.append(p)
#             idxs_n.append(n)

#         a = train_set.iloc[idxs_a].values.tolist()
#         b = train_set.iloc[idxs_p].values.tolist()
#         c = train_set.iloc[idxs_n].values.tolist()
#         # print(a, b, c)

#         train_in = []
#         train_in.extend(compute_input_arrays(a))
#         train_in.extend(compute_input_arrays(b))
#         train_in.extend(compute_input_arrays(c))

#         yield train_in, []  # data, label


# # train_data = get_triplets_batch(2,train['question'],unique_train_label,map_train_label_indices)
# # for item in train_data:
# #     exit()


# %%
def get_triplets_batch(k, train_set, unique_train_label, map_train_label_indices):
    
    # all_train = compute_input_arrays(train_set)
    # train_vecs = model.predict(all_train, batch_size=2048, verbose=1)  # primary embeddings
    
    idxs_a, idxs_p, idxs_n = [], [], []
    for label in tqdm(unique_train_label):
        rows = map_train_label_indices[label]
        if len(rows) > 2:
            a = rows[0]
            vecs = model.predict(compute_input_arrays(train_set[rows]), batch_size=128)
            dot_list = np.dot(vecs[1:], vecs[0])  # 点积
            min_idx = np.argmin(dot_list)
            p = rows[1:][min_idx]
            
            label_n = np.random.choice(np.delete(unique_train_label, np.where(unique_train_label == label)))
            rows_n = map_train_label_indices[label_n]
            vecs_n = model.predict(compute_input_arrays(train_set[rows_n]), batch_size=128)
            dot_list_n = np.dot(vecs_n, vecs[0])  # 点积
            max_idx = np.argmax(dot_list_n)   # 大多负例为同一个？
            n = rows_n[max_idx]
            
            idxs_a.append(a)
            idxs_p.append(p)
            idxs_n.append(n)

    a = train_set.iloc[idxs_a].values.tolist()
    b = train_set.iloc[idxs_p].values.tolist()
    c = train_set.iloc[idxs_n].values.tolist()
    # print(a[:20])
    # print(b[:20])
    # print(c[:20])

    train_in = []
    train_in.extend(compute_input_arrays(a))
    train_in.extend(compute_input_arrays(b))
    train_in.extend(compute_input_arrays(c))

    return train_in, []  # data, label


# train_data = get_triplets_batch(2,train['question'],unique_train_label,map_train_label_indices)
# print(train_data)
# exit()


# %%
from sklearn.metrics import classification_report, f1_score
unique_test_base = test["base_code"].unique().tolist()


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
                self.encoder.predict(valid_inputs, batch_size=self.batch_size)
            )
            assert len(self.valid_predictions[-1])==len(labels_test)

            candidate_data = train[train["base_code"] == base]  # candidate
            querys_cand = candidate_data["question"].tolist()
            labels_cand = candidate_data["knowledge_id"].astype(str).tolist()
            cand_inputs = compute_input_arrays(querys_cand)
            cand_vecs = self.encoder.predict(
                cand_inputs, batch_size=self.batch_size
            )  # primary embeddings
            assert len(cand_vecs)==len(labels_cand)

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
                                querys[idx], pred_one, one_anwser_list,
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


# %%
from keras.optimizers import Adam

n_epochs = 30
batch_size = 32
steps_per_epoch = 100
custom_callback = CustomCallback(batch_size=batch_size, encoder=model,)
with session_global.as_default():
    with session_global.graph.as_default():  # 用with新建一个graph，这样在运行完以及异常退出时就会释放内存
        train_model.compile(loss=None, optimizer=Adam(0.0001))
        train_model.fit(
            get_triplets_batch(
                batch_size,
                train["question"],
                unique_train_label,
                map_train_label_indices,
            ),
            epochs=n_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[custom_callback],
        )
        # train_model.fit(
        #     x=get_triplets_batch(
        #         batch_size,
        #         train["question"],
        #         unique_train_label,
        #         map_train_label_indices,
        #     )[0],
        #     y=[],
        #     epochs=n_epochs,
        #     batch_size=batch_size,
        #     callbacks=[custom_callback],
        # )

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
