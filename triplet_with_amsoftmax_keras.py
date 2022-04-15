# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import os
import keras
from keras.backend import tensorflow_backend
from keras.layers import Input, Layer
import keras.backend as K
import tensorflow as tf
from tqdm import tqdm
import numpy as np
# tf.enable_eager_execution() 

"""nohup python triplet_with_amsoftmax_keras.py > log_tri_am.txt 2>&1 &"""


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


learning_rate = 2e-5  # [1e-4, 3e-4, 1e-3, 3e-3]
batch_size = 32
n_epochs = 10

use_batch_norm = False
bn_momentum = 0.9
margin = 0.5  # 10
shuffle = False  # 是否每次迭代结束打乱数据
triplet_strategy = "batch_hard"  # "batch_hard"  # "batch_all" "batch_semihard"
squared = False
filtered = False  # 是否过滤无相似问的知识
dropout_rate = 0.5

maxlen = 512
use_postag = False
num_train_per_class = 15  # num_sample
scale, am_margin = 30, 0.15  # amsoftmax参数 30, 0.35


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
        df = pd.read_excel(filename, usecols=[0, 1, 2, 4], keep_default_na=False)
        for index, row in df.iterrows():
            lines = []
            kid, pri, sims, base_code = row
            lines.append([kid, clean(pri), base_code])
            for sim in sims.strip().strip("###").split("###"):
                if sim:
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

test = pd.read_csv("DatasetLXH/fewshot_test.csv", engine="python")
train = pd.read_csv("DatasetLXH/fewshot_train.csv", engine="python")  # , nrows =10240
# for index,row in data.iterrows():
# train = train[["knowledge_id", "question", "base_code"]]
# test = test[["knowledge_id", "question", "base_code"]]


# %%
from bert4keras.backend import search_layer
from bert4keras.models import build_transformer_model as build_bert_model
from bert4keras.tokenizers import Tokenizer

# root = r"/data/ningshixian/work/corpus/chinese_L-12_H-768_A-12"
root = r"../corpus/chinese_L-12_H-768_A-12"
tokenizer = Tokenizer(os.path.join(root, "vocab.txt"))  # 建立分词器
bert_model = build_bert_model(
    os.path.join(root, "bert_config.json"),
    os.path.join(root, "bert_model.ckpt"),
    model="bert",
)  # embed


# %%
"""Define functions to create the triplet loss with online triplet mining."""


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = (
        tf.expand_dims(square_norm, 1)
        - 2.0 * dot_product
        + tf.expand_dims(square_norm, 0)
    )

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(
        tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k
    )

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def batch_hard_triplet_loss(labels, embeddings, margin=1, squared=False):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
        1.0 - mask_anchor_negative
    )

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(
        hardest_positive_dist - hardest_negative_dist + margin, 0.0
    )

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """
    计算整个banch的Triplet loss。
    生成所有合格的triplets样本组，并只对其中>0的部分取均值
    Args:
        labels: 标签，shape=(batch_size,)
        embeddings: 形如(batch_size, embed_dim)的张量
        margin: Triplet loss中的间隔
        squared: Boolean. True->欧氏距离的平方，False->欧氏距离
    Returns:
        triplet_loss: 损失
    """
    # 获取banch中嵌入向量间的距离矩阵
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)

    # 计算一个形如(batch_size, batch_size, batch_size)的3D张量
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # 将invalid Triplet置零
    # label(a) != label(p) or label(a) == label(n) or a == p
    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # 删除负值
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # 计算正值
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
    return triplet_loss, fraction_positive_triplets


# %%
def get_loss_function(labels):
    def loss(y_true, y_pred):
        if triplet_strategy == "batch_hard":
            return batch_hard_triplet_loss(
                tf.squeeze(labels), y_pred, margin=margin, squared=squared
            )
        if triplet_strategy == "batch_all":
            return batch_all_triplet_loss(
                tf.squeeze(labels), y_pred, margin=margin, squared=squared
            )[0]
        if triplet_strategy == "batch_semihard":
            return tf.contrib.losses.metric_learning.triplet_semihard_loss(
                labels=tf.squeeze(labels), embeddings=y_pred, margin=margin
            )

    return loss


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
    for instance in tqdm(input_arrays):
        x1, x2 = _trim_input(instance[:maxlen], t_max_len=maxlen)
        x1_.append(x1)
        x2_.append(x2)
    x1_ = seq_padding(x1_)
    x2_ = seq_padding(x2_)
    return [x1_, x2_]


# %%
from pandas import Series, DataFrame


def generate_data(train, sample_per_class=10):
    dataset = train["question"]
    label = np.array(train["knowledge_id"].tolist())
    base = train["base_code"]
    # print(np.array(dataset).shape)  # (30344,)
    # print(np.array(label).shape)  # (30344,)
    data_filter = {"question": [], "knowledge_id": [], "base_code": []}
    x, y = None, None
    for i in tqdm(unique_train_label):
        pos_indices = np.argwhere(label == i)[:, 0]  # numpy.ndarray

        # 过滤无相似问的知识
        if filtered and len(pos_indices) < 2:
            continue

        # print("pos indices: {}, neg_indices: {}".format(pos_indices.shape, neg_indices.shape))
        num_sample = (
            sample_per_class
            if sample_per_class < len(pos_indices)
            else len(pos_indices)
        )
        choice_anchor = np.random.choice(
            pos_indices.shape[0], num_sample, replace=False
        )
        choice_anchor = pos_indices[choice_anchor]

        data_filter["question"].extend(dataset[choice_anchor])
        data_filter["knowledge_id"].extend(label[choice_anchor])
        data_filter["base_code"].extend(base[choice_anchor])

    data_filter = DataFrame(data_filter)
    return data_filter


# The hard triplets are computed for a batch. so, we need a generator
class BatchGenerator(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, shuffle):
        self.x1, self.x2 = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.shuffle = shuffle

    def on_epoch_end(self):
        indexes = np.arange(len(self.y))
        if self.shuffle:
            np.random.shuffle(indexes)
            self.x1 = self.x1[indexes]
            self.x2 = self.x2[indexes]
            self.y = self.y[indexes]

    def __len__(self):
        return int(np.ceil(len(self.y) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x1 = self.x1[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x2 = self.x2[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]

        return [batch_x1, batch_x2, batch_y], [
            np.ones(batch_y.shape),
            np.ones(batch_y.shape),
        ]


unique_train_label = train["knowledge_id"].unique().tolist()
train = generate_data(train, num_train_per_class)

querys_cand = train["question"]
train_x = compute_input_arrays(querys_cand)
train_x = np.array(train_x)
train_y = train["knowledge_id"].tolist()

label2id = {}
for i, label in enumerate(unique_train_label):
    label2id[label] = i + 1
train_y = np.array(list(map(lambda x: label2id[x], train_y)))  # to_category
cls_num = len(unique_train_label)

train_y = np.array(train_y)
print(train_x.shape, train_y.shape)  # (2, 29828, 193) (29828,)


batchGenerator = BatchGenerator(train_x.copy(), train_y.copy(), batch_size, shuffle)


# %%
from sklearn.metrics import classification_report, f1_score

unique_test_base = test["base_code"].astype(str).unique().tolist()


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
            # print(train['knowledge_id'])
            # print(list(map(lambda x:int(x), labels_test)))

            # candidate_data = train  # 候选集-不区分basecode
            candidate_data = train[train["base_code"] == base]  # 候选集-区分basecode
            querys_cand = candidate_data["question"]
            labels_cand = candidate_data["knowledge_id"].astype(str).tolist()
            cand_inputs = compute_input_arrays(querys_cand)
            cand_vecs = self.encoder.predict(
                cand_inputs, batch_size=self.batch_size
            )  # primary embeddings

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


# %%
# 稀疏版AM-Softmax
def get_amsoftmax_loss(labels, scale=30, am_margin=0.45):
    def sparse_amsoftmax_loss(y_true, y_pred):
        y_true = labels
        y_true = K.expand_dims(y_true[:, 0], 1)  # 保证y_true的shape=(None, 1)
        y_true = K.cast(y_true, "int32")  # 保证y_true的dtype=int32
        batch_idxs = K.arange(0, K.shape(y_true)[0])
        batch_idxs = K.expand_dims(batch_idxs, 1)
        idxs = K.concatenate([batch_idxs, y_true], 1)
        y_true_pred = tf.gather_nd(y_pred, idxs)  # 目标特征，用tf.gather_nd提取出来
        y_true_pred = K.expand_dims(y_true_pred, 1)
        y_true_pred_margin = y_true_pred - am_margin  # 减去margin
        _Z = K.concatenate([y_pred, y_true_pred_margin], 1)  # 为计算配分函数
        _Z = _Z * scale  # 缩放结果，主要因为pred是cos值，范围[-1, 1]
        logZ = K.logsumexp(_Z, 1, keepdims=True)  # 用logsumexp，保证梯度不消失
        logZ = logZ + K.log(
            1 - K.exp(scale * y_true_pred - logZ)
        )  # 从Z中减去exp(scale * y_true_pred)
        return -y_true_pred_margin * scale + logZ

    return sparse_amsoftmax_loss


# %%
from keras.layers import Input, Dense, Lambda, Layer
from keras.initializers import Constant
from keras.models import Model
from keras import backend as K

# # Custom loss layer
# # https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example.ipynb
# class CustomMultiLossLayer(Layer):
#     def __init__(self, nb_outputs=2, **kwargs):
#         self.nb_outputs = nb_outputs
#         self.is_placeholder = True
#         super(CustomMultiLossLayer, self).__init__(**kwargs)
        
#     def build(self, input_shape=None):
#         # initialise log_vars
#         self.log_vars = []
#         for i in range(self.nb_outputs):
#             self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
#                                               initializer=Constant(0.), trainable=True)]
#         super(CustomMultiLossLayer, self).build(input_shape)
 
#     def triplet_loss(self, y_true, y_pred):
#         if triplet_strategy == "batch_hard":
#             return batch_hard_triplet_loss(
#                 tf.squeeze(y_true), y_pred, margin=margin, squared=squared
#             )
#         if triplet_strategy == "batch_all":
#             return batch_all_triplet_loss(
#                 tf.squeeze(y_true), y_pred, margin=margin, squared=squared
#             )[0]
#         if triplet_strategy == "batch_semihard":
#             return tf.contrib.losses.metric_learning.triplet_semihard_loss(
#                 labels=tf.squeeze(y_true), embeddings=y_pred, margin=margin
#             )

#     def sparse_amsoftmax_loss(self, y_true, y_pred, scale=30, am_margin=0.45):
#         y_true = K.expand_dims(y_true[:, 0], 1)  # 保证y_true的shape=(None, 1)
#         y_true = K.cast(y_true, "int32")  # 保证y_true的dtype=int32
#         batch_idxs = K.arange(0, K.shape(y_true)[0])
#         batch_idxs = K.expand_dims(batch_idxs, 1)
#         idxs = K.concatenate([batch_idxs, y_true], 1)
#         y_true_pred = tf.gather_nd(y_pred, idxs)  # 目标特征，用tf.gather_nd提取出来
#         y_true_pred = K.expand_dims(y_true_pred, 1)
#         y_true_pred_margin = y_true_pred - am_margin  # 减去margin
#         _Z = K.concatenate([y_pred, y_true_pred_margin], 1)  # 为计算配分函数
#         _Z = _Z * scale  # 缩放结果，主要因为pred是cos值，范围[-1, 1]
#         logZ = K.logsumexp(_Z, 1, keepdims=True)  # 用logsumexp，保证梯度不消失
#         logZ = logZ + K.log(
#             1 - K.exp(scale * y_true_pred - logZ)
#         )  # 从Z中减去exp(scale * y_true_pred)
#         return -y_true_pred_margin * scale + logZ

#     def multi_loss(self, ys_true, ys_pred):
#         assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
#         loss = 0

#         for i, (y_true, y_pred, log_var) in enumerate(zip(ys_true, ys_pred, self.log_vars)):
#             precision = K.exp(-log_var[0])
#             if i==0:
#                 # softmax loss
#                 loss += K.sum(precision * self.sparse_amsoftmax_loss(y_true, y_pred, scale, am_margin) + log_var[0], -1)
#             else:
#                 # triplet loss
#                 t_loss = self.triplet_loss(y_true, y_pred)
#                 if tf.shape(t_loss).shape[0] == 0: 
#                     # loss += tf.constant(1e-9, dtype='float32')
#                     continue
#                 else:
#                     loss += K.sum(precision * t_loss + log_var[0], -1)
#         return K.mean(loss)

#     def call(self, inputs):
#         ys_true = inputs[:self.nb_outputs]
#         ys_pred = inputs[self.nb_outputs:]
#         loss = self.multi_loss(ys_true, ys_pred)
#         self.add_loss(loss, inputs=inputs)
#         # # We won't actually use the output.
#         # return K.concatenate(inputs, -1)
#         return loss


# %%
import os
import keras
from keras.models import Model
from keras.layers import *

# Input for anchor, positive and negative images
x1_in = Input(shape=(None,), name="anchor_input")
x2_in = Input(shape=(None,))
labels = Input(
    shape=(1,), name="label_input"
)  # this will be used for calculating loss only

# Output for anchor, positive and negative embedding vectors
# The bert_model instance is shared (Siamese network)
x = bert_model([x1_in, x2_in])
cls_embedding = Lambda(lambda x: x[:, 0], name="cls_embedding")(x)  # first_token

y = Dropout(dropout_rate, name="dp1")(cls_embedding)  # 添加Dropout
y = Lambda(lambda x: K.l2_normalize(x, axis=1))(y)  # (None, 768)
# from keras.layers.normalization import BatchNormalization
# y = BatchNormalization()(cls_embedding)

# 加入 softmax loss 辅助训练
p_out = Dense(
    cls_num,
    kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
    name="p_out",
)(y)

# Connect the inputs with the outputs
encoding_model = Model(inputs=[x1_in, x2_in], outputs=cls_embedding)
siamese_model = Model(inputs=[x1_in, x2_in, labels], outputs=[p_out, cls_embedding])
siamese_model.summary()

# # 通过“不确定性(uncertainty)”来调整损失函数中的加权超参
# # ValueError: An operation has `None` for gradient.
# m_loss = CustomMultiLossLayer(nb_outputs=2)([labels, labels, p_out, cls_embedding])
# # Connect the inputs with the outputs
# encoding_model = Model(inputs=[x1_in, x2_in], outputs=cls_embedding)
# siamese_model = Model(inputs=[x1_in, x2_in, labels], outputs=m_loss)
# siamese_model.summary()


# %%
from keras.optimizers import Adam

custom_callback = CustomCallback(
    batch_size=batch_size,
    encoder=encoding_model,
)
with session_global.as_default():
    with session_global.graph.as_default():  # 用with新建一个graph，这样在运行完以及异常退出时就会释放内存
        siamese_model.compile(
            loss={
                "cls_embedding": get_loss_function(labels),
                "p_out": get_amsoftmax_loss(labels, scale, am_margin),
            },
            optimizer=Adam(learning_rate),
        )
        history = siamese_model.fit_generator(
            batchGenerator,
            epochs=n_epochs,
            steps_per_epoch=None,
            callbacks=[custom_callback],
        )

import matplotlib.pyplot as plt

plt.plot(history.history["loss"], color="blue")
plt.savefig("./loss.jpg")
plt.show()
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
