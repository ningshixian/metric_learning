#! -*- coding:utf-8 -*-
import math
import os
import json
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import argparse
import xlrd

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold

from sentence_transformers import SentenceTransformer, SentencesDataset, util
from sentence_transformers import InputExample, evaluation, losses
from torch.utils.data import DataLoader
import torch

"""
基于双塔SBERT模型的智能语义计算实验: https://zhuanlan.zhihu.com/p/351678987
SBERT文档：https://www.sbert.net/docs/package_reference/losses.html?highlight=inputexample

nohup python sbert_triplet_pytorch.py > log_sbert.txt 2>&1 &
"""

# from util import read_tsv_data
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 选择测试集用的
sheet2base = {
    "HALO测试内容": "HALOBIGROOMBASE",
    "差旅测试内容": "CLXTBASE",
    "HR-π": "HRPBASE",
    "成本管理平台": "CBGLXTBASE",
    "场景化费用": "CJHFYXTBASE",
    "供应商": "GYSGXGLPTBASE",
    "商业资产": "C2ZGXTBASE",
}

batch_size = 32
sbert = SentenceTransformer(
    "../corpus/distilbert-multilingual-nli-stsb-quora-ranking", device="cuda"
)  # device='cuda' 如果为None，则检查是否可以使用GPU。


def read_xlrd(excelFile, tabel_index):
    data = xlrd.open_workbook(excelFile)
    table = data.sheet_by_index(tabel_index)
    dataFile = []

    for rowNum in range(table.nrows):
        # if 去掉表头
        if rowNum > 0:
            dataFile.append(table.row_values(rowNum))
    return dataFile


def sbert_model_train(data):
    # data = data[:2000]
    lenth = len(data)
    idx = int(lenth * 0.8)

    # Define your train examples.
    train_datas = []
    for i in data[:idx]:
        train_datas.append(InputExample(texts=[i[0], i[1], i[2]]))

    # Define your evaluation examples
    sentences1, sentences2, labels = [], [], []
    for i in data[idx:]:
        sentences1.append(i[0])
        sentences2.append(i[1])
        labels.append(1)
        sentences1.append(i[0])
        sentences2.append(i[2])
        labels.append(0)

    evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, labels)

    # Define your train dataset, the dataloader and the train loss
    train_dataset = SentencesDataset(train_datas, sbert)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    train_loss = losses.BatchHardSoftMarginTripletLoss(sbert)  # CosineSimilarityLoss TripletLoss BatchSemiHardTripletLoss

    # Tune the model
    sbert.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=10,
        warmup_steps=100,   # the learning rate is increased from o up to the maximal learning rate
        evaluator=evaluator,
        evaluation_steps=200,
        output_path="./sbert_triplet_model",
        save_best_model=True,
        callback=None,
        show_progress_bar=True,
    )
    print("训练完成！")
    return sbert


def sbert_model_test():
    all_pred_1_list, all_val_1_list = [], []
    all_pred_3_list, all_val_3_list = [], []
    all_pred_5_list, all_val_5_list = [], []
    # model = SentenceTransformer("./sbert_triplet_model")
    model = sbert

    for base in tqdm(unique_test_base):
        test_data = test[test["base_code"] == base]  # test
        querys = test_data["question"].tolist()
        valid_outputs = test_data["knowledge_id"].astype(str).tolist()  # 正确kid列表

        candidate_data = train[train["base_code"] == base]  # candidate
        all_cand_text = candidate_data["question"].tolist()
        all_cand_index = candidate_data["knowledge_id"].astype(str).tolist()

        # Sentences are encoded by calling model.encode()
        emb1 = model.encode(querys)
        emb2 = model.encode(all_cand_text)  # primary embeddings

        cos_sim = util.pytorch_cos_sim(emb1, emb2)
        print("Cosine-Similarity:", cos_sim)
        print("test num: ", len(cos_sim))
        print(base, len(cos_sim[0]))

        pred_1_list, val_1_list = [], []
        pred_3_list, val_3_list = [], []
        pred_5_list, val_5_list = [], []
        for idx, one in enumerate(cos_sim):  # one:input embedding
            # 正确kid列表
            one_anwser_list = valid_outputs[idx].split("###")
            one_anwser_list = [str(int(one_a)) for one_a in one_anwser_list]

            # top1预测结果
            max_idx = np.argmax(one)
            pred_one = str(int(all_cand_index[max_idx]))  # 预测的kid
            if one_anwser_list[0] not in all_cand_index:
                print(one_anwser_list, " not in candidate set")
                continue
            if pred_one in one_anwser_list:
                val_1_list.append(pred_one)
            else:
                val_1_list.append(one_anwser_list[0])
                # print("wrong {} pid:{} kid:{}".format(querys[idx], pred_one, one_anwser_list,))
            pred_1_list.append(pred_one)

            # top3
            max_idx = np.argpartition(one, -3)[-3:]
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
            max_idx = np.argpartition(one, -5)[-5:]
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

        print("cat: ", base)
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
    # self.model.save_weights("it_nlu-{name}.h5".format(name=epoch))
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


# 读取测试集 用户输入和主问题ID
def read_valid_dataset(input_categories):
    res = list()
    for idx, one_line in enumerate(input_categories):
        q = one_line[0]
        if (
            len(str(one_line[2])) <= 0
            or len(str(one_line[3])) <= 0
            or "," in str(one_line[2])
            or str(one_line[3]).strip() == ""
        ):
            continue
        label = str(one_line[3]).split("###")
        sim = [str(int(float(str(re.sub(r"\s+", "", one))))) for one in label]
        res.append((q, sim))
    print("测试集数量：", len(res))
    return res


#模型向量召回
def faiss_recall():
    from tqdm import tqdm
    import numpy as np
    import faiss                   # make faiss available

    ALL = []
    for i in tqdm(test_data):
        ALL.append(i[0])
        ALL.append(i[1])
    ALL = list(set(ALL))

    DT = model.encode(ALL)
    DT = np.array(DT, dtype=np.float32)

    # https://waltyou.github.io/Faiss-Introduce/
    index = faiss.IndexFlatL2(DT[0].shape[0])   # build the index
    print(index.is_trained)
    index.add(DT)                  # add vectors to the index
    print(index.ntotal)


def get_triplets_batch(train_set, unique_train_label, map_train_label_indices):
    idxs_a, idxs_p, idxs_n = [], [], []
    for label in tqdm(unique_train_label):
        rows = map_train_label_indices[label]
        if len(rows) > 2:
            a = rows[0]
            for p in rows[1:]:
                neg_label = np.random.choice(unique_train_label)
                while neg_label==label:
                    neg_label = np.random.choice(unique_train_label)
                n = map_train_label_indices[neg_label].tolist()[0]
                idxs_a.append(a)
                idxs_p.append(p)
                idxs_n.append(n)

    a = train_set.iloc[idxs_a].values.tolist()
    b = train_set.iloc[idxs_p].values.tolist()
    c = train_set.iloc[idxs_n].values.tolist()

    return list(zip(a,b,c))


if __name__ == "__main__":
    test = pd.read_csv("DatasetLXH/fewshot_test.csv", engine="python")
    train = pd.read_csv("DatasetLXH/fewshot_train.csv", engine="python")  # , nrows =10240

    unique_test_base = test["base_code"].unique().tolist()
    unique_train_label = np.array(train["knowledge_id"].unique().tolist())
    labels_train = np.array(train["knowledge_id"].tolist())
    map_train_label_indices = {
        label: np.flatnonzero(labels_train == label) for label in unique_train_label
    }  # 非零元素的索引

    # data_xlx = get_triplets_batch(train['question'],unique_train_label,map_train_label_indices) #[:100]
    # sbert = sbert_model_train(data_xlx)
    # sbert_model_test()

    n_epoch = 20
    train_batch_size = 128
    train_examples = [InputExample(texts=[row["question"]], label=int(row["knowledge_id"])) for index,row in train.iterrows()]
    train_dataset = SentencesDataset(train_examples, sbert)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.BatchSemiHardTripletLoss(model=sbert)  # BatchSemiHardTripletLoss BatchHardTripletLoss TripletLoss
    for i in range(n_epoch):
        # Tune the model
        sbert.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=100,   # the learning rate is increased from o up to the maximal learning rate
            # evaluator=evaluator,
            # evaluation_steps=200,
            # output_path="./sbert_triplet_model",
            # save_best_model=True,
            callback=None,
            show_progress_bar=True,
        )
        print("第{}轮次训练完成！".format(i))
        sbert_model_test()
