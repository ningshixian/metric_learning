
from base.base_model import BaseModel
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


class LVQ2(BaseModel):
    def __init__(self, config):
        super(LVQ2, self).__init__(config)
        self.train_model = self.build_model     # 返回方法本身

    """基于欧式距离的字符串相似度计算"""

    def distance(self, x, y):
        """ 计算两个向量之间的距离
        :param x: 第一个向量
        :param y: 第二个向量
        :return: 返回计算值
        """
        return np.sqrt(np.sum(np.power(x[:-1] - y[:-1], 2)))

    def euclid_distance(self, X, weight):
        X = np.expand_dims(X, axis=0)
        euclid_dist = np.linalg.norm(X - weight, axis=1)
        return np.expand_dims(euclid_dist, axis=0)

    def n_argmin(self, array, n, axis=0):
        sorted_argumets = array.argsort(axis=axis).ravel()
        return sorted_argumets[:n]

    def rand_initial_center(self, data, k_num, labels):
        # 原型向量
        v = np.empty((k_num, data.shape[1]), dtype=np.float32)  # (8896, 769)
        # 初始化原型向量，从每一类中随机选取样本，如果类别数小于聚类数，循环随机取各类别中的样本
        for i in range(k_num):
            # 获取当前label对应的原始数据集合
            samples = data[data[:, -1] == labels[i % k_num]]
            # 随机选择一个点作为初始簇心
            v[i] = random.choice(samples)
        return v

    """搭建网络"""

    def build_model(self, data: np, k_num: int, labels: list, lr=0.01, max_iter=15000, delta=1e-3, epsilon=0.1):
        """
        LVQ2算法
        :param data: 样本集, 最后一列feature表示原始数据的label
        :param k_num: 簇数，原型向量个数 即：len(labels)
        :param labels: 1-dimension list or array,label of the data（去重）
        :param max_iter: 最大迭代数
        :param lr: 学习效率
        :param delta: max distance for two vectors to be 'equal'.
        :param epsilon:
        :return: 返回向量中心点、簇标记
        """
        # 随机初始化K个原型向量
        v = self.rand_initial_center(data, k_num, labels)

        # 确认是否所有中心向量均已更新
        # all_vectors_updated = np.empty(shape=(k_num,), dtype=np.bool)     # 随机值
        all_vectors_updated = np.zeros(shape=(k_num,), dtype=np.bool)       # 全FALSE
        # 记录各个中心向量的更新次数
        v_update_cnt = np.zeros(k_num, dtype=np.float32)

        from collections import Counter
        co = [1 for k,v in dict(Counter(data[:, -1])).items() if v==1]
        print('无相似问的主问题数量：', len(co))    # 2222

        j = -1
        jyp = 0
        while True:
            j = j + 1
            if j % 100 == 0:
                nb = len(list(filter(lambda x:x==False, all_vectors_updated)))
                print("iter: ", j, "\t剩余未更新簇中心数量: ", nb)

            # 迭代停止条件：已到达最大迭代次数，且原型向量均已更新
            if j >= max_iter or all_vectors_updated.all():
                break
            # # 迭代停止条件：超过阈值且每个中心向量都更新超过5次则退出
            # if j >= max_iter and sum(v_update_cnt > 5) == k_num:
            #     break

            # # 随机选择一个样本, 并计算与当前各个簇中心点的距离, 取距离最小的和次小的
            # sel_sample = random.choice(data)
            # min_dist = self.distance(sel_sample, v[0])
            # sec_dist = min_dist
            # sel_k_1, sel_k_2 = 0, 0
            # for ii in range(1, k_num):
            #     dist = self.distance(sel_sample, v[ii])
            #     if min_dist > dist:
            #         sec_dist = min_dist
            #         min_dist = dist
            #         sel_k_2 = sel_k_1
            #         sel_k_1 = ii

            # # 随机选择一个样本
            # sel_sample = random.choice(data)

            # 按类别顺序来取样本
            samples = data[data[:, -1] == labels[jyp % k_num]]
            while len(samples) == 1:
                all_vectors_updated[jyp % k_num] = True
                jyp += 1
                samples = data[data[:, -1] == labels[jyp % k_num]]
            jyp += 1

            max_r = min(20, len(samples))
            # max_r = len(samples)   # 速度太慢了！
            for k in range(max_r):
                # sel_sample = random.choice(samples)
                sel_sample = samples[k]
                # 计算样本与当前各个簇中心点的距离, 取距离最小的和次小的
                output = self.euclid_distance(sel_sample, v)
                winner_subclasses = self.n_argmin(output, n=2, axis=1)
                sel_k_1, sel_k_2 = winner_subclasses
                min_dist, sec_dist = output[0, sel_k_1], output[0, sel_k_2]

                # 保存更新前向量
                temp_v = v[sel_k_1].copy()

                double_update_condition_satisfied = (
                    not sel_sample[-1] == v[sel_k_1][-1]
                    and (sel_sample[-1] == v[sel_k_2][-1])
                    and min_dist > ((1 - epsilon) * sec_dist)
                    and sec_dist < ((1 + epsilon) * min_dist)
                )

                # 更新v
                if double_update_condition_satisfied:
                    v[sel_k_1][0:-1] = v[sel_k_1][0:-1] - lr * (
                        sel_sample[0:-1] - v[sel_k_1][0:-1]
                    )
                    v[sel_k_2][0:-1] = v[sel_k_2][0:-1] + lr * (
                        sel_sample[0:-1] - v[sel_k_2][0:-1]
                    )
                elif sel_sample[-1] == v[sel_k_1][-1]:
                    v[sel_k_1][0:-1] = v[sel_k_1][0:-1] + lr * (
                        sel_sample[0:-1] - v[sel_k_1][0:-1]
                    )
                else:
                    v[sel_k_1][0:-1] = v[sel_k_1][0:-1] - lr * (
                        sel_sample[0:-1] - v[sel_k_1][0:-1]
                    )

                # 更新记录数组（原型向量更新很小甚至不再更新，即可）
                if self.distance(temp_v, v[sel_k_1]) < delta:
                    all_vectors_updated[sel_k_1] = True
                # v的更新次数+1
                v_update_cnt[sel_k_1] = v_update_cnt[sel_k_1] + 1

        # 更新完毕后, 把各个样本点进行标记, 记录放在categories变量里
        m, n = np.shape(data)
        cluster_assment = np.mat(np.zeros((m, 2)), dtype=np.float32)
        # for i in tqdm(range(m)):
        #     output = self.euclid_distance(data[i, :], v)
        #     min_distji_index = int(output.argmin())
        #     cluster_assment[i, 0] = min_distji_index
        #     cluster_assment[i, 1] = output[0, min_distji_index]

        return v, cluster_assment
