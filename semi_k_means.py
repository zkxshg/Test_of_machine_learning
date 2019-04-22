# import package：numpy、pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rand


def semi_km(ori, label, label_data, label_2, max_iter):
    attr_num = len(ori[0])  # number of attributes
    tup_num = len(ori)  # number of tuples
    labeled_num = len(label_data)  # number of labeled samples
    label_uni = list(set(label_2))  # total classes
    n_clusters = len(label_uni)  # k = number of classes
    clu_arr = [tup_num]  # save the cluster of each tuple
    mean_arr = []  # save the mean position
    label_arr = []  # save the clusters of labeled data
    change_num = tup_num + 1  # number of changed points in each iter
    # dist_arr = [n_clusters]
    # count_arr = []  # save the total number of each cluster
    # ===========================initial ==========================
    for j in range(0, labeled_num):
        label_arr.append(-1)
    for i in range(0, n_clusters):
        sum_label = []
        sum_num = 0
        for i2 in range(0, attr_num):
            sum_label.append(0)
        for j in range(0, labeled_num):
            if label_arr[j] == -1:
                if label_2[j] == label_uni[i]:
                    sum_label = np.add(sum_label, label_data[j])
                    label_arr[j] = i
                    sum_num += 1
        mean_arr.append(sum_label/sum_num)
    print(mean_arr)
    for i in range(0, tup_num):
        clu_arr.append(0)

    # =========================iteration ================================
    iter_num = 0
    change_con = int(tup_num*0.001)
    while iter_num < max_iter and change_num > 1:
        clu_arr, change_num = assign(mean_arr, ori, clu_arr)
        #  update
        mean_arr = update(ori, clu_arr, n_clusters, label_data, label_arr)
        iter_num += 1
        print("iter_num")
        # print(change_num)
        print(iter_num)
        SSE2 = 0
        for i in range(0, tup_num):
            SSE2 += np.square(np.linalg.norm((ori[i] - mean_arr[clu_arr[i]])))
        # for j in range(0, labeled_num):
        # SSE += np.square(np.linalg.norm((label_data[j] - mean_arr[label_arr[j]])))
        print(SSE2)
    # print(clu_arr)
    # print("=========")
    # print(change_num)

    # =========================accuracy ================================
    clu_class = label_uni
    print(clu_class)
    # compute the accuracy
    num_t = 0
    for i in range(0, tup_num):
        if clu_class[clu_arr[i]] == label[i]:
            num_t +=1
    for j in range(0, labeled_num):
            num_t += 1
    accuracy = num_t/(tup_num + labeled_num)
    print("Accuracy is : " )
    print(accuracy)

    # =========================SSE ================================
    SSE = 0
    for i in range(0, tup_num):
        SSE += np.square(np.linalg.norm((ori[i] - mean_arr[clu_arr[i]])))
    # for j in range(0, labeled_num):
        # SSE += np.square(np.linalg.norm((label_data[j] - mean_arr[label_arr[j]])))
    print(SSE)
    return SSE


# =========================assign================================
def assign(mean_arr, ori, pre_clu):
    tup_num = len(ori)
    n_clusters = len(mean_arr)
    clu_arr = []  # save the cluster of each tuple
    change_num = 0  # number of changed points in each iter
    for i in range(0, tup_num):
        clu_arr.append(0)
    for i in range(0, tup_num):
        min_in = -1
        # min_dist = np.linalg.norm((ori[i] - mean_arr[0]))
        # min_dist = np.dot(ori[i], mean_arr[0])
        # min_dist = np.dot(ori[i], mean_arr[0])/(np.linalg.norm(ori[i])*(np.linalg.norm(mean_arr[0])))
        min_dist = 999999
        for j in range(0, n_clusters):
            # dist = np.linalg.norm((ori[i] - mean_arr[j]))
            dist = np.linalg.norm((ori[i] - mean_arr[j]), ord=1)
            # dist = np.dot(ori[i], mean_arr[j])
            #  dist = np.dot(ori[i], mean_arr[j]) / (np.linalg.norm(ori[i]) * (np.linalg.norm(mean_arr[j])))
            # print("dist")
            # print(dist)
            if dist < min_dist:
                min_in = j
                min_dist = dist
        # print(min_dist)
        clu_arr[i] = min_in
        if clu_arr[i] != pre_clu[i]:
            change_num += 1
    return clu_arr, change_num


# =========================update================================
def update(ori, clu_arr, n_clusters, label_data, label_arr):
    mean_arr = []
    for i in range(0, n_clusters):
        sum_num = 0
        sum_arr = np.zeros(len(ori[0]))
        for j in range(0, len(ori)):
            if clu_arr[j] == i:
                sum_num += 1
                sum_arr = np.add(sum_arr, ori[j])
                # print(sum_arr)
        for j2 in range(0, len(label_data)):
            if label_arr[j2] == i:
                sum_num += 1
                sum_arr = np.add(sum_arr, label_data[j2])
        mean_arr.append(sum_arr/sum_num)
    return mean_arr


if __name__=='__main__':
    # 參數初始化
    output_file = 'data_type.xls'  # 結果保存路徑
    # k = 3  # 聚類類別
    iteration = 500  # 最大迭代次數
    # ==============================Iris=======================
    '''
    data = np.loadtxt('iris_unlabeled.csv', delimiter=",", skiprows=1, usecols=(0, 1, 2, 3))
    labels = np.loadtxt('iris_unlabeled.csv', delimiter=",", dtype=str, usecols=(4))
    labeled_data = np.loadtxt('iris_label.csv', delimiter=",", skiprows=1, usecols=(0, 1, 2, 3))
    labeled = np.loadtxt('iris_label.csv', delimiter=",", dtype=str, usecols=(4))
    '''
    # ==============================abalone=======================

    data = np.loadtxt('abalone_unlabeled.csv', delimiter=",", skiprows=1, usecols=(1, 2, 3, 4, 5, 6, 7))
    labels = np.loadtxt('abalone_unlabeled.csv', delimiter=",", dtype=str, usecols=(0))
    labeled_data = np.loadtxt('abalone_label.csv', delimiter=",", skiprows=1, usecols=(1, 2, 3, 4, 5, 6, 7))
    labeled = np.loadtxt('abalone_label.csv', delimiter=",", dtype=str, usecols=(0))
    # data = 1.0 * (data - data.mean()) / data.std()
    # labeled_data = 1.0 * (labeled_data - labeled_data.mean()) / labeled_data.std()

    # print(labels)
    model = semi_km(data, labels, labeled_data, labeled, iteration)

    # print(list(set(labels)))
    # 數據預處理
    # print(np.isnan(data).any())
    # print(data.isnull().sum())
    # data.fillna(0)
    # print(data.isnull().sum())
    # print(np.isnan(data).any())
    # data = 1.0*(data - data.mean())/data.std() #数据标准化
