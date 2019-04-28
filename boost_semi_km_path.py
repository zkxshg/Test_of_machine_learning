# import package：numpy、pandas
import numpy as np
import datetime
import sys
import random as rand


def semi_km(ori, label, label_data, label_2, max_iter, label_uni3):
    attr_num = len(ori[0])  # number of attributes
    tup_num = len(ori)  # number of tuples
    labeled_num = len(label_data)  # number of labeled samples

    label_uni = label_uni3  # total classes
    n_clusters = len(label_uni)  # k = number of classes

    clu_arr = [tup_num]  # save the cluster of each tuple
    mean_arr = []  # save the mean position
    label_arr = []  # save the clusters of labeled data

    change_num = tup_num + 1  # number of changed points in each iter
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
        mean_arr.append(sum_label/sum_num)  # the mean point of cluster/class j
    for i in range(0, tup_num):
        clu_arr.append(0)
    # =========================iteration ================================
    iter_num = 0
    change_con = int(tup_num*0.0001)
    while iter_num < 500 and change_num > 1:
        clu_arr, change_num = assign(mean_arr, ori, clu_arr)  # assign
        mean_arr = update(ori, clu_arr, n_clusters, label_data, label_arr)  # update
        iter_num += 1
    # =========================accuracy ================================
    clu_class = label_uni
    num_t = 0
    for i in range(0, tup_num):
        if clu_class[clu_arr[i]] == label[i]:
            num_t +=1
    accuracy = num_t/tup_num  # compute the accuracy
    # =========================SSE ================================
    SSE = sse(ori, clu_arr, label_data, label_arr, mean_arr)
    return clu_arr, label_arr


# =========================accuracy ================================
def acc(clu_arr, label, label_u, labeled_num):
    tup_num = len(clu_arr)
    num_t = 0
    for i3 in range(0, tup_num):
        if label_u[clu_arr[i3]] == label[i3]:
            num_t += 1
    for j in range(0, labeled_num):
            num_t += 1
    accuracy = num_t/(tup_num + labeled_num)
    return accuracy


# =========================SSE ===============================
def sse(ori, clu_arr, label_data, label_arr, mean_arr):
    SSE = 0
    tup_num = len(ori)  # number of tuples
    labeled_num = len(label_data)  # number of labeled samples
    for i in range(0, tup_num):
        SSE += np.square(np.linalg.norm((ori[i] - mean_arr[clu_arr[i]])))
    # for j in range(0, labeled_num):
        # SSE += np.square(np.linalg.norm((label_data[j] - mean_arr[label_arr[j]])))
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
            if dist < min_dist:
                min_in = j
                min_dist = dist
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


def main(argv):
    if len(argv) < 4:
        print("請按照以下格式輸入： unlabeled_data labeled_data max_iteration")
        return

    if argv[1] == './data/iris_unlabeled.csv':
        data = np.loadtxt(argv[1], delimiter=",", skiprows=1, usecols=(0, 1, 2, 3))
        labels = np.loadtxt(argv[1], delimiter=",", dtype=str, usecols=(4))
        labeled_data = np.loadtxt(argv[2], delimiter=",", skiprows=1, usecols=(0, 1, 2, 3))
        labeled = np.loadtxt(argv[2], delimiter=",", dtype=str, usecols=(4))
    else:
        data = np.loadtxt(argv[1], delimiter=",", skiprows=1, usecols=(1, 2, 3, 4, 5, 6, 7))
        labels = np.loadtxt(argv[1], delimiter=",", dtype=str, usecols=(0))
        labeled_data = np.loadtxt(argv[2], delimiter=",", skiprows=1, usecols=(1, 2, 3, 4, 5, 6, 7))
        labeled = np.loadtxt(argv[2], delimiter=",", dtype=str, usecols=(0))
    start_time = datetime.datetime.now()
    iteration = int(argv[3])  # 最大迭代次數
    label_uni2 = list(set(labeled))  # total classes
    k = len(label_uni2)  # k = number of classes
    attr_len = len(data[1])  # total number of attribute
    # initial the weight of attribute
    w_a = []
    for i in range(0, attr_len):
        w_a.append(1.0)
        # w_a.append(np.random.rand())

    # record the d(sse) of each attribute
    sse_p = []
    for i in range(0, attr_len):
        sse_p.append(0)

    # initial the iteration
    # data -> label -> mean -> sse
    (ori_label, label_2) = semi_km(data, labels, labeled_data, labeled, iteration, label_uni2)
    mean_ori = update(data, ori_label, k, labeled_data, label_2)
    ori_sse = sse(data, ori_label, labeled_data, label_2, mean_ori)
    # print(ori_sse)
    # initial the parameter
    sse_diff = ori_sse
    last_sse = ori_sse
    sse_w = 0
    acc_w = 0
    iter_num = 0
    data_w = np.multiply(data, w_a)
    labeled_w = np.multiply(labeled_data, w_a)

    n = 0.8  # Learning Factor
    while (abs(sse_diff) > 0.001) and iter_num < iteration:  # start iteration
        max_dff = -99999  # Save the Max(d(SSE))
        max_index = -1  # Save the index of changed attribute
        for p in range(0, attr_len):  # calculate the d(sse) of deleting attribute
            w_a_d = w_a.copy()
            w_a_d[p] = w_a_d[p] * n  # Calculate the new weight
            data_p = np.multiply(data, w_a_d)  # Weighting
            labeled_p = np.multiply(labeled_data, w_a_d)  # Weighting
            (p_label, labeled_2) = semi_km(data_p, labels, labeled_p, labeled, iteration, label_uni2)  # New model
            mean_p = update(data, p_label, k, labeled_data, labeled_2)  # Use the new labels to update the model
            model_p_sse = sse(data, p_label, labeled_data, labeled_2, mean_p)  # SSE after weighting
            sse_pa = last_sse - model_p_sse  # The change of SSE
            if sse_pa > max_dff:  # Select the model with min(sse)
                max_index = p
                max_dff = sse_pa
        if max_index != -1:
            w_a[max_index] = w_a[max_index] * n  # Apply the new weight
        data_w = np.multiply(data, w_a)  # Re-weighing and calculate the change of SSE
        labeled_w = np.multiply(labeled_data, w_a)  # apply to the data
        (model_label, model_2) = semi_km(data_w, labels, labeled_w, labeled, iteration, label_uni2)  # modeling
        mean_model = update(data, model_label, k, labeled_data, model_2)  # update
        acc_w = acc(model_label, labels, label_uni2, len(model_2))
        sse_w = sse(data, model_label, labeled_data, model_2, mean_model)  # SSE
        sse_diff = last_sse - sse_w  # Calculate the d(SSE)
        last_sse = sse_w  # Save the iter`s result
    print("SSE is:")
    print(sse_w)
    print("Accuracy is:")
    print(acc_w)
    end_time = datetime.datetime.now()
    print("time cost is:")
    print(end_time - start_time)


if __name__=='__main__':
    main(sys.argv)






