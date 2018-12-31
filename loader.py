# coding=utf-8
"""
@Author:        LiYangMing
@StartTime:     18/12/31
@Filename:      loader
@Software:      Pycharm
@LastModify:    18/12/31
"""

import os
import tqdm
import random
import pickle
import numpy as np
from scipy import misc
from copy import deepcopy
from itertools import combinations


def collect_data(data_dir):
    serial_dict = {}

    for class_index in tqdm.tqdm(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_index)
        serial_dict[class_index] = []

        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)

            image_array = misc.imread(file_path)
            serial_dict[class_index].append(image_array)

    return serial_dict


def generate_data(data_dict, num_pos, num_neg):
    single_index_list = deepcopy(data_dict.keys())

    # 随机选出训练集的序号表.
    np.random.shuffle(single_index_list)
    pos_index_list = single_index_list[:num_pos]

    # 随机选出测试集的序号表.
    pair_index_list = list(combinations(single_index_list, 2))
    np.random.shuffle(pair_index_list)
    neg_index_list = pair_index_list[:num_neg]

    pos_data_list = []
    for epoch in tqdm.tqdm(pos_index_list):

        # 可能有些类别只有一个样例.
        try:
            idx, jdx = np.random.choice(
                list(range(len(data_dict.get(epoch)))),
                2, replace=False
            )
        except ValueError:
            continue

        img_i = data_dict.get(epoch)[idx]
        img_j = data_dict.get(epoch)[jdx]

        pos_data_list.append(np.concatenate(
            [img_i[np.newaxis, :], img_j[np.newaxis, :]],
            axis=0
        ))

    neg_data_list = []
    for idx, jdx in tqdm.tqdm(neg_index_list):
        img_i = random.choice(data_dict.get(idx))
        img_j = random.choice(data_dict.get(jdx))

        neg_data_list.append(np.concatenate(
            [img_i[np.newaxis, :], img_j[np.newaxis, :]],
            axis=0
        ))

    x_list = pos_data_list + neg_data_list
    y_list = [1] * len(pos_data_list) + [0] * len(neg_data_list)

    # 把输入 x 和标注 y 都打包成数组 array.
    x_array = np.array(x_list)
    y_array = np.array(y_list)

    index_list = list(range(0, x_array.shape[0]))
    np.random.shuffle(index_list)

    # 将数据重新洗牌, data augment.
    rx_array = x_array[index_list, :]
    ry_array = y_array[index_list]

    return rx_array, ry_array


def dump_to_file(obj, file_path):
    with open(file_path, "wb") as fw:
        pickle.dump(obj, fw)


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    # 读取训练, 测试数据.
    train_dict = collect_data("data/train_img")
    test_dict = collect_data("data/test_img")

    # 产生张量数据用于训练.
    train_img, train_lbl = generate_data(train_dict, 500, 700)
    test_img, test_lbl = generate_data(test_dict, 149, 223)

    # 保存数据, dump 到文件路径.
    dump_to_file(train_img, "data/train_image.pkl")
    dump_to_file(train_lbl, "data/train_label.pkl")
    dump_to_file(test_img, "data/test_image.pkl")
    dump_to_file(test_lbl, "data/test_label.pkl")
