# coding=utf-8
"""
@Author:        LiYangMing
@StartTime:     18/12/31
@Filename:      main
@Software:      Pycharm
@LastModify:    19/1/2

个人认为用 Zoo 提供的 TFOptimizer 和 TFDataset 更方
便, 但似乎这个补丁版本还没有在 pip 上发行.
"""

import os
import json
import argparse
import pickle

from pyspark import SparkContext
from zoo import create_spark_conf
from zoo import init_engine

from zoo.pipeline.api import autograd
from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras.layers import Input
from zoo.pipeline.api.keras.layers import Dense
from zoo.pipeline.api.keras.layers import Flatten
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import TimeDistributed
from zoo.pipeline.api.keras.layers import Convolution2D
from zoo.pipeline.api.keras.layers import AveragePooling2D

from zoo.pipeline.api.net import TFDataset
from zoo.pipeline.api.net import TFOptimizer
from bigdl.optim.optimizer import MaxEpoch
from bigdl.optim.optimizer import TrainSummary
from bigdl.optim.optimizer import ValidationSummary

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", "--dd", type=str, default="./data")
parser.add_argument("--save_dir", "--sd", type=str, default="./save")
parser.add_argument("--num_epoch", "-ne", type=int, default=64)
parser.add_argument("--batch_size", "-bs", type=int, default=32)
parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)

LAYER_1_NUM_CHANNEL = 8         # 第一层卷积通道数.
CONVOLVE_1_KERNEL_SIZE = 9      # 第一层卷积核窗口大小.
POOLING_1_WINDOW_SIZE = 2       # 第一层池化层窗口大小.
POOLING_1_STRIDE_SIZE = 2       # 第一层池化层滑动步长大小.
LAYER_2_NUM_CHANNEL = 2         # 第二层卷积通道数.
CONVOLVE_2_KERNEL_SIZE = 5      # 第二层卷积核窗口大小.
POOLING_2_WINDOW_SIZE = 2       # 第二层池化层窗口大小.
POOLING_2_STRIDE_SIZE = 2       # 第二层池化层滑动步长大小.
FC_LINEAR_DIMENSION = 64        # 全连接层的线性维度大小.

args = parser.parse_args()
print json.dumps(args.__dict__, indent=True, ensure_ascii=False)

# 初始化 Pyspark + Analytic-Zoo 环境.
sc = SparkContext.getOrCreate(
    conf=create_spark_conf()
    .setMaster("local[16]")
    .set("spark.driver.memory", "512m")
    .setAppName("OneShotLearning")
)
init_engine()

# 读入经 dump 保存好的数据.
train_img = pickle.load(
    open(os.path.join(args.data_dir, "train_image.pkl"), "r")
)
train_lbl = pickle.load(
    open(os.path.join(args.data_dir, "train_label.pkl"), "r")
)
test_img = pickle.load(
    open(os.path.join(args.data_dir, "test_image.pkl"), "r")
)
test_lbl = pickle.load(
    open(os.path.join(args.data_dir, "test_label.pkl"), "r")
)

# 交换图像的维度适应 keras 并归一化像素值.
t_train_img = train_img.transpose((0, 1, 4, 2, 3)) / 225.0
t_test_img = test_img.transpose((0, 1, 4, 2, 3)) / 225.0

NUM_TRAIN_SMP, _, IMAGE_SIZE, _, NUM_IMAGE_CHANNEL = train_img.shape
NUM_TEST_SMP, NUM_CLASS_LABEL, _, _, _ = test_img.shape

# 将数据转为 RDD 的形式.
train_rdd = sc.parallelize(t_train_img).zip(sc.parallelize(train_lbl))
test_rdd = sc.parallelize(t_test_img).zip(sc.parallelize(test_lbl))

# 用 Zoo-Keras 定义模型的网络结构.
input_shape = (NUM_CLASS_LABEL, NUM_IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
both_input = Input(shape=input_shape)

convolve_net = Sequential()
convolve_net.add(Convolution2D(
    nb_filter=LAYER_1_NUM_CHANNEL,      # 通道: 4 -> 8.
    nb_row=CONVOLVE_1_KERNEL_SIZE,      # 尺寸: 32 - 9 + 1 = 24
    nb_col=CONVOLVE_1_KERNEL_SIZE,
    activation="relu",
    input_shape=(
        NUM_IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE
    )
))
convolve_net.add(AveragePooling2D(
    pool_size=(
        POOLING_1_WINDOW_SIZE,          # 尺寸: 24 / 2 = 12.
        POOLING_1_WINDOW_SIZE
    ),
    strides=(
        POOLING_1_STRIDE_SIZE,
        POOLING_1_STRIDE_SIZE
    )
))
convolve_net.add(Convolution2D(
    nb_filter=LAYER_2_NUM_CHANNEL,      # 通道: 8 -> 2.
    nb_row=CONVOLVE_2_KERNEL_SIZE,      # 尺寸: 12 - 5 + 1 = 8.
    nb_col=CONVOLVE_2_KERNEL_SIZE,
    activation="relu"
))
convolve_net.add(AveragePooling2D(
    pool_size=(
        POOLING_2_WINDOW_SIZE,          # 尺寸: 8 / 2 = 4.
        POOLING_2_WINDOW_SIZE
    ),
    strides=(
        POOLING_2_STRIDE_SIZE,
        POOLING_2_STRIDE_SIZE
    )
))
convolve_net.add(Flatten())             # 尺寸: 4 * 4 * 2 -> 32
convolve_net.add(Dense(
    output_dim=FC_LINEAR_DIMENSION,     # 尺寸: 32 -> 64.
    activation="sigmoid"
))

# BigDL 不支持 parameter sharing, 不得已而为之.
both_feature = TimeDistributed(
    layer=convolve_net,
    input_shape=input_shape
)(both_input)

encode_left = both_feature.index_select(1, 0)
encode_right = both_feature.index_select(1, 1)

distance = autograd.abs(encode_left - encode_right)
predict = Dense(
    output_dim=NUM_CLASS_LABEL,
    activation="sigmoid"
)(distance)

siamese_net = Model(
    input=both_input, output=predict
)
siamese_net.compile(
    optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics=["accuracy"]
)

# 构造分布式的数据集对象.
data_set = TFDataset.from_rdd(
    train_rdd, shapes=[input_shape, [1]],
    batch_size=args.batch_size, val_rdd=test_rdd
)

optimizer = TFOptimizer.from_keras(siamese_net, data_set)
app_name = "Siamese Network"

optimizer.set_train_summary(TrainSummary("tmp", app_name))
optimizer.set_val_summary(ValidationSummary("tmp", app_name))
optimizer.optimize(end_trigger=MaxEpoch(args.num_epoch))
