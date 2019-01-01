# PySpark + Intel Analytics 实现孪生神经网络

HIT 大数据课大作业,完成思路 (可用内存 < 2G, 图像数据 1 G)：

>+  使用 Spark Streaming 流式读入图像并做预处理;
>+  使用 Intel 提供了分布式深度学习框架搭建学习器;
>+  模仿结构 [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf);

环境要求 Python 2.7 + Pyspark 2.3.2 + BigDL 0.7.1 + Analytics-Zoo.
