# coding=utf-8
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

import re
import os
import tqdm
import time
import numpy as np
from PIL import Image


class ImageProcessor(object):
    """
    基于 PIL 库的图像预处理.
    """

    @staticmethod
    def resize_file(img, width, height):
        """
        重新调整一个 Image 的大小.
        """

        return img.resize((width, height), Image.BILINEAR)

    @staticmethod
    def single_channel(img):
        return img.convert("L")

    @staticmethod
    def convert_dir(in_dir, out_dir):
        """
        批量的处理一个目录下的所有图像.
        """

        if not os.path.exists(out_dir):
            os.system("mkdir -p " + out_dir)

        for img_file in tqdm.tqdm(os.listdir(in_dir)):
            img_path = os.path.join(in_dir, img_file)
            r_image = Image.open(img_path)

            c_image = ImageProcessor.resize_file(
                r_image, 32, 32
            ).convert("RGBA")
            out_path = os.path.join(out_dir, img_file)
            c_image.save(out_path.split(".")[0] + ".png")


class FileStream(object):
    """
    流批量 Spark Streaming 处理文件对象.
    """

    def __init__(self, sc):
        self._sc = sc

    @staticmethod
    def get_all_path(dir_list):
        path_list = []

        for directory in dir_list:
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                path_list.append(file_path)

        return path_list

    @staticmethod
    def bucket_segment(path_list, n_bucket):
        total_num = len(path_list)
        bucket_len = total_num // (n_bucket + 1) + 1

        segment_list = []
        for start in range(0, total_num, bucket_len):
            end = min(start + bucket_len, total_num)
            segment_list.append(path_list[start: end])

        return segment_list

    def stream_group(self, dir_list, out_dir, n_bucket):
        if not os.path.exists(out_dir):
            os.system("mkdir -p " + out_dir)

        # 对文件路径做切分.
        path_list = FileStream.get_all_path(dir_list)
        seg_path_list = FileStream.bucket_segment(path_list, n_bucket)

        # 初始化一个流式处理对象.
        ssc = StreamingContext(self._sc, 1)

        # 将路径列表压入队列.
        queue_rdd = ssc.queueStream(seg_path_list, oneAtATime=True)

        # RDD 的转换过程如下.
        pair_rdd = queue_rdd.map(
            lambda path: (re.split("[-.]", os.path.split(path)[-1])[0], path)
        )
        group_rdd = pair_rdd.groupByKey()

        def _dump_image(x):
            (index, p_list) = x
            w_dir = os.path.join(out_dir, str(index))

            if not os.path.exists(w_dir):
                os.system("mkdir -p " + w_dir)

            out_list = []
            for r_path in p_list:
                name = ("-".join(r_path.split("-")[1:])).replace("/", "$")
                w_path = os.path.join(w_dir, name)

                image = Image.open(r_path)
                image.save(w_path)

                out_list.append(w_path)

            return out_list

        group_rdd.map(_dump_image).pprint()

        ssc.start()
        time.sleep(40)
        ssc.stop(stopSparkContext=False)


if __name__ == '__main__':

    # 修改图片的格式, 包括尺寸, 通道.
    # ImageProcessor.convert_dir(
    #     "data/material/pokemon-a",
    #     "data/pokemon/pokemon-a"
    # )
    # ImageProcessor.convert_dir(
    #     "data/material/pokemon-b",
    #     "data/pokemon/pokemon-b"
    # )
    # ImageProcessor.convert_dir(
    #     "data/material/pokemon-tcg-images",
    #     "data/pokemon/pokemon-tcg-images"
    # )

    sc = SparkContext("local[1]", "Group Image")
    file_stream = FileStream(sc)

    # file_stream.stream_group(
    #     ["data/pokemon/pokemon-a", "data/pokemon/pokemon-b"],
    #     "data/train", n_bucket=16
    # )
    file_stream.stream_group(
        ["data/pokemon/pokemon-tcg-images"],
        "data/test", n_bucket=16
    )

