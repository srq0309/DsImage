# 图片分类评估
# 提供一个公共计算图(定义ImageClassification类对象），方便单张或批量提取图片特征值

from Inception import inception_v3
from Inception import inception_preprocessing
from Inception import config

import pickle
import os
import tensorflow as tf

slim = tf.contrib.slim


class ImageClassification(object):
    """
    定义图片分类计算图，提供单张和批量提取图片特征值的接口
    """

    def __init__(self):
        # 创建全局会话
        self.sess = tf.Session()
        # 加载图片预分类标签
        with open(config.CLASS_LABEL, "rb") as f:
            self.names = pickle.load(f)

        # graph-1: 计算特征值与预分类标签
        # 输入数据（可传入任意数量图片数据）
        self.image_input = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
        # 定义inception-v3模型
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            _, end_points = inception_v3.inception_v3(self.image_input,
                                                      num_classes=1001,
                                                      is_training=False)
        self.preLogits, self.preDictions = end_points['PreLogits'], end_points['Predictions']

        # graph-2: 图片解码与预处理
        self.image_string = tf.placeholder(dtype=tf.string, shape=[])
        self.preClassArr = tf.image.decode_jpeg(self.image_string, channels=3)
        self.preCnnArr = inception_preprocessing.preprocess_image(self.preClassArr,
                                                                  299, 299, is_training=False)

        # 加载检查点
        init_fn = slim.assign_from_checkpoint_fn(config.CHECK_POINT,
                                                 slim.get_model_variables('InceptionV3'))
        init_fn(self.sess)

    def __del__(self):
        # 关闭会话
        self.sess.close()

    def __pre_class_feature(self, image_info, classArr):
        """
        提取图片基本信息：图片名称、大小、主色调
        :param image_info: 图片信息字典
        :param npArr: 图片矩阵（shape：255,255,255,3）
        :return: None
        """
        # 图片大小
        image_info["image_size"] = classArr.shape[:2]
        # 提取图片主色调（随机采样大约1032+个点）
        r, g, b = 0, 0, 0
        num = 0
        for i in range(0, classArr.shape[0], classArr.shape[0] // 32):
            for j in range(0, classArr.shape[1], classArr.shape[1] // 32):
                num += 1
                r += classArr[i][j][0]
                g += classArr[i][j][1]
                b += classArr[i][j][2]
        r = int(r / num)
        g = int(g / num)
        b = int(b / num)
        image_info["image_color"] = (r, g, b)

    def __pre_cnn_feature(self, image_info, cnnArr):
        """
        利用inception-v3提取特征值与图片标签
        :param image_info: 图片信息字典
        :param tfTensor:
        :return: None
        """
        # 特征向量 2048
        logits = self.sess.run(self.preLogits,
                               feed_dict={self.image_input: [cnnArr]})
        # 分类标签 1001
        dictions = self.sess.run(self.preDictions,
                                 feed_dict={self.image_input: [cnnArr]})
        # 特征向量
        image_info["image_feature"] = logits[0, 0, 0, 0:]
        # 图片top-5置信最高的分类
        sorted_inds = [i[0] for i in sorted(enumerate(-dictions[0, 0:]), key=lambda x: x[1])]
        __top5 = []
        for j in range(5):
            index = sorted_inds[j]
            __top5.append((index, dictions[0][index]))
        image_info["image_top5"] = __top5

    def __cal_graph(self, image_path_list):
        """
        批量处理图片，返回图片信息
        :param image_path_list: 图片路径列表
        :return: 图片信息列表
            图片信息组织形式为：
                image_info["image_name"]：图片文件名
                image_info["image_feature"]：图片特征向量（2048个feature）
                image_info["image_top5"]：图片top-5置信最高的分类标签
                image_info["image_color"]：图片主色调
                image_info["image_size"]：图片分辨率
        """
        # 包含所有图片信息字典的列表：
        image_info_list = []
        for image_path in image_path_list:
            # 读取图片
            with open(image_path, "rb") as f:
                image_string = f.read()
            # 计算图片原始矩阵和预处理后的矩阵
            classArr, cnnArr = self.sess.run([self.preClassArr, self.preCnnArr],
                                             feed_dict={self.image_string: image_string})
            # 图片信息字典
            image_info = {}
            # 名称
            image_info["image_name"] = os.path.basename(image_path)
            # 提取图片基本属性
            self.__pre_class_feature(image_info, classArr)
            # 利用inception-v3提取特征和预分类标签
            self.__pre_cnn_feature(image_info, cnnArr)
            image_info_list.append(image_info)
        return image_info_list

    def classification_one(self, image_path):
        """
        提取单张图片特征
        :param image_path: 图片路径
        :return: 图片信息
        """
        return self.__cal_graph([image_path])[0]

    def classification_batch(self, image_path_list):
        """
        批量提取图片特征
        :param image_path_list: 图片路径列表
        :param func: 每次处理完单张图像后的回调函数，一般打印信息等
        :return: 图片信息列表
        """
        return self.__cal_graph(image_path_list)


if __name__ == '__main__':
    iC = ImageClassification()
    image_info = iC.classification_one(
        r"C:\Users\Administrator\Code\Project\DsImage\data\ILSVRC2015\Data\DET\test\ILSVRC2015_test_00000001.JPEG")
    print(image_info)
