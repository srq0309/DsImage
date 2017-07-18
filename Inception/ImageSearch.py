# 图片搜索评估
# 提供一个搜索类，提供不同的搜索方式接口

from Inception import config

import os
import pickle
import sqlite3
import tensorflow as tf


def __create_db():
    # 创建数据库
    conn = sqlite3.connect(config.SQLITE_DB)
    conn.execute('''CREATE TABLE image_info(
    image_name vchar(20) PRIMARY KEY,
    image_label_No1 INT,
    image_label_No2 INT,
    image_label_No3 INT,
    image_feature BLOB,
    image_size BLOB,
    image_color BLOB
    )''')
    conn.commit()


def make_db(image_path_list):
    """
    批量处理图片库，并将信息保存在数据库中
    :param image_path_list: 图片路径列表
    数据库为ImageNet2015新增测试图片11142张(ILSVRC2015_test_000?????.JPEG)
    """
    from Inception.ImageClassification import ImageClassification

    # __create_db()  # 创建数据库，第一次运行
    conn = sqlite3.connect(config.SQLITE_DB)
    iC = ImageClassification()
    for image_path in image_path_list:
        image_info = iC.classification_one(image_path)
        conn.execute("INSERT INTO image_info VALUES (?,?,?,?,?,?,?)",
                     (image_info['image_name'],
                      image_info['image_top5'][0][0],
                      image_info['image_top5'][1][0],
                      image_info['image_top5'][2][0],
                      pickle.dumps(image_info['image_feature']),
                      pickle.dumps(image_info['image_size']),
                      pickle.dumps(image_info['image_color']),
                      ))
        conn.commit()
        print("{} complete !".format(image_path))


class ImageSearch(object):
    """
    定义图片搜索计算图，提供常用搜索接口
    此类严重依赖于预构建的图片信息数据库
    """

    def __init__(self):
        # 创建全局会话
        self.sess = tf.Session()
        # 加载图片预分类标签
        with open(config.CLASS_LABEL, "rb") as f:
            self.names = pickle.load(f)
        # 数据库连接
        self.conn = sqlite3.connect(config.SQLITE_DB)

        # 构建特征向量相似度计算图
        self.input_feature = tf.placeholder(tf.float32, shape=[None])
        # 修改为不固定特征向量长度
        # 以便于公共特征匹配时可利用此计算图
        # 2017-05-15
        self.image_feature = tf.placeholder(tf.float32, shape=[None])
        self.dis = tf.reduce_sum(tf.square(self.input_feature - self.image_feature))  # 欧式距离

        # 构建二次检索计算图（多图片共性计算）
        self.common = {}
        self.multiply_input = tf.placeholder(tf.float32, shape=[None, 2048])
        self.common["reduce_mean"] = tf.reduce_mean(self.multiply_input, 0)  # 均值
        # 特征向量方差，由此表示每种特征向量在这几幅图片中离散程度，离散度越低，说明此特征为共有特征的可能性越高
        self.common["th_sum"] = tf.reduce_sum(tf.square(self.multiply_input - self.common["reduce_mean"]), 0)

    def image_search_from_image(self, image_info):
        """
        搜索相似图片
        :param image_info: 图片信息
            具体参考classification2.py line78: Classification.__classification_one()
        :return: 按照相似度排序后的列表
        """
        # setp 1：筛选标签关联的图片出来 3*3=9种情况
        top5 = image_info["image_top5"]
        sql = "SELECT image_name, image_feature FROM image_info WHERE image_label_No1={0} OR image_label_No2={0} OR image_label_No3={0}"
        res_tmp_list = []  # 避免性能损耗，不进行列表整合，图片信息分为三部分存储在这个临时二维列表
        for i in range(3):
            res_tmp_list.append(self.conn.execute(sql.format(top5[1][0])).fetchall())

        # step 2：根据计算得到特征向量空间中的几何距离，并排序
        feature_d = {}  # 存储几何距离的字典，在计算前利用其去重
        for res in res_tmp_list:
            for image in res:
                if image[0] not in feature_d:
                    features = pickle.loads(image[1])
                    d = self.sess.run(self.dis,
                                      feed_dict={self.input_feature: features,
                                                 self.image_feature: image_info["image_feature"]})
                    feature_d[image[0]] = d
        image_path_list = [i[0] for i in sorted(
            list(feature_d.items()), key=lambda x: x[1])]
        return image_path_list

    def image_search_from_key(self, key_word):
        """
        搜索指定标签搜索，预分类标签，没有实现模糊搜索
        :param key_word: 标签名称
        :return: 搜索结果列表
        """
        labels = []
        for label, name in self.names.items():
            for word in name.split(", "):
                if word.startswith(key_word) or word.endswith(' ' + key_word):
                    labels.append(label)
                    break
        image_path_list = []
        sql = "SELECT image_name, image_feature FROM image_info WHERE image_label_No1={}"
        for label in labels:
            res = self.conn.execute(sql.format(label))
            for image in [i[0] for i in res]:
                if image not in image_path_list:
                    image_path_list.append(image)
        return image_path_list

    def quadratic_search_by_multiply_image(self, image_list, image_name_list):
        """
        通过多张图片进行二次检索，依据共性计算相似度，排序后返回
        :param image_list: 二次检索前的列表
        :param image_name_list: 提供的图片
        """
        image_feature_list = []  # 图片特征列表，与image_name_list一一对应
        for image_name in image_name_list:
            res = self.conn.execute("SELECT image_feature FROM image_info WHERE image_name='{}'".format(
                image_name)).fetchall()[0][0]
            feature = pickle.loads(res)
            image_feature_list.append(feature)
        common = self.sess.run(self.common, feed_dict={
            self.multiply_input: image_feature_list})
        # 每种特征的均值和方差
        reduce_mean, th_features = common["reduce_mean"], common["th_sum"]
        # 排序后的结果
        sorted_inds = [i[0] for i in sorted(enumerate(th_features), key=lambda x: x[1])]
        N = 50  # 选取的属于共性特征的特征数目
        features = []
        for i in range(N):
            features.append(reduce_mean[sorted_inds[i]])

        feature_d = []  # 特征向量相似度数值，与image_list一一对应
        for image in image_list:
            res = self.conn.execute(
                "SELECT image_feature FROM image_info WHERE image_name='{}'".format(image)).fetchall()
            pre_feature = pickle.loads(res[0][0])
            fin_feature = []
            for i in range(N):
                fin_feature.append(pre_feature[sorted_inds[i]])
            d = self.sess.run(self.dis,
                              feed_dict={self.input_feature: fin_feature,
                                         self.image_feature: features})
            feature_d.append(d)
        sorted_inds_image_path = [i[0] for i in sorted(enumerate(feature_d), key=lambda x: x[1])]
        image_path = []  # 图片名称，按相似度大小排序
        for index in sorted_inds_image_path:
            image_path.append(image_list[index])
        return image_path

    def quadratic_search_by_color(self, image_list, image_color):
        """
        通过颜色进行二次检索，根据主色调差异排序后返回
        :param image_list: 二次检索前的列表
        :param image_color: 图片颜色
        :return: 最终排序结果
        """
        image_name_list = []  # 保存名称和颜色
        final_res = []  # 最终结果
        for image in image_list:
            res = self.conn.execute(
                "SELECT image_color FROM image_info WHERE image_name='{}'".format(image)).fetchall()
            color = pickle.loads(res[0][0])
            image_name_list.append((image, color))
        for name, color in image_name_list:
            d = self.sess.run(self.dis,
                              feed_dict={self.input_feature: image_color,
                                         self.image_feature: color})
            final_res.append((name, d))
        return [i[0] for i in sorted(final_res, key=lambda x: x[1])]

    def quadratic_search_by_size(self, image_list, image_sizes):
        """
        通过图片大小进行二次检索，根据大小范围筛选图片
        :param image_list: 二次检索前的列表
        :param image_sizes: 图片大小范围——((x1,y1),(x2,y2))
        :return: 排序结果
        """
        image_name_list = []  # 保存名称和大小
        final_res = []  # 最终结果
        for image in image_list:
            res = self.conn.execute(
                "SELECT image_size FROM image_info WHERE image_name='{}'".format(image)).fetchall()
            size = pickle.loads(res[0][0])
            image_name_list.append((image, size))
        for image, size in image_name_list:
            if (size[1] > image_sizes[0][0] and size[1] < image_sizes[1][0] and
                        size[0] > image_sizes[0][1] and size[0] < image_sizes[1][1]):
                final_res.append(image)
        return final_res


if __name__ == '__main__':
    image_list = []  # 保存图片路径
    # 第一次构建取消line-35的注释
    make_db(image_list)
