# 定义一些全局变量
# 修改图片数据库，系统迁移需修改此文件部分变量

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHECK_POINT = os.path.join(BASE_DIR, "Packages/inception_v3.ckpt")
CLASS_LABEL = os.path.join(BASE_DIR, "Packages/class_label.pack")
# DATA_INDEX = r"C:\Users\Administrator\Code\Project\DsImage\data\ILSVRC2015\ImageSets\DET\test.pack"
DATA_FILE_PATH = r"C:\Users\srq0309\Code\Projects\DsImage\data\ILSVRC\Data\DET\test"
SQLITE_DB = os.path.join(BASE_DIR, "Packages/image_info.sqlite3")

if __name__ == '__main__':
    print(CHECK_POINT)
    print(CLASS_LABEL)
