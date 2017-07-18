from Inception import ImageClassification
from Inception import ImageSearch
from multiprocessing.connection import Listener
import traceback

iC = ImageClassification.ImageClassification()
iS = ImageSearch.ImageSearch()


def echo_client(conn):
    try:
        while True:
            msg = conn.recv()
            if msg[0] == 1:
                image_list = iS.image_search_from_key(msg[1])
                print('根据标签搜索图片成功')
            elif msg[0] == 2:
                image_list = iS.image_search_from_image(iC.classification_one(msg[1]))
                print("根据已有搜索图片成功")
            else:  # 二次检索
                image_list = msg[1]
                if msg[0] == 3:  # 多图片共性
                    image_list = iS.quadratic_search_by_multiply_image(image_list, msg[2])
                elif msg[0] == 4:  # 颜色
                    image_list = iS.quadratic_search_by_color(image_list, msg[2])
                elif msg[0] == 5:  # 图片大小范围
                    image_list = iS.quadratic_search_by_size(image_list, msg[2])
                elif msg[0] == 6:
                    image_list = iS.quadratic_search_by_size(image_list, msg[3])
                    image_list = iS.quadratic_search_by_color(image_list, msg[2])
            conn.send(image_list)
    except EOFError:
        print('Connection closed')


def echo_server(address, authkey):
    serv = Listener(address, authkey=authkey)
    while True:
        try:
            client = serv.accept()
            echo_client(client)
        except Exception:
            traceback.print_exc()


if __name__ == '__main__':
    print("开启服务")
    echo_server(('', 25000), authkey=b'foo')
