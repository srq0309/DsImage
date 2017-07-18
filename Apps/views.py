from django.shortcuts import render
from multiprocessing.connection import Client
import pickle

image_tmp_path = r"C:\Users\srq0309\Code\Projects\DsImage\src\DsImage\Tmp\tmp.jpg"
image_list_path = r"C:\Users\srq0309\Code\Projects\DsImage\src\DsImage\Tmp\image_list.pack"


# Create your views here.

def index(request):
    """
    初始页面
    """
    return render(request, "index.html")


def search_keyword(request):
    """
    基于文本检索
    """
    try:
        key_word = request.POST['keyword']
    except:
        return render(request, "index.html", {'key': "None"})
    else:
        c = Client(('localhost', 25000), authkey=b'foo')
        c.send((1, key_word))
        image_list = c.recv()
        c.close()
        with open(image_list_path, "wb") as f:
            pickle.dump(image_list, f)
        return render(request, "index.html", {'image_list': image_list})


def search_file(request):
    """
    基于内容检索（以图搜图）
    """
    if request.method == 'POST':
        file = request.FILES.get("file", None)
        if file:
            f = request.FILES['file']
            with open(image_tmp_path, "wb") as image:
                image.write(f.read())
            image.close()
            # classification = classification2.Classification()
            c = Client(('localhost', 25000), authkey=b'foo')
            c.send((2, image_tmp_path))
            image_list = c.recv()
            image_tmp_list = image_list
            c.close()
            with open(image_list_path, "wb") as f:
                pickle.dump(image_list, f)
            return render(request, 'index.html', {'image_list': image_list})
        else:
            return render(request, 'index.html', {'key': "请选择正确的文件"})


def quadratic_search(request):
    """
    二次检索
    """
    try:
        images = request.POST.getlist("images")
        image_list = request.POST.getlist("image_list")
    except:
        print("err in quadratic_search")
    else:
        if len(images) == 0:
            return render(request, "index.html", {"key": "请先进行图片或问题检索并且勾选心仪图片之后在进行二次检索"})
        c = Client(('localhost', 25000), authkey=b'foo')
        c.send((3, image_list, images))
        image_list2 = c.recv()
        with open(image_list_path, "wb") as f:
            pickle.dump(image_list2, f)
        c.close()
        return render(request, "index.html", {'image_list': image_list2})


def search_filter(request):
    """
    条件筛选
    """
    try:
        color = request.POST["color"]
        low = request.POST["low"]
        high = request.POST["high"]
        tips = 0
        if color != "NO":
            color = eval(color)
            if low != "" and high != "":
                tips = 6
                low = eval(low)
                high = eval(high)
            else:
                tips = 4
        else:
            if low == "" or high == "":
                return render(request, "index.html", {'key': "请输入正确的筛选条件!"})
            tips = 5
            low = eval(low)
            high = eval(high)
    except:
        return render(request, "index.html", {'key': "请输入正确的筛选条件!"})
    else:
        try:
            with open(image_list_path, "rb") as f:
                image_list = pickle.load(f)
        except:
            return render(request, "index.html", {"key": "请先进行图片或问题检索再进行二次筛选"})
        else:
            c = Client(('localhost', 25000), authkey=b'foo')
            if tips == 4:
                c.send((4, image_list, color))
            elif tips == 5:
                c.send((5, image_list, (low, high)))
            elif tips == 6:
                c.send((6, image_list, color, (low, high)))
            else:
                print("err")
            image_list = c.recv()
            c.close()
            with open(image_list_path, "wb") as f:
                pickle.dump(image_list, f)
            return render(request, "index.html", {'image_list': image_list})
