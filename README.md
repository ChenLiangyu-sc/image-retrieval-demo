tensorflow图像检索演示demo

运行run_main.py 启动程序，等待将图片特征读入GPU，便可输入需要检索query路径，将返回与query图片距离最近的200张图片的子目录和图片名。

count_dist.py文件为图片的余弦距离的计算部分。

inference.py文件为vggs网络，为网络前传部分。

