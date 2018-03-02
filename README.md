# image-retrieval-demo
tensorflow图像检索演示demo

输入query路径，获得与query相似图片

例子

import retrieve_demo_gpu

retrieve_demo=retrieve_demo_gpu.image_retrieval() #初始化，将检索库数据存入显存。由于一个计算图只能容纳2G，这里建了4个计算图，加上默认计算图，共5个计算图

retrieve_demo.retrieval(path='query的路径')#获得query与图片库所有图片余弦距离，排序得到余弦距离最近的n长图片
