# 数据挖掘实验一

## 一、实验准备

1. 分词工具
&emsp;本次实验使用哈工大nlp团队制作的pyltp工具。本次实验主要用到分词，词性标注，以及命名实体识别功能。
<br>

2. 环境准备
    1. 首先从下方网站下载pyltp安装文件
https://mlln.cn/2018/01/31/pyltp%E5%9C%A8windows%E4%B8%8B%E7%9A%84%E7%BC%96%E8%AF%91%E5%AE%89%E8%A3%85/pyltp-0.2.1-cp36-cp36m-win_amd64.whl
    2. 运行`pip install pyltp-0.2.1-cp36-cp36m-win_amd64.whl`
    3. 从下方网站下载模型文件压缩包后解压。
http://ospm9rsnd.bkt.clouddn.com/model/ltp_data_v3.4.0.zip

&emsp;在本实验中需要Segmentor, Postagger, NamedEntityRecognizer三个模块，因此应在python文件中导入

```python
from pyltp import Segmentor, Postagger, NamedEntityRecognizer
```

## 二、 任务1—分词
1. 分词算法
   &emsp;传统的分词方法有基于词典的分词以及基于统计的分词（基于条件随机场模型）。
   &emsp;在ltp工具的官方文档中，引用了论文"Combining Statistical Model and Dictionary for Domain Adaption of Chinese Word Segmentation"其将传统的CRF分词与外部词典相结合，训练出了一款领域自适应性更好，F-measure指标更高的分词算法，其结构如下图所示。
   <div align=center><img src=pic/分词structure.jpg><br/>图1 分词网络结构</div>
   &emsp;如图所示，此模型将外部词典（包括领域词典和通用词典）所生成的特征与分词语料的特征进行融合，增强模型可以学习到的特征从而获得更好的分词方法。具体的特征融合方法请参考<sup>[1]</sup>。
   <br></br>
2. 利用预训练分词模型进行分词


   ## 参考文献
   [1]Meishan Zhang, Zhilong Deng，Wanxiang Che, and Ting Liu. Combining Statistical Model and Dictionary for Domain Adaption of Chinese Word Segmentation. Journal of Chinese Information Processing. 2012, 26 (2) : 8-12 (in Chinese)
