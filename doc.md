# 数据挖掘实验一

## 一、实验准备

1. 分词工具
&emsp;&emsp;本次实验使用哈工大nlp团队制作的pyltp工具。本次实验主要用到分词，词性标注，以及命名实体识别功能。
<br>

2. 环境准备
    1. 首先从下方网站下载pyltp安装文件
https://mlln.cn/2018/01/31/pyltp%E5%9C%A8windows%E4%B8%8B%E7%9A%84%E7%BC%96%E8%AF%91%E5%AE%89%E8%A3%85/pyltp-0.2.1-cp36-cp36m-win_amd64.whl
    2. 运行`pip install pyltp-0.2.1-cp36-cp36m-win_amd64.whl`
    3. 从下方网站下载模型文件压缩包后解压。
http://ospm9rsnd.bkt.clouddn.com/model/ltp_data_v3.4.0.zip

&emsp;&emsp;在本实验中需要Segmentor, Postagger, NamedEntityRecognizer三个模块，因此应在python文件中导入

```python
from pyltp import Segmentor, Postagger, NamedEntityRecognizer
```

## 二、 任务1—分词
1. LTP分词算法
   &emsp;&emsp;传统的分词方法有基于词典的分词以及基于统计的分词（基于条件随机场模型）。
   &emsp;&emsp;在ltp工具的官方文档中，引用了论文"Combining Statistical Model and Dictionary for Domain Adaption of Chinese Word Segmentation"其将传统的CRF分词与外部词典相结合，训练出了一款领域自适应性更好，F-measure指标更高的分词算法，其结构如下图所示。
   <div align=center><img src=pic/分词structure.jpg><br/>图1 分词网络结构</div>
   &emsp;&emsp;如图所示，此模型将外部词典（包括领域词典和通用词典）所生成的特征与分词语料的特征进行融合，增强模型可以学习到的特征从而获得更好的分词方法。具体的特征融合方法请参考<sup>[1]</sup>。
   <br></br>
2. 利用预训练分词模型进行分词
    1. 分词程序
        ```python
        # 导入Segmentor, pandas模块
        from pyltp import Segmentor
        import pandas as pd
        # 加载分词模型
        segmentor = Segmentor()
        segmentor.load('cws.model') 
        # 加载将要被分词的数据
        data_csv = pd.read_csv('../data.csv', encoding='utf-8-sig')
        # 选取'title'列
        datas = data_csv['title']
        # 打开新的文件，存放分词结果
        data_split = open('../data_split.csv', 'w', encoding='utf-8-sig')
        for data in datas:
            # 分词后结果写入文件
            words = segmentor.segment(data)
            data_split.write(' '.join(words) + '\n')
        ```
        分词结果存放在\$DM_EXPERIMENT\$/DivedeData/data.csv文件中
    2. 分词结果评价
        由于没有提供分词结果的Ground Truth， 因此对于分词测试的结果的评估只能以人工的方式进行评价，同样，以人工方式评价时，相关指标也会有人为因素的干扰，因此在这里不考虑统计准确度等指标，只考虑人为的评价。我们在测试语料中，随机抽样600条分词结果。

## 三、 任务2—寻找最长名词实体
&emsp;&emsp;本任务我们计划使用两种不同的方式来寻找最长的名词实体，一种是通过**词性标注(Part-of-speech Tagging) **，将连续的，相近词性的词语进行合并；另一种是通过**命名实体检测(Named Entity Recognition)** ，将词语进行合并。
1. 词性标注方法
&emsp;&emsp;词性标注方法是一种通过将分词结果进行词性标注，再将相近词性的词语合并，以找到最长名词实体的方法。
    a. LTP词性标注算法
 LTP工具的词性标注方法在2011年发表在ACL上的一篇叫做"Joint Models for Chinese POS Tagging and Dependency Parsing"[2]的论文中提及。其主要创新是将 词性标注(POS Tagging) 与 依存句法分析(Dependency Parsing) 任务合并，最终的实验结果在这两个任务中都有一定的提升。此外，还提供了一种对词性标注空间剪枝的方法，提高分析的速度。具体模型请参考[2]
    b. LTP词性标注程序
    ```python
        from utils import Utils
    from pyltp import Segmentor
    from pyltp import Postagger
    # 分词模型
    segmentor = Segmentor()
    segmentor.load('cws.model')
    # 词性标注模型
    postagger = Postagger()
    postagger.load('pos.model')

    # 加载将要被分词的数据
    data_csv = pd.read_csv('../data.csv', encoding='utf-8-sig')
    datas = data_csv['title']

    util = Utils()
    data_processed = open('./data_processed_postagger.csv', 'w', encoding='utf-8')
    for data in datas:
        words = segmentor.segment(data)  # 分词
        postags = postagger.postag(words) # 标注
        word_split = ' '.join(words).split(' ') 
        postags_split = ' '.join(postags).split(' ')
        # 连接词语
        concat_word = util.concat(word_split, postags_split, type='postags')
        data_processed.write(concat_word + '\n')
    data_processed.close()
    ```
    连接词语使用utils.py中的concat函数，具体实现参见`$DM_EXPERIMENT$/DivideData/model/utils.py`
    词性标注并合并的后的结果存放在`$DM_EXPERIMENT$/DivideData/model/ data_processed_postagger.csv`

2. 命名实体识别方法
命名实体识别又称作“专名识别”，它可以识别文本中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等。
    a. 命名实体识别代码
    ```python
     # 分词模型
    segmentor = Segmentor()
    segmentor.load('cws.model')
    # 词性标注模型
    postagger = Postagger()
    postagger.load('pos.model')
    # 命名实体模型
    recognizer = NamedEntityRecognizer()
    NamedEntityRecognizer.load('ner.model')
    # 加载将要被分词的数据
    data_csv = pd.read_csv('../data.csv', encoding='utf-8-sig')
    datas = data_csv['title']

    util = Utils()
    data_processed = open('./data_processed_recognizer.csv', 'w', encoding='utf-8')
    for data in datas:
        words = segmentor.segment(data)
        postags = postagger.postag(words)
        word_split = ' '.join(words).split(' ')
        netags = recognizer.recognize(words, postags)
        netag_split = ' '.join(netags).split(' ')
        concat_word = util.concat(word_split, netag_split, tag='netags')
        data_processed.write(concat_word + '\n')
    data_processed.close()
    ```
    连接词语使用utils.py中的concat函数，具体实现参见`$DM_EXPERIMENT$/DivideData/model/utils.py`
    词性标注并合并的后的结果存放在`$DM_EXPERIMENT$/DivideData/model/ data_processed_recognizer.csv`

3. 方法对比与评价
    事实上，命名实体标注(ner)是基于分词和词性标注基础上的任务，它使用了这两个结果来检测命名实体。因此这两种方法的效果相差并不多。
    在总体效果上，命名实体标注的方法较词性标注更好一些。例如，下面这句分词后的结果：

    ```
    分词后结果：“汇丰 控股 集团 主席 葛霖 ： 没有 在 印度 上市 的 计划”

    词性标注结果：['nz', 'v', 'n', 'n', 'nh', 'wp', 'v', 'p', 'ns', 'v', 'u', 'n']
    词性标注方法：“汇丰 控股 集团主席葛霖 ： 没有 在 印度 上市 的 计划”

    命名实体标注结果：['B-Ni', 'I-Ni', 'E-Ni', 'O', 'S-Nh', 'O', 'O', 'O', 'S-Ns', 'O', 'O', 'O']
    命名实体标注方法：“汇丰控股集团主席葛霖 ： 没有 在 印度 上市 的 计划”
    ```
    &emsp;&emsp;可见，由于词性标注将“控股”一次标注为了"v"，即动词，因此我们编写的词性标注方法不可能将动词和其它词语连接起来，导致无法找到更长的名词实体。相反将分词结果和词性标注结果送入ner模型后，即使“控股”为动词，也可以通过上下文来预测出“控股”属于“Ni”类型，即机构名。
    &emsp;&emsp;实际上，我们所做的只是利用命名实体识别或者词性标注的结果，将分词数据合并了起来。而命名实体识别可以看作是在词性标注上多做了一步，因此我们使用命名实体识别的结果可以使得识别结果更加具有鲁棒性。
## 四、文件说明
&emsp;&emsp;DivideData目录中，data.xls为原始数据，data.csv为原始数据的csv版本，data_split.csv为分词后数据，可运行model/task.py，调用split_data函数获得。
&emsp;&emsp;DivideData/model目录中，\*.model文件为pyltp的预训练模型文件；task.py包括任务一、二的代码，可直接运行对应函数获得(寻找最长名词实体的)结果；split.py文件是完整的程序文件，可直接处理raw data获得最长名词实体合并后的结果(data_processed_\*.csv)以及作业要求的实体名词个数统计结果(target_\*.csv)。
&emsp;&emsp;split.py支持两种方法，调用`python ./split.py --method=postagger`可使用词性标注方法获得结果，调用`python ./split.py --method=recognizer`可使用命名实体识别方法获得结果。
&emsp;&emsp;utils.py为工具包，用来连接词语。

## 参考文献
[1]Meishan Zhang, Zhilong Deng，Wanxiang Che, and Ting Liu. Combining Statistical Model and Dictionary for Domain Adaption of Chinese Word Segmentation. Journal of Chinese Information Processing. 2012, 26 (2) : 8-12 (in Chinese)
[2]Zhenghua Li, Min Zhang, Wanxiang Che, Ting Liu, Wenliang Chen, and Haizhou Li. Joint Models for Chinese POS Tagging and Dependency Parsing. In Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing (EMNLP 2011). 2011.07, pp. 1180-1191. Edinburgh, Scotland, UK.