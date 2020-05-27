'''
@author Rongqing Li,     
@date 5/27/2020
@description Codes for writing documents(splited to many parts)
'''
# 导入Segmentor, pandas模块
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
import pandas as pd
import random
from utils import Utils
random_select_number = 600
# Task1-分词
def split_data():
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
    segmentor.release()

# Task1-随机抽样600条结果
def random_select():
    file_name = '../data_split.csv'
    with open(file_name, 'r', encoding='utf-8-sig') as data_split:
        datas = data_split.readlines()
        datas = [data.strip() for data in datas]
        random.shuffle(datas)
        random_select_file = open('random_select.txt', 'w', encoding='utf-8-sig')
        for i in range(random_select_number):
            print("{0}:{1}".format(i, datas[i]), file=random_select_file)
            if i != 0 and i % 200 == 0:
                print(file=random_select_file)

# Task2-词性标注方法
def postag_data():
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
    data_processed = open('../data_processed_postagger.csv', 'w', encoding='utf-8')
    for data in datas:
        words = segmentor.segment(data)  # 分词
        postags = postagger.postag(words) # 标注
        word_split = ' '.join(words).split(' ') 
        postags_split = ' '.join(postags).split(' ')
        # 连接词语
        concat_word = util.concat(word_split, postags_split, type='postags')
        data_processed.write(concat_word + '\n')
    data_processed.close()

# Task2-命名实体识别方法
def ner_data():
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

if __name__ == '__main__':
    # split_data
    # random_select()
    # postag_data