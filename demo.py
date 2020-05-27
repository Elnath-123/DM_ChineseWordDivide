#coding:utf-8

import thulac as tl
import pandas as pd
import numpy as np
import os
from gensim.models import word2vec
import multiprocessing


def Divide():
    data_csv = pd.read_csv("data.csv")
    data = data_csv['title'].tolist()
    thu = tl.thulac(seg_only=True)  #设置模式为行分词模式
    f = open("divided_data.txt", "w", encoding='utf-8')

    for item in data:
        f.write(thu.cut(item, text=True) + '\n')

def train(sentences,  model_path, embedding_size = 128, window = 5, min_count = 5):
    model = word2vec.Word2Vec(sentences, sg=1, hs=1, sample=1e-3, size=embedding_size, 
                                window=window, min_count=min_count, workers=multiprocessing.cpu_count())
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save(os.path.join(model_path))
    return model 

def load(model_path):
    return word2vec.Word2Vec.load(model_path)
if __name__ == '__main__':
    corpus = []
    model_path = './model/word_model.model'
    model = load(model_path)

    sentences = word2vec.PathLineSentences("divided_data.txt")
    # print(sentences)
    word_1 = model.wv.vocab['中国']
    word_2 = model.wv.vocab['经济']
    print(word2vec.score_sg_pair(model,word_1,word_2))
    #train(sentences, model_path=model_path, embedding_size=128, window=5, min_count=5)
    
    #print(model.wv.similarity('亚洲', "公司"))
    
