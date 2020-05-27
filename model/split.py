# coding=utf-8
'''
@name split.py
@author Jing Wang, Rongqing Li,     
        Yuting Ling, Qitan Shao
@date 5/26/2020
@description Complete procedure of task 1,2
'''
import os
import pandas as pd
import csv
import argparse
from utils import Utils
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer

parser = argparse.ArgumentParser(description='Pyltp for datamining experiment')
parser.add_argument('--method', help='Set the method of combining entities', default='postaget', type=str)
args = parser.parse_args()


def PostagResult(datas, postagger, segmentor):
    entities = dict()
    util = Utils(entities)
    
    data_processed = open('./data_processed_postagger.csv', 'w', encoding='utf-8')
    for data in datas:
        words = segmentor.segment(data)  # divide 
        postags = postagger.postag(words) # pos tagging
        word_split = ' '.join(words).split(' ')
        postags_split = ' '.join(postags).split(' ')
        concat_word = util.concat(word_split, postags_split, tag_ner=None, mode='postags')
        data_processed.write(concat_word + '\n')
    entities = util.entities
    data_processed.close()
    return entities

def NameEntityResult(datas, postagger, segmentor, recognizer):
    entities = dict()
    util = Utils(entities)
    data_processed = open('./data_processed_recognizer.csv', 'w', encoding='utf-8')
    for data in datas:
        words = segmentor.segment(data)
        postags = postagger.postag(words)
        postags_split = ' '.join(postags).split(' ')
        word_split = ' '.join(words).split(' ')
        netags = recognizer.recognize(words, postags)
        netag_split = ' '.join(netags).split(' ')
        concat_word = util.concat(word_split, postags_split, netag_split, mode='netags')
        data_processed.write(concat_word + '\n')
    entities = util.entities
    data_processed.close()
    return entities


def dump_result_to_csv(target):
    writer = csv.writer(target, delimiter=',')
    writer.writerow(['entity', 'times', 'length', 'source'])
    for k, v in sorted_entities:
        writer.writerow([k, v[0], len(k), v[1]])

if __name__ == '__main__':
    segmentor = Segmentor()  # Initialize model
    postagger = Postagger()
    recognizer = NamedEntityRecognizer()
    segmentor.load('cws.model')  # Load model
    postagger.load('pos.model')
    recognizer.load('ner.model')

    # Load raw data
    data_csv = pd.read_csv('../data.csv')

    # Fetch column 'title'
    datas = data_csv['title']

    # Run in different method according to arg.method
    if args.method == 'postagger':
        entities = PostagResult(datas, postagger, segmentor)
        target_file = open('target_postagger.csv', 'w', encoding='utf-8-sig')
    elif args.method == 'recognizer':
        entities = NameEntityResult(datas, postagger,segmentor, recognizer)
        target_file = open('target_recognizer.csv', 'w', encoding='utf-8-sig')
    else:
        print("Invalid method!")
        exit(0)

    postagger.release()  # Release model
    segmentor.release()  # Release model

    # Write final result to csv file
    sorted_entities = sorted(entities.items(), key=lambda t:len(t[0]), reverse=True)
    dump_result_to_csv(target_file)   