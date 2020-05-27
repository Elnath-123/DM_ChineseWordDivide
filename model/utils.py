'''
@name utils.py
@author Jing Wang, Rongqing Li,     
        Yuting Ling, Qitan Shao
@date 5/26/2020
@description utils package for task 1,2
'''

class Utils:
    def __init__(self, entities=None):
        self.entities = entities

    def isnoun(self, tag):
        noun = ['n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'ws']
        if tag in noun:
            return True
        return False 

    def isnumber(self, tag):
        if tag == 'm':
            return True
        return False

    def isquantity(self, tag):
        if tag == 'q':
            return True
        return False

    def isadj(self, tag):
        if tag == 'a':
            return True
        return False
    
    def isner(self, tag):
        ner = ['B-Nh', 'B-Ni', 'B-Ns', 'I-Nh','I-Ni','I-Ns','E-Nh','E-Ni','E-Ns', 'S-Nh', 'S-Ns', 'S-Ni']
        if tag in ner:
            return True
        return False

    def concat(self, words, tag_postag, tag_ner, mode='netags'):
        concat_word = []
        if mode == 'postags':
            longest = words[0]
            for i in range(1, len(tag_postag)):
                #noun-noun 海上-天然气-项目
                if self.isnoun(tag_postag[i - 1]) and self.isnoun(tag_postag[i]):
                    longest += words[i]
                #adj-noun 美丽-风景
                elif self.isadj(tag_postag[i - 1]) and self.isnoun(tag_postag[i]):
                    longest += words[i]
                #number-quantity 一-个  一-家
                elif self.isnumber(tag_postag[i - 1]) and self.isquantity(tag_postag[i]): 
                    longest += words[i]
                #quantity-noun (一)家-公司
                elif self.isquantity(tag_postag[i - 1]) and self.isnoun(tag_postag[i]):
                    longest += words[i]
                else:
                    concat_word.append(longest)
                    if self.entities and longest not in self.entities.keys():
                        self.entities[longest] = [1, ''.join(words)]
                    else:
                        self.entities[longest][0] += 1
                    longest = words[i]
            concat_word.append(longest)
        elif mode == 'netags':
            longest = words[0]
            for i in range(1, len(tag_ner)):
                if self.isnoun(tag_postag[i]):
                    longest += words[i] 
                elif self.isner(tag_ner[i]) and self.isner(tag_ner[i - 1]):
                    longest += words[i]
                else:
                    concat_word.append(longest)
                    if self.entities != None and longest not in self.entities.keys():
                        self.entities[longest] = [1, ''.join(words)]
                    else:
                        self.entities[longest][0] += 1
                    longest = words[i]
            concat_word.append(longest) 
        return ' '.join(concat_word)


if __name__ == '__main__':
    from pyltp import Segmentor
    from pyltp import Postagger
    from pyltp import NamedEntityRecognizer
    utils = Utils()
    segmentor = Segmentor()
    postagger = Postagger()
    recognizer = NamedEntityRecognizer()
    segmentor.load('cws.model')
    postagger.load('pos.model')
    recognizer.load('ner.model')
    word = input()
    seg_result = ' '.join(segmentor.segment(word)).split(' ')
    print(seg_result)
    pos_result = ' '.join(postagger.postag(seg_result)).split(' ')
    print(pos_result)
    ner_result = ' '.join(recognizer.recognize(seg_result, pos_result)).split(' ')
    print(ner_result)
    