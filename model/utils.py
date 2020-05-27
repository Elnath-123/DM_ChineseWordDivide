
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

    def concat(self, words, tags, tag='netags'):
        concat_word = []
        if tag == 'postags':
            longest = words[0]
            for i in range(1, len(tags)):
                #noun-noun 海上-天然气-项目
                if self.isnoun(tags[i - 1]) and self.isnoun(tags[i]):
                    longest += words[i]
                #adj-noun 美丽-风景
                elif self.isadj(tags[i - 1]) and self.isnoun(tags[i]):
                    longest += words[i]
                #number-quantity 一-个  一-家
                elif self.isnumber(tags[i - 1]) and self.isquantity(tags[i]): 
                    longest += words[i]
                #quantity-noun (一)家-公司
                elif self.isquantity(tags[i - 1]) and self.isnoun(tags[i]):
                    longest += words[i]
                else:
                    concat_word.append(longest)
                    if self.entities and longest not in self.entities.keys():
                        self.entities[longest] = [1, ''.join(words)]
                    else:
                        self.entities[longest][0] += 1
                    longest = words[i]
            concat_word.append(longest)
        elif tag == 'netags':
            longest = words[0]
            for i in range(1, len(tags)):

                if self.isner(tags[i]) and self.isner(tags[i - 1]):
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
    utils = Utils()
    segmentor = Segmentor()
    postagger = Postagger()
    segmentor.load('cws.model')
    postagger.load('pos.model')
    word = input()
    seg_result = ' '.join(segmentor.segment(word)).split(' ')
    print(seg_result)
    pos_result = ' '.join(postagger.postag(seg_result)).split(' ')
    print(pos_result)
    