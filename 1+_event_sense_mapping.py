# from nltk.corpus import wordnet as wn

# x = wn.synsets("dog")[-1]
# y = wn.synsets("chase")[-1]
# d = x.shortest_path_distance(y,simulate_root=True)
# print(d)
# def wordnet_staticis():
#     map_sk2syn = {}
#     for synset in wn.all_synsets():
#         for lemma in synset.lemmas():
#             map_sk2syn[lemma.key()] = synset.name()
#     num_synset = []
#     num_lemma = []
#     for key, value in map_sk2syn.items():
#         num_synset.append(value)
#         num_lemma.append(key)
#     num_synset = set(num_synset)
#     num_lemma = set(num_lemma)
#     len(map_sk2syn)
import os
import numpy as np
from transformers_encoder import TransformersEncoder
from vectorspace import SensesVSM
from wn_utils import WN_Utils
from collections import Counter
from logging.handlers import SMTPHandler

import numpy as np
import traceback
import pickle
import logging
import sys
import spacy
import json
base_path = "../SimCSE-main/LDCCorpus/gigaword_eng_5/data/nyt_online_4"

#使用指令 python 1+_event_sense_mapping.py 1994 1995 1996

class disambiguation(object):
    def __init__(self) -> None:
        super().__init__()
        self.event_sense_mapping = {}
        self.en_nlp = spacy.load('/home/anaconda3/envs/LMMS/lib/python3.6/site-packages/en_core_web_sm/en_core_web_sm-1.2.0')  # required for lemmatization and POS-tagging
        print("Spacy load successfully!")


        # with open('./results_save/wsd_encoder.pkl', "rb") as wsd:
        #     self.wsd_encoder = pickle.load(wsd)
        # with open('./results_save/senses_vsm.pkl', "rb") as senses:
        #     self.senses_vsm = pickle.load(senses) 
        self.wn_utils = WN_Utils()  # WordNet auxilliary methods (just for describing results)
    
        # # NLM/LMMS paths and parameters
        vecs_path = './data/vectors/lmms-sp-wsd.albert-xxlarge-v2.vectors.txt'

        wsd_encoder_cfg = {
            'model_name_or_path': 'albert-xxlarge-v2',
            'min_seq_len': 0,
            'max_seq_len': 512,
            'layers': [-n for n in range(1, 12 + 1)],  # all layers, with reversed indices
            'layer_op': 'ws',
            'weights_path': 'data/weights/lmms-sp-wsd.albert-xxlarge-v2.weights.txt',
            'subword_op': 'mean'
        }
        print('Loading NLM and sense embeddings ...')  # (takes a while)
        self.wsd_encoder = TransformersEncoder(wsd_encoder_cfg)
        self.senses_vsm = SensesVSM(vecs_path, normalize=True)
        # with open('./results_save/wsd_encoder.pkl', "wb") as wsd: 
        #     pickle.dump(self.wsd_encoder, wsd)
        # with open('./results_save/senses_vsm.pkl', "wb") as senses: 
        #     pickle.dump(self.senses_vsm, senses)
        print('Done')

    def LMMS(self, sub_doc, ctx_embeddings, event_target_idxs):

        
        for current_event, target_idxs in event_target_idxs.items():
            syn_the_best_match = {}
            for target_idx in target_idxs:
                target_pos = sub_doc[target_idx].pos_
                target_lemma = '_'.join([sub_doc[i].lemma_ for i in [target_idx]])
                target_embedding = np.array([ctx_embeddings[i][1] for i in [target_idx]]).mean(axis=0)
                target_embedding = target_embedding / np.linalg.norm(target_embedding)
                matches = self.senses_vsm.match_senses(target_embedding, lemma=target_lemma, postag=target_pos, topn=1)
                if matches!= []:
                    sk, sim = matches[0]
                    the_best_match = self.wn_utils.sk2syn(sk)
                    the_best_sim = sim
                    # syn_the_best_match.append({sub_doc[target_idx]: [the_best_match, the_best_sim]})
                    syn_the_best_match[sub_doc[target_idx].text] = [the_best_match._name, np.float(the_best_sim)] #sub_doc[target_idx]是spacy的token类
                    #np.float不用，会导致json.dump时报错 --> TypeError: Object of type 'float32' is not JSON serializable解决方案
                else:
                    continue
        
            if current_event not in self.event_sense_mapping.keys():
                self.event_sense_mapping[current_event] = syn_the_best_match
            elif self.event_sense_mapping[current_event] == syn_the_best_match:
                logger.warning( '{} is saved before. No Changed'.format(current_event) )
                continue
            else:
                logger.warning( '{} is saved before. Former: {}'.format(current_event, self.event_sense_mapping[current_event]) )
                arg_key_a = [*self.event_sense_mapping[current_event].keys()]
                arg_value_a = [*self.event_sense_mapping[current_event].values()]
                arg_key_b = [*syn_the_best_match.keys()]
                arg_value_b = [*syn_the_best_match.values()]
         
                for index_b, akb in enumerate(arg_key_b):
                    try:
                        index_a = arg_key_a.index(akb)
                        if arg_value_a[index_a][1] < arg_value_b[index_b][1]:
                            arg_value_a[index_a] = arg_value_b[index_b]
                    except:
                        arg_key_a.insert(index_b, akb)
                        arg_value_a.insert(index_b, arg_value_b[index_b])
                
                self.event_sense_mapping[current_event] = dict(zip(arg_key_a, arg_value_a))
                logger.warning('{} is saved before. Latter: {}'.format(current_event, self.event_sense_mapping[current_event]) )
                    
        
        
    def process_sentence(self, sentence, all_events):
        sub_doc = self.en_nlp(sentence)
        tokens = [t.text for t in sub_doc]
        ctx_embeddings = self.wsd_encoder.token_embeddings([tokens])
        if ctx_embeddings == None:
            return 
        else:
            ctx_embeddings = ctx_embeddings[0]

        words = []
        for word in sub_doc:
            words.append(word.text)
        word_result = Counter(words)
        event_target_idxs = {} #一个sentence对应多个event，需要筛选出一部分event出来。

        for event in all_events:
            arg_count_mapping = [{arg: word_result[arg]} for arg in event]
            arg_count = []
            # for a in arg_count_mapping:
            #     for _, v in a.items():
            #         arg_count += v   [{'Sirakov': 1}, {'put in': 0}, {'minute': 2}]导致利用arg_count = 3判断出错。
            for acm in arg_count_mapping:
                for _, v in acm.items():
                    arg_count.append(v)
            if arg_count == [1, 1, 1]:
                event_target_idxs[" ".join(event)] = [words.index(arg) for arg in event]

            else:
                continue #event中的arg词在sentence中出现多次 或 arg由两个词组成
                # print("The sentence is: %s.\n The event argument %s occurs %d times".format(sentence, arg, word_result[arg]))
        self.LMMS(sub_doc, ctx_embeddings, event_target_idxs)
       



def event_sense_mapping(dis, file_name):
    
    with open(os.path.join(base_path, file_name)) as f:
        for line_num, line in enumerate(f): #line_num从0开始
            
            # if line_num != 1888:# 有CudaError是从这一行开始的
            #     continue
            raw_single_data = line.split("|SENT") #一个file
            for sentence in raw_single_data:
                ele = sentence.strip("|").split("|")
                sentence = ele[2]
                all_events = []
                for index in range(3, len(ele)):
                    if ele[index] == "TUP":
                        all_events.append([])  
                    else:
                        all_events[-1].append(ele[index])  
                        
                try:
                    dis.process_sentence(sentence, all_events)
                except Exception as e:
                    logger.error(e)
                    logger.error("{} in {} error. Then sentence is {}".format(line_num, raw_single_data[0].split("|")[0], sentence), exc_info=sys.exc_info())
                    # logging.error(traceback.format_exc())
            logger.info("{} event mapping finished!".format(raw_single_data[0].split("|")[0]))
            

            if line_num % 1000 == 0:  #每1000个文件保存一次
                with open('./results_save/event_sense_mapping_{}.json'.format(file_name[0:4]), 'w') as esm:
                    json.dump(dis.event_sense_mapping, esm)
                    esm.close()
                    
        with open('./results_save/event_sense_mapping_{}.json'.format(file_name[0:4]), 'w') as esm:
                json.dump(dis.event_sense_mapping, esm)
                esm.close()
        f.close()


    logger.info("All the events in {} mapping finished!".format(file_name))

if __name__ == '__main__':
    dis = disambiguation()
    for i in range(1, len(sys.argv)):
        file_prefix = sys.argv[i]
        logging.basicConfig(filename='1_event_sense_mapping_{}.log'.format(file_prefix), format='%(asctime)s | %(levelname)s | %(message)s', level=logging.DEBUG, filemode='w') #有filename是文件日志输出,filemode是’w’的话，文件会被覆盖之前生成的文件会被覆盖
        global logger
        logger = logging.getLogger('__name__')
        mail_handler = SMTPHandler(
            mailhost=('smtp.qq.com', 25),
            fromaddr='524139952@qq.com',
            toaddrs='weishuchong@iie.ac.cn',
            subject='代码出现问题啦！！！',
            credentials=('524139952@qq.com', 'wyftpscaofhucbdg'))
        # 4. 单独设置 mail_handler 的日志级别为 ERROR
        mail_handler.setLevel(logging.ERROR)
        # 5. 将 Handler 添加到 logger 中
        logger.addHandler(mail_handler)
        
        dis.event_sense_mapping = {}
        file_name = file_prefix + ".txt"    
        event_sense_mapping(dis, file_name)



    
    #全部运行
    # for root, dirs, files in os.walk(base_path):
    #     for name in files:
    #         with open(os.path.join(base_path , name)) as f:
    #             for line in f:
    #                 raw_single_data = line.split("|SENT")
    #                 for sentence in raw_single_data:
    #                     ele = sentence.strip("|").split("|")
    #                     sentence = ele[2]
    #                     all_events = []
    #                     all_words = []
    #                     for index in range(3, len(ele)):
    #                         if ele[index] == "TUP":
    #                             all_events.append([])  
    #                         else:
    #                             all_events[-1].append(ele[index])
                            
    #                     dis.process_sentence(sentence, all_events)
    #                 logging.info("{} event mapping finished!".format(raw_single_data[0].split("|")[0]))
    #             f.close()


                          
                

        