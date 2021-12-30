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
from regex import sub
from transformers_encoder import TransformersEncoder
from vectorspace import SensesVSM
from wn_utils import WN_Utils
from collections import Counter
from logging.handlers import SMTPHandler

import torch
import numpy as np
import traceback
import pickle
import logging
import sys
import spacy
import json
import sys
base_path = "../SimCSE-main/LDCCorpus/gigaword_eng_5/data/nyt_online_4"
# base_path = "./nyt_online_4"

#Readme
#使用指令 python 1++_event_sense_mapping.py 1994 1995 1996
#transfer to another machine : base_path, spacy.load路径
#nltk 下载wordnet数据集

class disambiguation(object):
    def __init__(self, file_prefix) -> None:
        super().__init__()
        if os.path.exists('./results_save/event_sense_mapping_{}.json'.format(file_prefix)):
            with open('./results_save/event_sense_mapping_{}.json'.format(file_prefix), 'r') as esm:
                self.event_sense_mapping = json.load(esm)
        logger.info('{} load successfully!'.format('event_sense_mapping_{}.json'.format(file_prefix)))

        self.en_nlp = spacy.load('/home/anaconda3/envs/LMMS/lib/python3.6/site-packages/en_core_web_sm/en_core_web_sm-1.2.0')  # required for lemmatization and POS-tagging
        # self.en_nlp = spacy.load('/home/iielct/anaconda3/envs/LMMS/lib/python3.6/site-packages/en_core_web_sm/en_core_web_sm-1.2.0')  # required for lemmatization and POS-tagging
        print("Spacy load successfully!")


        with open('./results_save/wsd_encoder.pkl', "rb") as wsd:
            self.wsd_encoder = pickle.load(wsd)
        with open('./results_save/senses_vsm.pkl', "rb") as senses:
            self.senses_vsm = pickle.load(senses) 
        self.wn_utils = WN_Utils()  # WordNet auxilliary methods (just for describing results)
    
        # # NLM/LMMS paths and parameters
        # vecs_path = './data/vectors/lmms-sp-wsd.albert-xxlarge-v2.vectors.txt'
        # wsd_encoder_cfg = {
        #     'model_name_or_path': 'albert-xxlarge-v2',
        #     'min_seq_len': 0,
        #     'max_seq_len': 512,
        #     'layers': [-n for n in range(1, 12 + 1)],  # all layers, with reversed indices
        #     'layer_op': 'ws',
        #     'weights_path': 'data/weights/lmms-sp-wsd.albert-xxlarge-v2.weights.txt',
        #     'subword_op': 'mean'
        # }
        # print('Loading NLM and sense embeddings ...')  # (takes a while)
        # self.wsd_encoder = TransformersEncoder(wsd_encoder_cfg)
        # self.senses_vsm = SensesVSM(vecs_path, normalize=True)
        # with open('./results_save/wsd_encoder.pkl', "wb") as wsd: 
        #     pickle.dump(self.wsd_encoder, wsd)
        # with open('./results_save/senses_vsm.pkl', "wb") as senses: 
        #     pickle.dump(self.senses_vsm, senses)
        print('Done')
        
    def judge(self, lemma=None, postag=None):
        relevant_sks = []
        for sk in self.senses_vsm.labels:
            if (lemma is None) or (self.senses_vsm.sk_lemmas[sk] == lemma):
                if (postag is None) or (self.senses_vsm.sk_postags[sk] == postag):
                    relevant_sks.append(sk)
        return relevant_sks
    
    def LMMS(self, sub_doc_batch, ctx_embeddings_batch, event_target_idxs_batch):

        for sub_doc, ctx_embeddings, event_target_idxs in zip(sub_doc_batch, ctx_embeddings_batch, event_target_idxs_batch):
            for current_event, target_idxs in event_target_idxs.items():
                syn_the_best_match = {}
                for target_idx in target_idxs:
                    target_pos = sub_doc[target_idx].pos_
                    target_lemma = '_'.join([sub_doc[i].lemma_ for i in [target_idx]])
                    
                    relevant_sks = self.judge(lemma=target_lemma, postag=target_pos)
                    if relevant_sks != []:
                        target_embedding = np.array([ctx_embeddings[i][1] for i in [target_idx]]).mean(axis=0)
                        target_embedding = target_embedding / np.linalg.norm(target_embedding)
                        matches = self.senses_vsm.match_senses(target_embedding, relevant_sks, topn=1)
                        assert matches != []
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
                    # logger.info('{} has mapped'.format(current_event))

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
                    
        
        
    def process_sentence(self, sentence_batch, all_events_batch):
        torch.cuda.empty_cache()
        #sub_doc_batch,tokens_batch,ctx_embeddings_batch,words_batch,word_result_batch,event_target_idxs_batch应该长度一致
        sub_doc_batch = []
        tokens_batch = []
        for sentence in sentence_batch:
            sub_doc = self.en_nlp(sentence)
            sub_doc_batch.append(sub_doc)
        for sub_doc in sub_doc_batch:
            tokens = [t.text for t in sub_doc]
            tokens_batch.append(tokens)
        ctx_embeddings_batch, exceed_len_index = self.wsd_encoder.token_embeddings(tokens_batch)

        words_batch = []
        for sub_doc in sub_doc_batch:
            words = []
            for word in sub_doc:
                words.append(word.text)
            words_batch.append(words)

        word_result_batch = []
        for words in words_batch:
            word_result = Counter(words)
            word_result_batch.append(word_result)

        # 删除超过长度的
        if exceed_len_index != []:
            words_batch = [words_batch[i] for i in range(0, len(words_batch), 1) if i not in exceed_len_index]
            word_result_batch = [word_result_batch[i] for i in range(0, len(word_result_batch), 1) if i not in exceed_len_index]
            sub_doc_batch = [sub_doc_batch[i] for i in range(0, len(sub_doc_batch), 1) if i not in exceed_len_index]
            all_events_batch = [all_events_batch[i] for i in range(0, len(all_events_batch), 1) if i not in exceed_len_index]

    
        event_target_idxs_batch = []
        for index, all_events in enumerate(all_events_batch):
            event_target_idxs = {} #一个sentence对应多个event，需要筛选出一部分event出来。
            for event in all_events:
                arg_count_mapping = [{arg: word_result_batch[index][arg]} for arg in event]
                arg_count = []
                # for a in arg_count_mapping:
                #     for _, v in a.items():
                #         arg_count += v   [{'Sirakov': 1}, {'put in': 0}, {'minute': 2}]导致利用arg_count = 3判断出错。
                for acm in arg_count_mapping:
                    for _, v in acm.items():
                        arg_count.append(v)
                if arg_count == [1, 1, 1]:
                    event_target_idxs[" ".join(event)] = [words_batch[index].index(arg) for arg in event]

                else:
                    continue #event中的arg词在sentence中出现多次 或 arg由两个词组成
                    # print("The sentence is: %s.\n The event argument %s occurs %d times".format(sentence, arg, word_result[arg]))
            event_target_idxs_batch.append(event_target_idxs)
        assert len(ctx_embeddings_batch) == len(words_batch) == len(word_result_batch) == len(event_target_idxs_batch)
        self.LMMS(sub_doc_batch, ctx_embeddings_batch, event_target_idxs_batch)
       



def event_sense_mapping(dis, file_name, line_continue):
    batch = 32
    sentence_batch = []
    all_events_batch = []
    line_num_batch = []
    
    with open(os.path.join(base_path, file_name)) as f:
        for line_num, line in enumerate(f): #line_num从0开始
            if line_num <= int(line_continue):
                logger.info("The line {} has processed before!".format(line_num))
                continue
            line_num_batch.append(line_num)#添加batch

            # if line_num != 1888:# 1995.txt有CudaError是从这一行开始的，LMMS输入长度太长
            #     continue
            raw_single_data = line.split("|SENT") #一个file
            for sentence in raw_single_data:
                ele = sentence.strip("|").split("|")
                sentence = ele[2]
                sentence_batch.append(sentence) #添加batch
                all_events = []
                for index in range(3, len(ele)):
                    if ele[index] == "TUP":
                        all_events.append([])  
                    else:
                        all_events[-1].append(ele[index]) 
                all_events_batch.append(all_events)       #添加batch
                if len(sentence_batch) == batch and len(all_events_batch) == batch:
                    try:
                        dis.process_sentence(sentence_batch, all_events_batch)
                        for line_num in line_num_batch:
                            logger.info("{}th line's event mapping finished!".format(line_num))
                        sentence_batch = []
                        all_events_batch = []
                        line_num_batch = []
                    except Exception as e:
                        # logger.error(e)
                        line_string = ''
                        for line_num in line_num_batch:
                            line_string = line_string + ' ' + str(line_num)
                        logger.error("Year: {} Line: {} exists error".format(file_name, line_string), exc_info=sys.exc_info())  # logging.error(traceback.format_exc())
                        with open('./results_save/event_sense_mapping_{}.json'.format(file_name[0:4]), 'w') as esm:
                            json.dump(dis.event_sense_mapping, esm)
                            esm.close() 
            if line_num % 500 == 0:  #每1000个文件保存一次
                with open('./results_save/event_sense_mapping_{}.json'.format(file_name[0:4]), 'w') as esm:
                    json.dump(dis.event_sense_mapping, esm)
                    esm.close() 
        # 处理最后一个batch  
    if len(sentence_batch) > 0:          
        try:
            dis.process_sentence(sentence_batch, all_events_batch)
            for line_num in line_num_batch:
                logger.info("{}th line's event mapping finished!".format(line_num))
            logger.info("The last sentence {} is processed well".format(sentence_batch[-1]))
        except Exception as e:
            # logger.error(e)
            line_string = ''
            for line_num in line_num_batch:
                line_string = line_string + ' ' + str(line_num)
            logger.error("Year: {} Line: {} exists error".format(file_name, line_string), exc_info=sys.exc_info())
                          
        with open('./results_save/event_sense_mapping_{}.json'.format(file_name[0:4]), 'w') as esm:
                json.dump(dis.event_sense_mapping, esm)
                esm.close()
        f.close()


    logger.info("All the events in {} mapping finished!".format(file_name))

if __name__ == '__main__':

    file_prefix = sys.argv[1]
    line_continue = sys.argv[2]
    logging.basicConfig(filename='./log/1++_event_sense_mapping_{}.log'.format(file_prefix), format='%(asctime)s | %(levelname)s | %(message)s', level=logging.DEBUG, filemode='w') #有filename是文件日志输出,filemode是’w’的话，文件会被覆盖之前生成的文件会被覆盖
    global logger
    logger = logging.getLogger('__name__')
    mail_handler = SMTPHandler(
        mailhost=('smtp.qq.com', 25),
        fromaddr='524139952@qq.com',
        toaddrs='weishuchong19@mails.ucas.ac.cn',
        subject='代码出现问题啦！！！',
        credentials=('524139952@qq.com', 'wyftpscaofhucbdg'))
    # 4. 单独设置 mail_handler 的日志级别为 ERROR
    mail_handler.setLevel(logging.ERROR)
    # 5. 将 Handler 添加到 logger 中
    logger.addHandler(mail_handler)
    
    file_name = file_prefix + ".txt"   
    dis = disambiguation(file_prefix) 
    event_sense_mapping(dis, file_name, line_continue)



    
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


                          
                

        