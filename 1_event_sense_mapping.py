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

import pickle
import logging
import sys
import spacy
base_path = "../SimCSE-main/LDCCorpus/gigaword_eng_5/data/nyt_online_4"



class disambiguation(object):
    def __init__(self) -> None:
        super().__init__()
        self.event_sense_mapping = {}
        self.en_nlp = spacy.load('/home/anaconda3/envs/LMMS/lib/python3.6/site-packages/en_core_web_sm/en_core_web_sm-1.2.0')  # required for lemmatization and POS-tagging
        print("Spacy load successfully!")
        
        with open('wsd_encoder.pkl', "rb") as wsd:
            self.wsd_encoder = pickle.load(wsd)
        with open('senses_vsm.pkl', "rb") as senses:
            self.senses_vsm = pickle.load(senses) 
        
        self.wn_utils = WN_Utils()  # WordNet auxilliary methods (just for describing results)
        # # NLM/LMMS paths and parameters
        # # vecs_path = '/media/dan/ElementsWORK/-xxlarge-v2/albertlmms-sp-wsd.albert-xxlarge-v2.vectors.txt'
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
        # print('Done')

    def LMMS(self, sub_doc, ctx_embeddings, event_target_idxs, current_event):

        syn_the_best_match = []
        for target_idxs in event_target_idxs.values():
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
                    syn_the_best_match.append({sub_doc[target_idx]: [the_best_match, the_best_sim]})
                else:
                    continue
        
        if current_event not in self.event_sense_mapping.keys():
            self.event_sense_mapping[current_event] = syn_the_best_match
        else:
            logging.warning('{} is saved before'.format(current_event))
                    
        
        
    def process_sentence(self, sentence, all_events):
        sub_doc = self.en_nlp(sentence)
        tokens = [t.text for t in sub_doc]
        ctx_embeddings = self.wsd_encoder.token_embeddings([tokens])[0]
        words = []
        for word in sub_doc:
            words.append(word.text)
        word_result = Counter(words)
        event_target_idxs = {}

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
                self.LMMS(sub_doc, ctx_embeddings, event_target_idxs, " ".join(event))

            else:
                continue #event中的arg词在sentence中出现多次 或 arg由两个词组成
                # print("The sentence is: %s.\n The event argument %s occurs %d times".format(sentence, arg, word_result[arg]))
                    



def event_sense_mapping(dis, file_name):
    with open(os.path.join(base_path , file_name)) as f:
        for line in f:
            raw_single_data = line.split("|SENT")
            for sentence in raw_single_data:
                ele = sentence.strip("|").split("|")
                sentence = ele[2]
                all_events = []
                for index in range(3, len(ele)):
                    if ele[index] == "TUP":
                        all_events.append([])  
                    else:
                        all_events[-1].append(ele[index])
                    
                dis.process_sentence(sentence, all_events)
            logging.info("{} event mapping finished!".format(raw_single_data[0].split("|")[0]))
        f.close()

    with open("event_sense_mapping_{}.pkl".format(file_name[0:4]), "wb") as esm:
        pickle.dump(dis.event_sense_mapping, esm)
    logging.info("All the events in {} mapping finished!".format(file_name))

if __name__ == '__main__':
    dis = disambiguation()
    for i in range(1, len(sys.argv)):
        file_prefix = sys.argv[i]
        dis.event_sense_mapping = {}
        logging.basicConfig(filename='1_event_sense_mapping_{}.log'.format(file_prefix), level=logging.DEBUG)
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


                          
                

        