from nltk.corpus import wordnet as wn
import networkx as nx 
import pickle
import os
import math
from torch import neg


def shortest_path(undir_graph_path):
    udrtG = nx.read_gpickle(undir_graph_path)
    shortest_path = dict(nx.shortest_path_length(udrtG)) 
    file = open('results_save/shortest_path.pickle', 'wb')
    pickle.dump(shortest_path, file)
    file.close()




class synset_path_statistic:
    def __init__(self, dir_graph_path = "./results_save/2+_event_sense_mapping_graph.gpickle"):
        if os.path.exists(dir_graph_path):
            self.G = nx.read_gpickle(dir_graph_path)
        self.synset_nodes, self.event_nodes = self.synset_node_identify()

    # 有问题
    def synset_path(self):
        self.synset_where_path = {}
        self.synset_path = {}

        for synset in self.synset_nodes:
            self.synset_where_path[synset] = []
        num = 0
        for synset in self.synset_nodes:
            self.synset_path[num] = []
            synset = wn.synset(synset)
            try:
                for hypo in synset.hyponyms():
                    self.synset_where_path[hypo].append(num)
                    self.synset_path[num].insert(0, hypo)
            except:
                print(synset.name())

            try:
                for hyper in synset.hypernyms():
                    self.synset_where_path[hyper].append(num)
                    self.synset_path[num].append(hyper)
            except:
                print(synset.name())

            num += 1



        print("一共有{}条synset path".format(num))



    def att_synset(self, n):
        return 'type' in self.G.nodes[n] and self.G.nodes[n]['type'] == 'synset'
    def att_event(self, n):
        return 'type' in self.G.nodes[n] and self.G.nodes[n]['type'] == 'event'
    def synset_node_identify(self):
        synset_nodes = []
        event_nodes = []
        for n in list(self.G.nodes):
            if self.att_synset(n):
                synset_nodes.append(n)   
            else:
                event_nodes.append(n)   
        if len(synset_nodes) + len(event_nodes) == len(self.G.nodes):
            return synset_nodes , event_nodes



def normalize_weight(weight):
    for index, x in enumerate(weight):
        if x != 0:
            weight[index] = -math.log(x)
    sum = 0
    for x in weight:
        sum += x
    
    if sum == 0:
        return [0.3, 0.3, 0.3]
    weight = [x/sum for x in weight] 
    return weight

def max_sim(synset_list_1, synset_list_2):
    # if synset_list_1 == synset_list_2:
    #     return 0.1

    max = 0
    for synset_1 in synset_list_1:
        for synset_2 in synset_list_2:
            # sim = synset_1.path_similarity(synset_2)
            # try:
            sim = synset_1.path_similarity(synset_2)
            # except:
            #     sim = 0

            if sim > max:
                max = sim

    return max
def hard_observation(file_path = "./hard.txt"):


    num_correct = 0
    all_line = 0
    for line in open(file_path): 
        event_arg = line.strip('\n').split('|')
        pos_subj_1 = event_arg[0].strip(' ')
        pos_subj_2 = event_arg[3].strip(' ')
        neg_subj_1 = event_arg[6].strip(' ')
        neg_subj_2 = event_arg[9].strip(' ')

        pos_verb_1 = event_arg[1].strip(' ')
        pos_verb_2 = event_arg[4].strip(' ')
        neg_verb_1 = event_arg[7].strip(' ')
        neg_verb_2 = event_arg[10].strip(' ')

        pos_obj_1 = event_arg[2].strip(' ')
        pos_obj_2 = event_arg[5].strip(' ')
        neg_obj_1 = event_arg[8].strip(' ')
        neg_obj_2 = event_arg[11].strip(' ')


        try:

            syn_pos_subj_1 = wn.synsets(pos_subj_1, pos=wn.NOUN)
            syn_pos_subj_2 = wn.synsets(pos_subj_2, pos=wn.NOUN)
            syn_neg_subj_1 = wn.synsets(neg_subj_1, pos=wn.NOUN)
            syn_neg_subj_2 = wn.synsets(neg_subj_2, pos=wn.NOUN)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

            syn_pos_verb_1 = wn.synsets(pos_verb_1, pos=wn.VERB)
            syn_pos_verb_2 = wn.synsets(pos_verb_2, pos=wn.VERB)
            syn_neg_verb_1 = wn.synsets(neg_verb_1, pos=wn.VERB)
            syn_neg_verb_2 = wn.synsets(neg_verb_2, pos=wn.VERB)

            syn_pos_obj_1 = wn.synsets(pos_obj_1, pos=wn.NOUN)
            syn_pos_obj_2 = wn.synsets(pos_obj_2, pos=wn.NOUN)
            syn_neg_obj_1 = wn.synsets(neg_obj_1, pos=wn.NOUN)
            syn_neg_obj_2 = wn.synsets(neg_obj_2, pos=wn.NOUN)


            # syn_pos_subj_1 = wn.synsets(pos_subj_1)
            # syn_pos_subj_2 = wn.synsets(pos_subj_2)
            # syn_neg_subj_1 = wn.synsets(neg_subj_1)
            # syn_neg_subj_2 = wn.synsets(neg_subj_2)

            # syn_pos_verb_1 = wn.synsets(pos_verb_1)
            # syn_pos_verb_2 = wn.synsets(pos_verb_2)
            # syn_neg_verb_1 = wn.synsets(neg_verb_1)
            # syn_neg_verb_2 = wn.synsets(neg_verb_2)

            # syn_pos_obj_1 = wn.synsets(pos_obj_1)
            # syn_pos_obj_2 = wn.synsets(pos_obj_2)
            # syn_neg_obj_1 = wn.synsets(neg_obj_1)
            # syn_neg_obj_2 = wn.synsets(neg_obj_2)


            pos_subj_sim  = max_sim(syn_pos_subj_1, syn_pos_subj_2)
            neg_subj_sim = max_sim(syn_neg_subj_1, syn_neg_subj_2)
            pos_verb_sim  = max_sim(syn_pos_verb_1, syn_pos_verb_2)
            neg_verb_sim = max_sim(syn_neg_verb_1, syn_neg_verb_2)  
            pos_obj_sim  = max_sim(syn_pos_obj_1, syn_pos_obj_2)
            neg_obj_sim = max_sim(syn_neg_obj_1, syn_neg_obj_2)  

            # if pos_subj_sim == 0 or neg_subj_sim == 0 or pos_verb_sim == 0 or neg_verb_sim == 0 or pos_obj_sim == 0 or neg_obj_sim == 0:
            #     print(line)
            #     continue

            pos_weight = normalize_weight([pos_subj_sim, pos_verb_sim, pos_obj_sim])
            neg_weight = normalize_weight([neg_subj_sim, neg_verb_sim, neg_obj_sim])
            pos_sim = pos_weight[0] * pos_subj_sim + pos_weight[1] * pos_verb_sim + pos_weight[2] * pos_obj_sim
            neg_sim = neg_weight[0] * neg_subj_sim + neg_weight[1] * neg_verb_sim + neg_weight[2] * neg_obj_sim

            print("pos_sim : {}, neg_sim : {}".format(pos_sim, neg_sim))
            if pos_sim > neg_sim:
                num_correct += 1
            all_line += 1
        except:
            continue
        # print(line)
    print("Accuracy : {}, all line : {}".format(num_correct/all_line, all_line))




def training_events_observation(file_path = "./hard_hyper_verb_centric_triple_events.txt"):

    num_correct = 0
    all_line_len = 0
    for line in open(file_path): 
        all_line_len += 1
        split_line = line.split("||")
        pos_1_event, pos_1_arg = split_line[1].strip(" ").split("<>")
        pos_2_event, pos_2_arg = split_line[2].strip(" ").split("<>")
        neg_3_event, neg_3_arg = split_line[3].strip("\n").strip(" ").split("<>")
        raw_subj_1, raw_verb_1, raw_obj_1 = pos_1_event.split(" ")
        pos_subj_1, pos_verb_1, pos_obj_1 = pos_2_event.split(" ")
        neg_subj_1, neg_verb_1, neg_obj_1 = neg_3_event.split(" ")


        syn_raw_subj_1 = wn.synsets(raw_subj_1, pos=wn.NOUN)
        syn_pos_subj_1 = wn.synsets(pos_subj_1, pos=wn.NOUN)
        syn_neg_subj_1 = wn.synsets(neg_subj_1, pos=wn.NOUN)
        syn_raw_verb_1 = wn.synsets(raw_verb_1, pos=wn.VERB)
        syn_pos_verb_1 = wn.synsets(pos_verb_1, pos=wn.VERB)
        syn_neg_verb_1 = wn.synsets(neg_verb_1, pos=wn.VERB)
        syn_raw_obj_1 = wn.synsets(raw_obj_1, pos=wn.NOUN)
        syn_pos_obj_1 = wn.synsets(pos_obj_1, pos=wn.NOUN)
        syn_neg_obj_1 = wn.synsets(neg_obj_1, pos=wn.NOUN)


        pos_subj_sim  = max_sim(syn_raw_subj_1, syn_pos_subj_1)
        neg_subj_sim  = max_sim(syn_raw_subj_1, syn_neg_subj_1)
        pos_verb_sim  = max_sim(syn_raw_verb_1, syn_pos_verb_1)
        neg_verb_sim  = max_sim(syn_raw_verb_1, syn_neg_verb_1)
        pos_obj_sim  = max_sim(syn_raw_obj_1, syn_pos_obj_1)
        neg_obj_sim  = max_sim(syn_raw_obj_1, syn_neg_obj_1)

        pos_sim = pos_subj_sim + pos_verb_sim + pos_obj_sim
        neg_sim = neg_subj_sim + neg_verb_sim + neg_obj_sim
        if pos_sim > neg_sim:
            num_correct += 1
        
    print("Accuracy : {}".format(num_correct/all_line_len))
if __name__ == '__main__':
    graph_path = "results_save/"
    undir_graph_path = graph_path + "2+_undir_event_sense_mapping_graph.gpickle"
    # hard_observation("./hard_extend.txt")
    hard_observation()
    # training_events_observation()




    # sps = synset_path_statistic()
    # sps.synset_path()
    # du = shortest_path(undir_graph_path)