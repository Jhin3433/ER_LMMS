import networkx as nx    
import os 
import itertools
from random import choices
import pickle

from torch import combinations

class dataset_util:
    def __init__(self, path) -> None:
        dir_graph_path = path + "2+_event_sense_mapping_graph.gpickle"
        undir_graph_path = path + "2+_undir_event_sense_mapping_graph.gpickle"

        if os.path.exists(dir_graph_path) and os.path.exists(undir_graph_path):
            self.G = nx.read_gpickle(dir_graph_path)
            # self.udrtG = nx.read_gpickle(undir_graph_path)

        #self.shortest_path = dict(nx.shortest_path_length(self.udrtG)) 
        # file = open('shortest_path.pickle', 'wb')
        # pickle.dump(self.shortest_path, file)
        # file.close()

        self.synset_nodes, self.event_nodes = self.synset_node_identify()
        
        self.dict_event_to_synset = {}
        self.dict_synset_to_event = {}
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
    def dataset_sample(self):
        n = 0
        f = open ("results_save/verb_centric_triple_events.txt", "w")
        for synset_node in self.synset_nodes:
            if synset_node.find(".v.") == -1:
                continue
            adjs_syn_node = self.G[synset_node]
            # event_nodes_adj_for_syn_nodes = filter(self.att_event, adjs_syn_node)
            combination_number = len(adjs_syn_node) * (len(adjs_syn_node) - 1) // 2
            event_positive_pairs = itertools.combinations(adjs_syn_node, 2)
            
            #event-synset 和 synset-event转化
            self.dict_synset_to_event[synset_node] = []
            for event_node in adjs_syn_node:
                event_node = event_node.lower()
                self.dict_synset_to_event[synset_node].append(event_node)
                
                if event_node not in self.dict_event_to_synset:
                    self.dict_event_to_synset[event_node] = [synset_node]
                else:
                    self.dict_event_to_synset[event_node].append(synset_node)

            neg_events = choices(self.event_nodes, k = combination_number)#https://stackoverflow.com/questions/43281886/get-a-random-sample-with-replacement
            
            # if len(self.event_nodes) > len(combination_number):
            #     neg_events = random.sample(self.event_nodes, combination_number)  sample无重复选择self.event_nodes长度的个数
            # else:
            #     event_positive_pairs = random.sample(event_positive_pairs, len(self.event_nodes)//2)
            #     neg_events = random.sample(self.event_nodes, combination_number)

            # else:
            #     print("The len of self.event_nodes is {}. But the num of event_positive_pairs is {}".format(len(self.event_nodes), len(event_positive_pairs)))
            # for index, event_triples in enumerate(zip(event_positive_pairs, neg_events)):
            #     event_positive_pair = event_triples[0]
            #     neg_event = event_triples[1]
            #     # if nx.shortest_path_length(self.udrtG, source=event_positive_pair[0], target=neg_event) < 10 or nx.shortest_path_length(self.udrtG, source=event_positive_pair[1], target=neg_event) < 10:
            #     #     while nx.shortest_path_length(self.udrtG, source=event_positive_pair[0], target=neg_event) < 10 or nx.shortest_path_length(self.udrtG, source=event_positive_pair[1], target=neg_event) < 10:
            #     #         neg_events[index] = random.choice(self.event_nodes)
            #     if self.shortest_path[event_positive_pair[0]][neg_event] < 10 or  self.shortest_path[event_positive_pair[1]][neg_event]< 10:
            #         while self.shortest_path[event_positive_pair[0]][neg_event] < 10 or self.shortest_path[event_positive_pair[1]][neg_event] < 10:
            #             neg_events[index] = random.choice(self.event_nodes) 
                    
            for event_triple in zip(event_positive_pairs, neg_events):
                neg_events_arg = []
                for adj_neg_event in nx.all_neighbors(self.G, event_triple[1]):
                    neg_events_arg.append(self.G[adj_neg_event][event_triple[1]]['arg'])
                f.writelines(synset_node + " || " + event_triple[0][0] + "<>" + self.G[synset_node][event_triple[0][0]]['arg'] + " || " + event_triple[0][1] + "<>" + self.G[synset_node][event_triple[0][1]]['arg'] + " || " + event_triple[1] + "<>" + ",".join(neg_events_arg) + " \n")
                # n = n + 1
                # if n > 5000000:
                #     f.close()
                #     f = open ("results_save/verb_centric_triple_events_{}.txt".format(n//5000000), "w")
        
        f.close()
        
        
        file = open('results_save/dict_event_to_synset.pickle', 'wb')
        pickle.dump(self.dict_event_to_synset, file)
        file.close()
        
        file2 = open('results_save/dict_synset_to_event.pickle', 'wb')
        pickle.dump(self.dict_synset_to_event, file2)
        file2.close()

        print("dict_event_to_synset and dict_synset_to_event are saved well.")

        print("Dataset is constructed well. The Total num of line is {}.".format(n))

        return
    
    
    # #为超图准备
    # def collect_event_synset_dict(self):
    #     self.dict_event_to_synset = {}
    #     self.dict_synset_to_event = {}
        
    #     for synset_node in self.synset_nodes:
    #         adjs_syn_node = self.G.adj[synset_node]
    #         event_nodes_adj_for_syn_nodes =  filter(self.att_event, adjs_syn_node)
    #         self.dict_synset_to_event[synset_node] = []
            
    #         for event_node in event_nodes_adj_for_syn_nodes:
    #             event_node = event_node.lower()
    #             self.dict_synset_to_event[synset_node].append(event_node)
                
    #             if event_node not in self.dict_event_to_synset:
    #                 self.dict_event_to_synset[event_node] = [synset_node]
    #             else:
    #                 self.dict_event_to_synset[event_node].append(synset_node)
        
    #     file = open('results_save/dict_event_to_synset.pickle', 'wb')
    #     pickle.dump(self.dict_event_to_synset, file)
    #     file.close()
        
    #     file2 = open('results_save/dict_synset_to_event.pickle', 'wb')
    #     pickle.dump(self.dict_synset_to_event, file2)
    #     print("dict_event_to_synset and dict_synset_to_event are saved well.")
    #     file2.close()

        
if __name__ == '__main__':
    
    du = dataset_util(path = "results_save/")
    du.dataset_sample()
    # du.collect_event_synset_dict()