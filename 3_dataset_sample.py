import networkx as nx    
import os 
import itertools
from random import choices, sample
import pickle
import random
import logging

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
        
        
    def collect_event_synset_dict(self):
        # self.dict_synset_to_event[synset_node] = []
        # for event_node in adjs_syn_node:
        #     # event_node = event_node.lower()
        #     self.dict_synset_to_event[synset_node].append(event_node)
            
        #     if event_node not in self.dict_event_to_synset:
        #         self.dict_event_to_synset[event_node] = [synset_node]
        #     else:
        #         self.dict_event_to_synset[event_node].append((self.G[synset_node][event_node]['arg'], synset_node)) #
        self.dict_synset_to_event = {}
        self.dict_event_to_synset = {}
        for synset_node in self.synset_nodes:
            if synset_node not in self.dict_synset_to_event:
                self.dict_synset_to_event[synset_node] = []
            adjs_syn_node = self.G[synset_node]
            for event_node in adjs_syn_node:
                
                arg = self.G[synset_node][event_node]["arg"].lower()
                event_node = event_node.lower()
                
                self.dict_synset_to_event[synset_node].append((arg, event_node))
                if event_node not in self.dict_event_to_synset:
                    self.dict_event_to_synset[event_node] = []
                else:
                    self.dict_event_to_synset[event_node].append((arg, synset_node))
                        
            
                
                
        file = open('results_save/dict_event_to_synset.pickle', 'wb')
        pickle.dump(self.dict_event_to_synset, file)
        file.close()
        
        file2 = open('results_save/dict_synset_to_event.pickle', 'wb')
        pickle.dump(self.dict_synset_to_event, file2)
        file2.close()
        print("dict_event_to_synset and dict_synset_to_event are saved well.")

        return
        
        
    def dataset_sample(self, sample_num = 100):
        # n = 0
        if os.path.exists("./results_save/verb_centric_triple_events.txt"):
            os.remove("./results_save/verb_centric_triple_events.txt") 
        
        
        for synset_node in self.synset_nodes:
            if synset_node.find(".v.") == -1:
                continue
            else:
                adjs_syn_node = self.G[synset_node]
                event_nodes_adj_for_syn_nodes = list(filter(self.att_event, adjs_syn_node))
                # combination_number = len(adjs_syn_node) * (len(adjs_syn_node) - 1) // 2
                # print("{} : {}".format(synset_node, len(event_nodes_adj_for_syn_nodes)))
                if  sample_num < len(event_nodes_adj_for_syn_nodes) :
                    sample_events = list(random.sample(event_nodes_adj_for_syn_nodes, k = sample_num))
                    event_positive_pairs = list(itertools.combinations(sample_events, 2))
                    neg_events = list(random.sample(self.event_nodes, k = len(event_positive_pairs)))
                else:
                    
                    event_positive_pairs = list(itertools.combinations(event_nodes_adj_for_syn_nodes, 2))
                    neg_events = list(random.sample(self.event_nodes, k = len(event_positive_pairs)))

                # com_event_positive_pairs = list(itertools.combinations(event_nodes_adj_for_syn_nodes, 2))
                # print("comb num is : {}".format(len(com_event_positive_pairs)))
                # event_positive_pairs = list(random.sample(com_event_positive_pairs, k = min(100, len(com_event_positive_pairs))))    #只保留100个组合
                # del com_event_positive_pairs
                
                
                
                # #event-synset 和 synset-event转化
                # self.dict_synset_to_event[synset_node] = []
                # for event_node in adjs_syn_node:
                #     # event_node = event_node.lower()
                #     self.dict_synset_to_event[synset_node].append(event_node)
                    
                #     if event_node not in self.dict_event_to_synset:
                #         self.dict_event_to_synset[event_node] = [synset_node]
                #     else:
                #         self.dict_event_to_synset[event_node].append((self.G[synset_node][event_node]['arg'], synset_node)) #
                # neg_events = list(random.sample(self.event_nodes, k = sample_num / 2))#https://stackoverflow.com/questions/43281886/get-a-random-sample-with-replacement

                # neg_events = choices(self.event_nodes, k = 100)#https://stackoverflow.com/questions/43281886/get-a-random-sample-with-replacement
                
                assert len(neg_events) == len(event_positive_pairs)
                
                
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
                write_lines = []
                for index, event_positive_pair in enumerate(event_positive_pairs):
                    neg_events_arg = []
                    for adj_neg_event in nx.all_neighbors(self.G, neg_events[index]):
                        neg_events_arg.append(self.G[adj_neg_event][neg_events[index]]['arg'])
                
                    lines = synset_node + " || " + event_positive_pair[0] + "<>" + self.G[synset_node][event_positive_pair[0]]['arg'] + " || " + event_positive_pair[1] + "<>" + self.G[synset_node][event_positive_pair[1]]['arg'] + " || " + neg_events[index] + "<>" + ",".join(neg_events_arg) + "\n"
                    write_lines.append(lines.lower())
                    # print(synset_node + " || " + event_positive_pair[0] + " || " + event_positive_pair[1] + " || " + neg_events[index] + "\n")

                
                # file = open('results_save/write_lines.pickle', 'wb')
                # pickle.dump(write_lines, file)
                # file.close()
                
                with open ("results_save/verb_centric_triple_events.txt", mode = "a+", encoding = "utf-8") as f:#需要指定mode和encoding，否则an integer is required (got type str)

                    try:
                        f.writelines(write_lines)
                    except:
                        for write_line in write_lines:
                            logging.info(write_line)
            
            # del event_positive_pair
            # del event_positive_pairs
            # del neg_events
            # del adjs_syn_node
            
            # for event_triple in zip(event_positive_pairs, neg_events):
            #     # neg_events_arg = []
            #     # for adj_neg_event in nx.all_neighbors(self.G, event_triple[1]):
            #     #     neg_events_arg.append(self.G[adj_neg_event][event_triple[1]]['arg'])
            #     # f.writelines(synset_node + " || " + event_triple[0][0] + "<>" + self.G[synset_node][event_triple[0][0]]['arg'] + " || " + event_triple[0][1] + "<>" + self.G[synset_node][event_triple[0][1]]['arg'] + " || " + event_triple[1] + "<>" + ",".join(neg_events_arg) + " \n")
            #     f.writelines(synset_node + " || " + event_triple[0][0] + " || " + event_triple[0][1] + " || " + event_triple[1] + " \n")
                
                
                
                # n = n + 1
                # if n > 5000000:
                #     f.close()
                #     f = open ("results_save/verb_centric_triple_events_{}.txt".format(n//5000000), "w")
        
        logging.info("Dataset is constructed well.")

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
    logging.basicConfig(filename='./log/3_dataset_sample.log', format='%(asctime)s | %(levelname)s | %(message)s', level=logging.DEBUG, filemode='w') #有filename是文件日志输出,filemode是’w’的话，文件会被覆盖之前生成的文件会被覆盖

    du = dataset_util(path = "results_save/")
    du.dataset_sample()
    # du.collect_event_synset_dict()