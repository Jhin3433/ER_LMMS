from functools import partial
from nltk.corpus import wordnet as wn
import json
import networkx as nx
import os
import re
import matplotlib.pyplot as plt
import logging
import gc
import sys
#筛选NetworkX graph以列出来自具有特定属性的节点的所有边 : https://www.cnpython.com/qa/169566
#Python包 network-x : https://www.cnblogs.com/ljhdo/p/10662902.html
#networkx官方references : https://networkx.org/documentation/stable/reference/index.html
#networkx官方tutorial : https://networkx.org/documentation/stable/tutorial.html#adding-attributes-to-graphs-nodes-and-edges

#如何使用python nrtworkx将图放大？ : https://segmentfault.com/q/1010000009235034
#networkx画图时显示节点和边的属性 : https://blog.csdn.net/qq_41854763/article/details/103405760

class event_sense_graph:
    def __init__(self) -> None:
        if os.path.exists("results_save/2+_undir_event_sense_mapping_graph.gpickle"):
            self.G = nx.read_gpickle("results_save/2+_undir_event_sense_mapping_graph.gpickle")
        else:
            self.G = nx.Graph()#创建空的无向图
            
    def att_event(self, n):
        return 'type' in self.G.nodes[n] and self.G.nodes[n]['type'] == 'event' #查看边属性self.G.get_edge_data('bleary.s.02',"It 's fuzzy")
    def att_synset(self, n):
        return 'type' in self.G.nodes[n] and self.G.nodes[n]['type'] == 'synset'
    
    def modify_graph(self, dir_name = "./results_save/event_sense_mapping_json"):
        for path,dir_list,file_list in os.walk(dir_name)  :
            for file_name in file_list:
                with open(os.path.join(path, file_name), "r") as esm:
                    event_sense_mappings = json.load(esm)
                    for event, event_sense_mapping in event_sense_mappings.items():
                        for arg, sense in event_sense_mapping.items():
                            if self.G[sense[0]][event]["arg"] == arg:
                                if self.G[sense[0]][event]["sim_weight"] < sense[1]:
                                    self.G[sense[0]][event]["sim_weight"] = sense[1]
                                    logging.info("Change valid. {} -> {} : {} < New sim_weight is {}".format(sense[0], event, self.G[sense[0]][event]["sim_weight"], sense[1]))   
                                else:
                                    logging.info("Change invalid. {} -> {} : {} > New sim_weight is {}".format(sense[0], event, self.G[sense[0]][event]["sim_weight"], sense[1]))     
                            else:
                                logging.error("{} -> {} arg: {} . But new arg is {}".format(sense[0], event, self.G[sense[0]][event]["arg"], arg))     
                                
    def multi_graph_from_json(self, dir_name = "./results_save/event_sense_mapping_json"):
        event_nodes = []
        synset_nodes = []
        edge_lists = []
        edge_verify_dict = {}#用来验证event和synset之间是否已有边相连
        for path,dir_list,file_list in os.walk(dir_name)  :
            for file_name in file_list:
                with open(os.path.join(path, file_name), "r") as esm:
                    event_sense_mappings = json.load(esm)
                    for event, event_sense_mapping in event_sense_mappings.items():
                        event_nodes.append(event)
                        for arg, sense in event_sense_mapping.items():
                            synset_nodes.append(sense[0])
                            
                            if sense[0] + "|" + event not in edge_verify_dict.keys():
                                edge_verify_dict[sense[0] + "|" + event] = sense[1]
                                edge_lists.append((sense[0], event, {"arg":arg, "sim_weight":sense[1], "type":"event_synset_edge"})) 
                            else:
                                logging.info("{} -> {} is existing.".format(sense[0], event))

                                
                logging.info("{} is processed well.".format(file_name))    
                logging.info("The Current event_node number in {} is {}.".format(file_name, len(event_nodes)))   
        logging.info("The number of ALL event_nodes is {}.".format(len(event_nodes))) 
        
        
        self.G.add_nodes_from(event_nodes, type = "event")
        self.G.add_nodes_from(synset_nodes, type = "synset")
        self.G.add_edges_from(edge_lists)
        logging.info("The Nodes and Edges are added well.") 

        self.modify_graph()
        logging.info("The Graph is modified well.") 
        nx.write_gpickle(self.G, "results_save/2+_undir_event_sense_mapping_graph.gpickle")

    

if __name__ == "__main__":
    logging.basicConfig(filename='2+_construct_undir_graph.log', format='%(asctime)s | %(levelname)s | %(message)s', level=logging.DEBUG, filemode='w') #有filename是文件日志输出,filemode是’w’的话，文件会被覆盖之前生成的文件会被覆盖



    esg = event_sense_graph()
    esg.multi_graph_from_json( )