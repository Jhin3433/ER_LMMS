from nltk.corpus import wordnet as wn
import json
import networkx as nx
import os
import re
import matplotlib.pyplot as plt
#筛选NetworkX graph以列出来自具有特定属性的节点的所有边 : https://www.cnpython.com/qa/169566
#Python包 network-x : https://www.cnblogs.com/ljhdo/p/10662902.html
#networkx官方references : https://networkx.org/documentation/stable/reference/index.html
#networkx官方tutorial : https://networkx.org/documentation/stable/tutorial.html#adding-attributes-to-graphs-nodes-and-edges



class event_sense_graph:
    def __init__(self) -> None:
        if os.path.exists("results_save/event_sense_mapping_graph.gpickle"):
            self.G = nx.read_gpickle("results_save/event_sense_mapping_graph.gpickle")
        else:
            self.G = nx.DiGraph()#创建空的有向图
        
    def att_event(self, n):
        return 'type' in self.G.nodes[n] and self.G.nodes[n]['type'] == 'event' #查看边属性self.G.get_edge_data('bleary.s.02',"It 's fuzzy")
    def att_synset(self, n):
        return 'type' in self.G.nodes[n] and self.G.nodes[n]['type'] == 'synset'
    
    def single_graph_from_json(self, file_name):
        with open(file_name, "r") as esm:
            event_sense_mappings = json.load(esm)

        for event, event_sense_mapping in event_sense_mappings.items():
            #判断event节点是否已添加到图中
            if event not in self.G:
                self.G.add_node(event, type = "event")
            for arg, sense in event_sense_mapping.items():
                if sense[0] not in self.G: #判断synset节点是否在图中
                    self.G.add_node(sense[0], type = "synset")
                if self.G.has_edge(sense[0], event) is not True: #判断synset节点和event节点是否有边连接
                    self.G.add_edge(sense[0], event, arg = arg, sim_weight = sense[1], type = "event_synset_edge")
                else:
                    sim = nx.get_edge_attributes(self.G, 'sim_weight')
                    if sim[sense[0], event] < sense[1]:
                        self.G[sense[0]][event]['sim_weight'] = sense[1]
        #synset和synset之间的关系未连接

    def multi_graph_from_json(self, dir_name = "./results_save/event_sense_mapping_json" ): 
        for path, dir_list, file_list in os.walk(dir_name):  
            for file_name in file_list:
                if re.search("event_sense_mapping_\d+.json", file_name) is not None:
                    self.single_graph_from_json(os.path.join(path, file_name))     
                else:
                    continue
        self.synset_link_synset()
        nx.write_gpickle(self.G, "results_save/event_sense_mapping_graph.gpickle")
        
    def synset_link_synset(self):           
        for n in list(self.G.nodes):
            if self.att_synset(n):
                syn = wn.synset(n)
                hyp_paths = syn.hypernym_paths()
                for hyp_path in hyp_paths:
                    for index in range(0, len(hyp_path) - 1):
                        if hyp_path[index]._name in self.G and hyp_path[index + 1]._name in self.G and not self.G.has_edge(hyp_path[index]._name, hyp_path[index + 1]._name):
                            self.G.add_edge(hyp_path[index]._name, hyp_path[index + 1]._name, type = "synset_synset_edge")
    
    def add_graph_to_existing(self, file_name):
        self.single_graph_from_json(file_name)
        self.synset_link_synset()

                
if __name__ == "__main__":
    esg = event_sense_graph()
    esg.multi_graph_from_json()
    nx.draw(esg.G, pos=nx.spring_layout(esg.G))
    plt.show()
    plt.savefig("graph.png")