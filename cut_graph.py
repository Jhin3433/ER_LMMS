
import os
import networkx as nx
from glove_utils import Glove
from nltk.corpus import wordnet as wn

class cut:
    def __init__(self):
        dir_graph_path = "./results_save/" + "2+_event_sense_mapping_graph.gpickle"
        if os.path.exists(dir_graph_path):
            self.G = nx.read_gpickle(dir_graph_path)
        self.Glove_file = Glove('./results_save/glove.6B.100d.ext.txt')
        self.synset_nodes, self.event_nodes = self.synset_node_identify()


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

    def delete_event_not_in_Glove(self):

        print("现有event节点{}个，synset节点{}个".format(len(self.event_nodes), len(self.synset_nodes)))
    



    
        remove_nodes = []
        for event_node in self.event_nodes:
            for arg in event_node.split(" "):
                if arg not in self.Glove_file.vocab_id:
                    remove_nodes.append(event_node)
                    self.event_nodes.remove(event_node)
                    break
        

        for synset_node in self.synset_nodes:
            if self.G.degree(synset_node) == 0:
                remove_nodes.append(event_node)
                self.synset_nodes.remove(synset_node)

        self.G.remove_nodes_from(remove_nodes)


        assert len(self.G.nodes) == len(self.synset_nodes) + len(self.event_nodes)
        print("删减后event节点{}个，synset节点{}个".format(len(self.event_nodes), len(self.synset_nodes)))
        nx.write_gpickle(self.G, "results_save/2+_cut_event_sense_mapping_graph.gpickle")


cut_instance = cut()
cut_instance.delete_event_not_in_Glove()