import networkx as nx    
import os 
import itertools
class dataset_util:
    def __init__(self, path) -> None:
        if os.path.exists(path):
            self.G = nx.read_gpickle(path)
        
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
    def dataset_sample(self):
        for synset_node in self.synset_nodes:
            adjs_syn_node = self.G.adj[synset_node]
            event_nodes_adj_for_syn_nodes =  filter(self.att_event, adjs_syn_node)
            event_positive_pairs = list(itertools.combinations(event_nodes_adj_for_syn_nodes, r = 2)) 
        

if __name__ == '__main__':
    graph_path = "results_save/event_sense_mapping_graph.gpickle"
    
    du = dataset_util(graph_path)