import networkx as nx    
import os 
import itertools
import random
import pickle
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
        # n = 0
        f = open ("results_save/triple_events.txt", "w")
        for synset_node in self.synset_nodes:
            adjs_syn_node = self.G.adj[synset_node]
            event_nodes_adj_for_syn_nodes =  filter(self.att_event, adjs_syn_node)
            event_positive_pairs = list(itertools.combinations(event_nodes_adj_for_syn_nodes, r = 2)) 
            
            if len(self.event_nodes) > len(event_positive_pairs):
                neg_events = random.sample(self.event_nodes, len(event_positive_pairs))
            else:
                print("The len of self.event_nodes is {}. But the num of event_positive_pairs is {}".format(len(self.event_nodes), len(event_positive_pairs)))
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
                f.writelines(synset_node + " || " + event_triple[0][0] + "$" + self.G[synset_node][event_triple[0][0]]['arg'] + " || " + event_triple[0][1] + "$" + self.G[synset_node][event_triple[0][1]]['arg'] + " || " + event_triple[1] + "$" + ",".join(neg_events_arg) + " \n")
            #     n = n + 1
            #     if n > 10:
            #         break
            # break
        f.close()
            

        return
    
    


if __name__ == '__main__':
    graph_path = "results_save/"
    
    du = dataset_util(graph_path)
    du.dataset_sample()