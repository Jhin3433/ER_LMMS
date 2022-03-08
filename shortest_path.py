
import networkx as nx 
import pickle

def shortest_path(undir_graph_path):
    udrtG = nx.read_gpickle(undir_graph_path)
    shortest_path = dict(nx.shortest_path_length(udrtG)) 
    file = open('results_save/shortest_path.pickle', 'wb')
    pickle.dump(shortest_path, file)
    file.close()
if __name__ == '__main__':
    graph_path = "results_save/"
    undir_graph_path = graph_path + "2+_undir_event_sense_mapping_graph.gpickle"

    du = shortest_path(undir_graph_path)