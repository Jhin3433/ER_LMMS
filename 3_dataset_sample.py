import networkx as nx    
import os 
import itertools
from random import choices, sample
import pickle
import random
import logging
import datetime
import spacy
from lemminflect import getLemma
import sys
class dataset_util:
    def __init__(self, path, if_other_hyper, if_filter_retain_verb) -> None:

        self.if_other_hyper = if_other_hyper
        self.if_filter_retain_verb = if_filter_retain_verb
        self.save_path = path


        dir_graph_path = "./results_save/" + "2+_event_sense_mapping_graph.gpickle"
        undir_graph_path = "./results_save/" + "2+_undir_event_sense_mapping_graph.gpickle"

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
        self.all_used_synset = set()
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
        self.vocab_id = {}
        self.word_set = set()
        for synset_node in self.synset_nodes:
            if synset_node not in self.dict_synset_to_event:
                self.dict_synset_to_event[synset_node] = []
            adjs_syn_node = self.G[synset_node]
            for event_node in adjs_syn_node:
                
                arg = self.G[synset_node][event_node]["arg"]

                arg_index = event_node.split(" ").index(arg)
                # arg = self.G[synset_node][event_node]["arg"].lower()
                # event_node = event_node.lower()
                # 没加event lemma
                # for word in event_node.split(" "):
                #     self.word_set.add(word)

                # self.dict_synset_to_event[synset_node].append((arg, event_node))
                # if event_node not in self.dict_event_to_synset:
                #     self.dict_event_to_synset[event_node] = []
                # self.dict_event_to_synset[event_node].append((arg, synset_node))

                lemma_event_node, arg = self.lemma_event(event_node, arg_index)


                for word in lemma_event_node.split(" "):
                    self.word_set.add(word)

                self.dict_synset_to_event[synset_node].append((arg, lemma_event_node))
                if lemma_event_node not in self.dict_event_to_synset:
                    self.dict_event_to_synset[lemma_event_node] = []
                self.dict_event_to_synset[lemma_event_node].append((arg, synset_node))



        for i in self.word_set:
            self.vocab_id[i] = len(self.vocab_id) + 1


        # dict_synset_index生成
        # all_verb_synset = set()
        # for synset in self.dict_synset_to_event:

        #     if self.if_filter_retain_verb and synset.find(".v.") == -1:#标志位为True 且synset不是v的时候，continue
        #         continue
        #     else:
        #         all_verb_synset.add(synset)


        save_pickle = []
        save_pickle.append(self.dict_event_to_synset)
        save_pickle.append(self.dict_synset_to_event)
        save_pickle.append(self.vocab_id)
        file = open(self.save_path + "synset_event_graph_info.pickle", 'wb')
        pickle.dump(save_pickle, file)
        file.close()

        logging.info("dict_event_to_synset, dict_synset_to_event, vocab_id are saved well.")

        return
        
        
    def dataset_sample(self, sample_num = 100):
        ''' 只利用wordnet中词性为动词的synset并创造数据集
        '''
        # n = 0
        if os.path.exists(self.save_path + "verb_centric_triple_events.txt"):
            os.remove(self.save_path + "verb_centric_triple_events.txt") 
        
        
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
                    if neg_events_arg == []:#不让落单的neg_event作为负例
                        continue
                    lines = synset_node + " || " + event_positive_pair[0] + "<>" + self.G[synset_node][event_positive_pair[0]]['arg'] + " || " + event_positive_pair[1] + "<>" + self.G[synset_node][event_positive_pair[1]]['arg'] + " || " + neg_events[index] + "<>" + ",".join(neg_events_arg) + "\n"
                    write_lines.append(lines.lower())
                    # print(synset_node + " || " + event_positive_pair[0] + " || " + event_positive_pair[1] + " || " + neg_events[index] + "\n")

                
                # file = open('results_save/write_lines.pickle', 'wb')
                # pickle.dump(write_lines, file)
                # file.close()
                
                with open (self.save_path + "verb_centric_triple_events.txt", mode = "a+", encoding = "utf-8") as f:#需要指定mode和encoding，否则an integer is required (got type str)

                    try:
                        f.writelines(write_lines)
                    except:
                        for write_line in write_lines:
                            logging.info(write_line)
        
        logging.info("Dataset is constructed well.")

        return
    
    



    def hard_dataset_sample(self,sample_num = 100):

        ''' 收取测试集hard.txt中的verb，根据其 利用wordnet中词性为动词的synset并创造数据集。 
        '''
        if os.path.exists(self.save_path + "hard_verb_centric_triple_events.txt"):
            os.remove(self.save_path + "hard_verb_centric_triple_events.txt") 
        

        hard_verb = set()
        for line in open('./hard.txt'): 
            event_arg = line.strip('\n').split('|')
            hard_verb.add(event_arg[1].strip(' '))
            hard_verb.add(event_arg[3].strip(' '))
            hard_verb.add(event_arg[7].strip(' '))
            hard_verb.add(event_arg[10].strip(' '))
        hard_synset = set()
        for synset_node in self.synset_nodes:
            adjs_syn_node = self.G[synset_node]
            for event_node in adjs_syn_node:
                arg = self.G[synset_node][event_node]["arg"].lower()
                if arg not in hard_verb:
                    continue
                else:
                    hard_synset.add(synset_node)

        logging.info("The len of hard_verb is {}".format(len(hard_verb)))
        logging.info("The len of hard_synset is {}".format(len(hard_synset)))

        for synset_node in hard_synset:
            adjs_syn_node = self.G[synset_node]
            event_nodes_adj_for_syn_nodes = list(filter(self.att_event, adjs_syn_node)) 
            if  sample_num < len(event_nodes_adj_for_syn_nodes) :
                sample_events = list(random.sample(event_nodes_adj_for_syn_nodes, k = sample_num))
                event_positive_pairs = list(itertools.combinations(sample_events, 2))
                neg_events = list(random.sample(self.event_nodes, k = len(event_positive_pairs)))
            else:
                event_positive_pairs = list(itertools.combinations(event_nodes_adj_for_syn_nodes, 2))
                neg_events = list(random.sample(self.event_nodes, k = len(event_positive_pairs)))

            assert len(neg_events) == len(event_positive_pairs)
            write_lines = []
            for index, event_positive_pair in enumerate(event_positive_pairs):
                neg_events_arg = []
                for adj_neg_event in nx.all_neighbors(self.G, neg_events[index]):
                    neg_events_arg.append(self.G[adj_neg_event][neg_events[index]]['arg'])
                if neg_events_arg == []:#不让落单的neg_event作为负例
                    continue


                lines = synset_node + " || " + event_positive_pair[0] + "<>" + self.G[synset_node][event_positive_pair[0]]['arg'] + " || " + event_positive_pair[1] + "<>" + self.G[synset_node][event_positive_pair[1]]['arg'] + " || " + neg_events[index] + "<>" + ",".join(neg_events_arg) + "\n"
                write_lines.append(lines.lower())
                # print(synset_node + " || " + event_positive_pair[0] + " || " + event_positive_pair[1] + " || " + neg_events[index] + "\n")

            
            # file = open('results_save/write_lines.pickle', 'wb')
            # pickle.dump(write_lines, file)
            # file.close()
            
            with open (self.save_path + "hard_verb_centric_triple_events.txt", mode = "a+", encoding = "utf-8") as f:#需要指定mode和encoding，否则an integer is required (got type str)

                try:
                    f.writelines(write_lines)
                except:
                    for write_line in write_lines:
                        logging.info(write_line)
    
        logging.info("Dataset is constructed well.")


    def Hyper_Construction(self, init_event, if_other_hyper = False):
        
        """为event创造超图；将raw event按超图处理，保存其对应的synset
        """
        event = init_event #.lower()
        other_event = []
        synsets_list = self.dic_event_to_synset[event]
        other_event.append(event) #event本身应该是doc的一部分

        for arg_synset in synsets_list:
            synset = arg_synset[1]
            self.all_used_synset.add(synset) 


            if if_other_hyper:

                if self.if_filter_retain_verb and synset.find(".v.") == -1:
                    continue
                for arg_event in self.dict_synset_to_event[synset]:
                    other_event.append(arg_event[1])

        if if_other_hyper:
            random.shuffle(other_event)#不用返回值，直接对参数变量进行修改
            other_event = [other_event[0]] + other_event[1:9] if len(other_event) > 10 else other_event


            return ','.join(other_event)
        else:
                return None

    def lemma_event(self, event, arg_index = None):

        arg_list = []
        subj, verb , obj = event.split(" ")
        subj = getLemma(subj, upos = "NOUN")[0]
        verb = getLemma(verb, upos = "VERB")[0]
        obj = getLemma(obj, upos = "NOUN")[0]

        arg_list.append(subj)
        arg_list.append(verb)
        arg_list.append(obj)
        if arg_index == None:
            return ' '.join(arg_list)
        else:
            return ' '.join(arg_list), arg_list[arg_index]


    def hard_hyer_dataset_sample(self, sample_num = 50):
        if os.path.exists(self.save_path + "hard_hyper_verb_centric_triple_events.txt"):
            os.remove(self.save_path + "hard_hyper_verb_centric_triple_events.txt") 
        if os.path.exists(self.save_path + "hard_hyper_events.txt"):
            os.remove(self.save_path + "hard_hyper_events.txt") 

        num = 0
        save_pickle = pickle.load(open(self.save_path + "synset_event_graph_info.pickle","rb"))

        self.dic_event_to_synset = save_pickle[0]
        self.dict_synset_to_event = save_pickle[1]


        hard_verb = set()
        for line in open('./hard.txt'): 
            event_arg = line.strip('\n').split('|')
            hard_verb.add(event_arg[1].strip(' '))
            hard_verb.add(event_arg[4].strip(' '))
            hard_verb.add(event_arg[7].strip(' '))
            hard_verb.add(event_arg[10].strip(' '))
        hard_synset = set()
        for synset_node in self.synset_nodes:
            adjs_syn_node = self.G[synset_node]
            for event_node in adjs_syn_node:
                arg = self.G[synset_node][event_node]["arg"].lower()
                if arg not in hard_verb:
                    continue
                else:
                    hard_synset.add(synset_node)
       
        logging.info("The len of hard_verb is {}".format(len(hard_verb)))
        logging.info("The len of hard_synset is {}".format(len(hard_synset)))

        for synset_node in hard_synset:
            adjs_syn_node = self.G[synset_node]
            event_nodes_adj_for_syn_nodes = list(filter(self.att_event, adjs_syn_node)) 
            if  sample_num < len(event_nodes_adj_for_syn_nodes) :
                sample_events = list(random.sample(event_nodes_adj_for_syn_nodes, k = sample_num))
                event_positive_pairs = list(itertools.combinations(sample_events, 2))
                neg_events = list(random.sample(self.event_nodes, k = len(event_positive_pairs)))
            else:
                event_positive_pairs = list(itertools.combinations(event_nodes_adj_for_syn_nodes, 2))
                neg_events = list(random.sample(self.event_nodes, k = len(event_positive_pairs)))

            assert len(neg_events) == len(event_positive_pairs)
            write_lines = []
            write_lines2 = []
            for index, event_positive_pair in enumerate(event_positive_pairs):
                # neg_events_arg = []
                # for adj_neg_event in nx.all_neighbors(self.G, neg_events[index]):
                    # neg_events_arg.append(self.G[adj_neg_event][neg_events[index]]['arg'])
        
                raw_event = event_positive_pair[0]
                pos_event = event_positive_pair[1]
                neg_event = neg_events[index]

                raw_event_list  = raw_event.split(" ")
                raw_arg = self.G[synset_node][raw_event]["arg"]
                raw_arg_index = raw_event_list.index(raw_arg)

                pos_event_list  = pos_event.split(" ")
                pos_arg = self.G[synset_node][pos_event]["arg"]
                pos_arg_index = pos_event_list.index(pos_arg)


                #增加词型还原
                lemma_raw_event = self.lemma_event(raw_event)
                lemma_pos_event = self.lemma_event(pos_event)
                lemma_neg_event = self.lemma_event(neg_event)

                if lemma_raw_event not in self.dict_event_to_synset or lemma_pos_event not in self.dict_event_to_synset or lemma_neg_event not in self.dict_event_to_synset:
                    continue


                neg_events_arg = []
                for neg_synset_node in nx.all_neighbors(self.G, neg_event):
                    neg_event_list  = neg_event.split(" ")
                    neg_arg = self.G[neg_synset_node][neg_event]["arg"]
                    neg_arg_index = neg_event_list.index(neg_arg)
                    neg_events_arg.append(lemma_neg_event.split(" ")[neg_arg_index])

                if neg_events_arg == []:#不让落单的neg_event作为负例
                    continue
                lines = synset_node + " || " + lemma_raw_event + "<>" + lemma_raw_event.split(" ")[raw_arg_index] + " || " + lemma_pos_event + "<>" + lemma_pos_event.split(" ")[pos_arg_index] + " || " + lemma_neg_event + "<>" + ",".join(neg_events_arg) + "\n"
                write_lines.append(lines)      
                
                # lines = synset_node + " || " + lemma_raw_event + "<>" + self.G[synset_node][raw_event]['arg'] + " || " + lemma_pos_event + "<>" + self.G[synset_node][pos_event]['arg'] + " || " + lemma_neg_event + "<>" + ",".join(neg_events_arg) + "\n"
                # write_lines.append(lines.lower())
                # print(synset_node + " || " + event_positive_pair[0] + " || " + event_positive_pair[1] + " || " + neg_events[index] + "\n")



                raw_hyper = self.Hyper_Construction(lemma_raw_event, self.if_other_hyper)#lemma未改
                pos_hyper = self.Hyper_Construction(lemma_pos_event, self.if_other_hyper)
                neg_hyper = self.Hyper_Construction(lemma_neg_event, self.if_other_hyper)
                if self.if_other_hyper and raw_hyper != None and pos_hyper != None and neg_hyper != None:
                    lines2 = raw_hyper + " || " + pos_hyper + " || " + neg_hyper + "\n"
                    write_lines2.append(lines2.lower())
            
            # file = open('results_save/write_lines.pickle', 'wb')
            # pickle.dump(write_lines, file)
            # file.close()
            
            with open (self.save_path + "hard_hyper_verb_centric_triple_events.txt", mode = "a+", encoding = "utf-8") as f:#需要指定mode和encoding，否则an integer is required (got type str)

                try:
                    f.writelines(write_lines)
                except:
                    for write_line in write_lines:
                        logging.info(write_line)

            if self.if_other_hyper and write_lines2 != []:
                with open (self.save_path + "hard_hyper_events.txt", mode = "a+", encoding = "utf-8") as f:#需要指定mode和encoding，否则an integer is required (got type str)

                    try:
                        f.writelines(write_lines2)
                    except:
                        for write_line in write_lines2:
                            logging.info(write_line)

            # num +=1 
            # if num>2:
            #     break
         


        
        #并非只有动词的synset，raw_event的arg涉及到的synset都有
        self.dict_synset_index = {}
        for i in self.all_used_synset:
            self.dict_synset_index[i] = len(self.dict_synset_index) # 是否需要加1？ 4.8ans:不需要 类比分类中的preprocess.py/keywords_dic, generate_lda.py/keywords_dic

        save_pickle.append(self.dict_synset_index) 



        # 验证
        with open(self.save_path + "hard_hyper_verb_centric_triple_events.txt", "r") as f:
            for line in f:
        
                split_line = line.split("||")
                synset_node = split_line[0].strip(" ")
                # pos_1_event = split_line[1].strip(" ")
                # pos_2_event = split_line[2].strip(" ")
                # neg_3_event = split_line[3].strip("\n").strip(" ")
                # self.Event_Triples.append([synset_node, pos_1_event.lower(), pos_2_event.lower(), neg_3_event.lower()])

    
                # pos_1_event, pos_1_arg = [x.lower() for x in split_line[1].strip(" ").split("<>")]
                pos_1_event, pos_1_arg = split_line[1].strip(" ").split("<>")
                pos_2_event, pos_2_arg = split_line[2].strip(" ").split("<>")
                neg_3_event, neg_3_arg = split_line[3].strip("\n").strip(" ").split("<>")
                try:
                    synset_list = self.dic_event_to_synset[pos_1_event]
                    for arg_synset in synset_list:
                        assert arg_synset[1] in self.dict_synset_index
                    synset_list = self.dic_event_to_synset[pos_2_event]
                    for arg_synset in synset_list:
                        assert arg_synset[1] in self.dict_synset_index
                    synset_list = self.dic_event_to_synset[neg_3_event]
                    for arg_synset in synset_list:
                        assert arg_synset[1] in self.dict_synset_index
                except:
                    logging.error("This Line exist error: {}".format(line), exc_info=sys.exc_info())  # logging.error(traceback.format_exc())

        if os.path.exists(self.save_path + "synset_event_graph_info.pickle"):
            os.remove(self.save_path + "synset_event_graph_info.pickle") 
            file = open(self.save_path + "synset_event_graph_info.pickle", 'wb')
            pickle.dump(save_pickle, file)
            file.close()

        logging.info("Dataset is constructed well.")      
        logging.info("Hyper is constructed well.")      
 

def shuffle_dataset(file_name):
    open_diff = open(file_name, 'r') # 源文本文件
    diff_line = open_diff.readlines()

    line_list = []
    for line in diff_line:
        line_list.append(line)

    random.shuffle(line_list)


    count = len(line_list) # 文件行数
    print('源文件数据行数：',count)
    # 切分diff
    diff_match_split = [line_list[i:i+100000000] for i in range(0,len(line_list),100000000)]# 每个文件的数据行数

    open_diff.close()


    if os.path.exists(file_name):
        os.remove(file_name) 
        # 将切分的写入多个txt中
        for i,j in zip(range(0,int(count/100000000+1)),range(0,int(count/100000000+1))): # 写入txt，计算需要写入的文件数
            with open(file_name, 'w+') as temp:
                for line in diff_match_split[i]:
                    temp.write(line)
        print('拆分后文件的个数：',i+1)
 
        
if __name__ == '__main__':

    if_other_hyper = False # 是否为event创造doc
    if_filter_retain_verb = True #是否过滤非v的synset
    path = "./results_save/no_other_hyper/" #no_other_hyper不为raw event创建doc
    log_path = "./log/" + path.split("/")[-2] + "/"

    logging.basicConfig(filename = log_path + "3_dataset_sample.log", format = '%(asctime)s | %(levelname)s | %(message)s', level = logging.DEBUG, filemode = 'w') #有filename是文件日志输出,filemode是’w’的话，文件会被覆盖之前生成的文件会被覆盖

    start_time = datetime.datetime.now()
    logging.info("The start time is {}".format(start_time))

    du = dataset_util(path, if_other_hyper = if_other_hyper, if_filter_retain_verb = if_filter_retain_verb)
    # du.dataset_sample()
    # du.hard_dataset_sample()
    du.collect_event_synset_dict()
    du.hard_hyer_dataset_sample()

    shuffle_dataset(path + "hard_hyper_verb_centric_triple_events.txt")




    end_time = datetime.datetime.now()
    logging.info("The end time is {}".format(end_time))

    interval = (end_time - start_time).seconds / 3600
    logging.info("The time interval is {} hours".format(interval))



# NOTE
# save_pikcle 保存顺序 :self.dict_event_to_synset, self.dict_synset_to_event, self.vocab_id, self.dict_synset_index
# self.dict_event_to_synset, self.dict_synset_to_event, self.vocab_id 是在整个图上统计的
# self.dict_synset_index可以选择是否设置过滤掉 非v的synset；包含raw_event所涉及的所有synset，其中为v的synset创建doc超图




























    #  def Hyper_Construction(self, init_event, if_other_hyper = False):
        
    #     """为event创造超图；将raw event按超图处理，保存其对应的synset
    #     """
    #     event = init_event.lower()
    #     other_event = []
    #     synsets_list = self.dic_event_to_synset[event]
    

    #     for arg_synset in synsets_list:
    #         synset = arg_synset[1]


    #         # 将raw event按超图处理，保存其对应的synset
    #         if self.G[synset][init_event]['arg'].lower() not in self.arg_related_synset_only_raw_event:
    #             self.arg_related_synset_only_raw_event[self.G[synset][init_event]['arg'].lower()] = synset
    #             self.all_used_synset.add(synset)
    #         else:
    #             assert self.arg_related_synset_only_raw_event[self.G[synset][init_event]['arg'].lower()] == synset


    #         if if_other_hyper:

    #             if self.if_filter_retain_verb and synset.find(".v.") == -1:
    #                 continue
    #             for arg_event in self.dict_synset_to_event[synset]:
    #                 # other_event.append(list(ECL_model.Glove.transform(arg_event[1])))
    #                 other_event.append(arg_event[1])
    #                 arg = arg_event[0]
    #                 if arg not in self.arg_related_synset: #and arg_id != 1:
    #                     self.arg_related_synset[arg] = synset
    #                     self.all_used_synset.add(synset)
    #                 else:
    #                     assert self.arg_related_synset_only_raw_event[self.G[synset][init_event]['arg'].lower()] == synset

    #         else:
    #             return None
    #     if if_other_hyper:
    #         random.shuffle(other_event)#不用返回值，直接对参数变量进行修改
    #         other_event = other_event[0:10] if len(other_event) > 10 else other_event


    #         return ','.join(other_event)

        
    # def hard_hyer_dataset_sample(self, sample_num = 100):
    #     if os.path.exists(self.save_path + "hard_hyper_verb_centric_triple_events.txt"):
    #         os.remove(self.save_path + "hard_hyper_verb_centric_triple_events.txt") 
    #     if os.path.exists(self.save_path + "hard_hyper_events.txt"):
    #         os.remove(self.save_path + "hard_hyper_events.txt") 

    #     num = 0
    #     save_pickle = pickle.load(open(self.save_path + "synset_event_graph_info.pickle","rb"))

    #     self.dic_event_to_synset = save_pickle[0]
    #     self.dict_synset_to_event = save_pickle[1]

    #     self.arg_related_synset = {}
    #     self.arg_related_synset_only_raw_event = {}


    #     hard_verb = set()
    #     for line in open('./hard.txt'): 
    #         event_arg = line.strip('\n').split('|')
    #         hard_verb.add(event_arg[1].strip(' '))
    #         hard_verb.add(event_arg[3].strip(' '))
    #         hard_verb.add(event_arg[7].strip(' '))
    #         hard_verb.add(event_arg[10].strip(' '))
    #     hard_synset = set()
    #     for synset_node in self.synset_nodes:
    #         adjs_syn_node = self.G[synset_node]
    #         for event_node in adjs_syn_node:
    #             arg = self.G[synset_node][event_node]["arg"].lower()
    #             if arg not in hard_verb:
    #                 continue
    #             else:
    #                 hard_synset.add(synset_node)
       
    #     logging.info("The len of hard_verb is {}".format(len(hard_verb)))
    #     logging.info("The len of hard_synset is {}".format(len(hard_synset)))

    #     for synset_node in hard_synset:
    #         adjs_syn_node = self.G[synset_node]
    #         event_nodes_adj_for_syn_nodes = list(filter(self.att_event, adjs_syn_node)) 
    #         if  sample_num < len(event_nodes_adj_for_syn_nodes) :
    #             sample_events = list(random.sample(event_nodes_adj_for_syn_nodes, k = sample_num))
    #             event_positive_pairs = list(itertools.combinations(sample_events, 2))
    #             neg_events = list(random.sample(self.event_nodes, k = len(event_positive_pairs)))
    #         else:
    #             event_positive_pairs = list(itertools.combinations(event_nodes_adj_for_syn_nodes, 2))
    #             neg_events = list(random.sample(self.event_nodes, k = len(event_positive_pairs)))

    #         assert len(neg_events) == len(event_positive_pairs)
    #         write_lines = []
    #         write_lines2 = []
    #         for index, event_positive_pair in enumerate(event_positive_pairs):
    #             neg_events_arg = []
    #             for adj_neg_event in nx.all_neighbors(self.G, neg_events[index]):
    #                 neg_events_arg.append(self.G[adj_neg_event][neg_events[index]]['arg'])
    #             if neg_events_arg == []:#不让落单的neg_event作为负例
    #                 continue
                

    #             raw_event = event_positive_pair[0]
    #             pos_event = event_positive_pair[1]
    #             neg_event = neg_events[index]
    #             raw_hyper = self.Hyper_Construction(raw_event, self.if_other_hyper)
    #             pos_hyper = self.Hyper_Construction(pos_event, self.if_other_hyper)
    #             neg_hyper = self.Hyper_Construction(neg_event, self.if_other_hyper)
                
                
    #             lines = synset_node + " || " + raw_event + "<>" + self.G[synset_node][raw_event]['arg'] + " || " + pos_event + "<>" + self.G[synset_node][pos_event]['arg'] + " || " + neg_event + "<>" + ",".join(neg_events_arg) + "\n"
    #             write_lines.append(lines.lower())
    #             # print(synset_node + " || " + event_positive_pair[0] + " || " + event_positive_pair[1] + " || " + neg_events[index] + "\n")

    #             if self.if_other_hyper and raw_hyper != None and pos_hyper != None and neg_hyper != None:
    #                 lines2 = raw_hyper + " || " + pos_hyper + " || " + neg_hyper + "\n"
    #                 write_lines2.append(lines2.lower())
            
    #         # file = open('results_save/write_lines.pickle', 'wb')
    #         # pickle.dump(write_lines, file)
    #         # file.close()
            
    #         with open (self.save_path + "hard_hyper_verb_centric_triple_events.txt", mode = "a+", encoding = "utf-8") as f:#需要指定mode和encoding，否则an integer is required (got type str)

    #             try:
    #                 f.writelines(write_lines)
    #             except:
    #                 for write_line in write_lines:
    #                     logging.info(write_line)

    #         if self.if_other_hyper and write_lines2 != []:
    #             with open (self.save_path + "hard_hyper_events.txt", mode = "a+", encoding = "utf-8") as f:#需要指定mode和encoding，否则an integer is required (got type str)

    #                 try:
    #                     f.writelines(write_lines2)
    #                 except:
    #                     for write_line in write_lines2:
    #                         logging.info(write_line)

    #         num +=1 
    #         if num>2:
    #             break
         


        
    #     #并非只有动词的synset，raw_event的arg涉及到的synset都有
    #     self.dict_synset_index = {}
    #     for i in self.all_used_synset:
    #         self.dict_synset_index[i] = len(self.dict_synset_index) # 是否需要加1？ 4.8ans:不需要 类比分类中的preprocess.py/keywords_dic, generate_lda.py/keywords_dic
    #     for arg in self.arg_related_synset_only_raw_event:
    #         self.arg_related_synset_only_raw_event[arg] = self.dict_synset_index(self.arg_related_synset_only_raw_event[arg])
    #     for arg in self.arg_related_synset:
    #         self.arg_related_synset[arg] = self.dict_synset_index(self.arg_related_synset[arg])

    #     save_pickle.append(self.dict_synset_index) 
    #     save_pickle.append(self.arg_related_synset_only_raw_event)
    #     save_pickle.append(self.arg_related_synset)#为raw_event创建的doc所涉及的synset
        

    #     if os.path.exists(self.save_path + "synset_event_graph_info.pickle"):
    #         os.remove(self.save_path + "synset_event_graph_info.pickle") 
    #         file = open(self.save_path + "synset_event_graph_info+arg_related_synset.pickle", 'wb')
    #         pickle.dump(save_pickle, file)
    #         file.close()

    #     logging.info("Dataset is constructed well.")      
    #     logging.info("Hyper is constructed well.")      
 