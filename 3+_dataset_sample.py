import networkx as nx    
import os 
import itertools
from random import choices, sample
import pickle
import random
import logging
import datetime
from lemminflect import getLemma
import pdb 
from glove_utils import Glove
from nltk.corpus import wordnet as wn
# 将事件都进行 词型还原，不为event创造doc
# 参数
# path = "./results_save/no_other_hyper/" #no_other_hyper不为raw event创建doc
# log_path = "./log/" + path.split("/")[-2] + "/"
# sample_max_num = 5


class dataset_util:
    def __init__(self, path, if_other_hyper, if_filter_retain_verb) -> None:

        self.if_other_hyper = if_other_hyper
        self.if_filter_retain_verb = if_filter_retain_verb
        self.save_path = path
        self.if_lower = True


        self.Glove_file = Glove('./results_save/glove.6B.100d.ext.txt')
        dir_graph_path = "./results_save/" + "2+_event_sense_mapping_graph.gpickle"
        undir_graph_path = "./results_save/" + "2+_undir_event_sense_mapping_graph.gpickle"

        if os.path.exists(dir_graph_path) and os.path.exists(undir_graph_path):
            self.G = nx.read_gpickle(dir_graph_path)
        self.synset_nodes, self.event_nodes = self.synset_node_identify()
        
        self.dict_synset_to_event = {}
        self.dict_event_to_synset = {}
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




    def lemma_event(self, event, arg_index, if_lower = True):

        if_lower = self.if_lower

        arg_list = []
        subj, verb , obj = event.split(" ")
        subj = getLemma(subj, upos = "NOUN")[0]
        verb = getLemma(verb, upos = "VERB")[0]
        obj = getLemma(obj, upos = "NOUN")[0]

        arg_list.append(subj)
        arg_list.append(verb)
        arg_list.append(obj)

        if if_lower:
            return ' '.join(arg_list).lower(), arg_list[arg_index].lower()

    #带词形还原
    def verb_centric_collect_event_synset_dict_and_data_sample(self, sample_max_num = 5):
        self.dict_synset_to_event = {}
        self.dict_event_to_synset = {}
        self.vocab_id = {}
        self.word_set = set()

        # 统计验证集涉及到的verb 
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

    
        writelines = []
        for synset_node in hard_synset:

            line = []
            if synset_node not in self.dict_synset_to_event:
                self.dict_synset_to_event[synset_node] = []

            adjs_syn_node = self.G[synset_node]
            for event_node in adjs_syn_node:
                flag = True
                arg = self.G[synset_node][event_node]["arg"]
                arg_index = event_node.split(" ").index(arg)
                lemma_event_node, arg = self.lemma_event(event_node, arg_index)

                for word in lemma_event_node.split(" "):
                    if word == "" or word not in self.Glove_file.vocab_id:
                        flag = False
                        continue
                
                
                if flag:
                    for word in lemma_event_node.split(" "):
                        self.word_set.add(word)
                    self.dict_synset_to_event[synset_node].append((arg, lemma_event_node))
                    if lemma_event_node not in self.dict_event_to_synset:
                        self.dict_event_to_synset[lemma_event_node] = []
                    self.dict_event_to_synset[lemma_event_node].append((arg, synset_node))

                    line.append(lemma_event_node + "<>" + arg)    


            if  sample_max_num < len(line) :
                sample_events = list(random.sample(line, k = sample_max_num))
                event_positive_pairs = list(itertools.combinations(sample_events, 2))
            else:
                event_positive_pairs = list(itertools.combinations(line, 2))



            neg_events = []
            while len(neg_events) != len(event_positive_pairs):
                all_synset_nodes = []
                all_synset_nodes += self.synset_nodes #不可变性
                assert len(all_synset_nodes) == 44898
                all_synset_nodes.remove(synset_node)#remove返回值为None
                neg_synset = list(random.sample(all_synset_nodes, k = 1))[0]
           
                flag = True
                adjs_syn_node = self.G[neg_synset]
                neg_event = list(random.sample(list(adjs_syn_node), k = 1))[0]
                arg = self.G[neg_synset][neg_event]["arg"]
                arg_index = neg_event.split(" ").index(arg)
                lemma_event_node, arg = self.lemma_event(neg_event, arg_index)

                for word in lemma_event_node.split(" "):
                    if word == "" or word not in self.Glove_file.vocab_id:
                        flag = False
                        continue
                
                if flag:
                    for word in lemma_event_node.split(" "):
                        self.word_set.add(word)
                    self.dict_synset_to_event[synset_node].append((arg, lemma_event_node))
                    if lemma_event_node not in self.dict_event_to_synset:
                        self.dict_event_to_synset[lemma_event_node] = []
                    self.dict_event_to_synset[lemma_event_node].append((arg, synset_node))

                    neg_events.append(lemma_event_node + "<>" + arg)

            assert len(event_positive_pairs) == len(neg_events)
            for index in range(len(event_positive_pairs)):
                raw_event = event_positive_pairs[index][0]
                pos_event = event_positive_pairs[index][1]
                writelines.append(synset_node + " || " + raw_event + " || " + pos_event + " || " + neg_events[index] + "\n" )#neg_events[index] )

            if len(writelines) > 500:
                with open ('./results_save/no_other_hyper/hard_hyper_verb_centric_triple_events.txt', mode = "a+", encoding = "utf-8") as f:#需要指定mode和encoding，否则an integer is required (got type str)
                    f.writelines(writelines)
                writelines = []
 
        if len(writelines) != []:
            with open ('./results_save/no_other_hyper/hard_hyper_verb_centric_triple_events.txt', mode = "a+", encoding = "utf-8") as f:#需要指定mode和encoding，否则an integer is required (got type str)
                f.writelines(writelines)




        save_pickle = []
        save_pickle.append(self.dict_event_to_synset)
        save_pickle.append(self.dict_synset_to_event)

        for i in self.word_set:
            self.vocab_id[i] = len(self.vocab_id) + 1
        save_pickle.append(self.vocab_id)


        #并非只有动词的synset，raw_event的arg涉及到的synset都有
        for synset in self.dict_synset_to_event:
            self.all_used_synset.add(synset) 
        self.dict_synset_index = {}
        for i in self.all_used_synset:
            self.dict_synset_index[i] = len(self.dict_synset_index)

        save_pickle.append(self.dict_synset_index) 
        file = open(self.save_path + "synset_event_graph_info.pickle", 'wb')
        pickle.dump(save_pickle, file)
        file.close()

        logging.info("dict_event_to_synset, dict_synset_to_event, vocab_id are saved well.")

        return



    #想根据路径取样，没完成
    def collect_event_synset_dict_and_data_sample(self, sample_max_num = 5, max_path_length = 2):#不包含当前synset
        self.dict_synset_to_event = {}
        self.dict_event_to_synset = {}



        if os.path.exists("./results_save/hard_related.pickle"):
            hard_related = pickle.load(open("./results_save/hard_related.pickle","rb"))
            noun_verb = hard_related[0]
            hard_synset = hard_related[1]
            del hard_related
        else:

            # 统计验证集涉及到的verb 、noun
            noun_verb = set()
            for line in open('./hard.txt'): 
                event_arg = line.strip('\n').split('|')
                noun_verb.add(event_arg[0].strip(' '))
                noun_verb.add(event_arg[3].strip(' '))
                noun_verb.add(event_arg[6].strip(' '))
                noun_verb.add(event_arg[9].strip(' '))

                noun_verb.add(event_arg[1].strip(' '))
                noun_verb.add(event_arg[4].strip(' '))
                noun_verb.add(event_arg[7].strip(' '))
                noun_verb.add(event_arg[10].strip(' '))

                noun_verb.add(event_arg[2].strip(' '))
                noun_verb.add(event_arg[5].strip(' '))
                noun_verb.add(event_arg[8].strip(' '))
                noun_verb.add(event_arg[11].strip(' '))
            hard_synset = set()
            for synset_node in self.synset_nodes:
                adjs_syn_node = self.G[synset_node]
                for event_node in adjs_syn_node:
                    arg = self.G[synset_node][event_node]["arg"].lower()
                    if arg not in noun_verb:
                        continue
                    else:
                        hard_synset.add(synset_node)
       
        logging.info("The len of hard_verb is {}".format(len(noun_verb))) #482
        logging.info("The len of hard_synset is {}".format(len(hard_synset))) #2261

        # self.synset_path = {}
        writelines = []
        for synset_node in hard_synset:
            synset = wn.synset(synset_node)
            if synset_node not in self.dict_synset_to_event:
                self.dict_synset_to_event[synset_node] = []

            
            all_hyperpath_event_in_Glove = []
            for hyper_path in synset.hypernym_paths():

                if len(hyper_path) < max_path_length + 1:
                    logging.info("The length of hyper path is short.-- {}".format(" ".join([x.name() for x in hyper_path])))
                    continue

                partial_hyper_path = hyper_path[-1:-5:-1]
                for synset in partial_hyper_path:
                    if len(list(self.G[synset_node])) == 0:
                        logging.info("The {} has no event node.-- {}".format(synset))


                    #pass 验证有的synset没在sel.synset_nodes里


                # 统计上位词synset对应的event_node
                hyperpath_event_in_Glove = {}
                for index, hyper in enumerate(partial_hyper_path):
                    hyperpath_event_in_Glove[index] = [hyper.name()]  #第一个上位词距离原synset为1
                    for event_node in self.G[hyper.name()]:
                        flag = True
                        for word in event_node.split(" "):
                            if word not in self.Glove_file.vocab_id:
                                flag = False
                                break
                        if flag:
                            hyperpath_event_in_Glove[index].append(event_node)   

                all_hyperpath_event_in_Glove.append(hyperpath_event_in_Glove)

            # 对event_node取样，然后构造 self.dict_synset_to_event和 self.dict_event_to_synset
            for hyperpath_event_in_Glove in all_hyperpath_event_in_Glove:
                for synset_num in hyperpath_event_in_Glove:
                    hyperpath_event_in_Glove[synset_num] = hyperpath_event_in_Glove[synset_num][0] + list(random.sample(hyperpath_event_in_Glove[synset_num][1:], k = sample_max_num))
                for synset_num in hyperpath_event_in_Glove:
                    for event_node in hyperpath_event_in_Glove[synset_num]:
                        if event_node == hyperpath_event_in_Glove[synset_num][0] :
                            continue
                        arg = self.G[synset_node][event_node]["arg"]
                        self.dict_synset_to_event[synset_node].append((arg, event_node))
                        if event_node not in self.dict_event_to_synset:
                            self.dict_event_to_synset[event_node] = []
                        self.dict_event_to_synset[event_node].append((arg, synset_node))     

                
                event_positive_triples = itertools.product(hyperpath_event_in_Glove[0][1:], hyperpath_event_in_Glove[1][1:], hyperpath_event_in_Glove[0][2:])
                for index in range(len(event_positive_triples)):
                    raw_event = event_positive_triples[index][0]
                    pos_event = event_positive_triples[index][1]
                    neg_event = event_positive_triples[index][2]
                    writelines.append(hyperpath_event_in_Glove[0][0] + "<>" + raw_event + "<>" + str(0) + " || " + hyperpath_event_in_Glove[1][0] + "<>" + pos_event + "<>" + str(1) + " || " + hyperpath_event_in_Glove[2][0] + "<>" + neg_event + "<>" + str(2) + "\n" )



                if len(writelines) > 100:
                    with open ('./results_save/no_other_hyper/hard_hyper_triple_events.txt', mode = "a+", encoding = "utf-8") as f:#需要指定mode和encoding，否则an integer is required (got type str)
                        f.writelines(writelines)
                    writelines = []
 
        if len(writelines) != []:
            with open ('./results_save/no_other_hyper/hard_hyper_triple_events.txt', mode = "a+", encoding = "utf-8") as f:#需要指定mode和encoding，否则an integer is required (got type str)
                f.writelines(writelines)




        save_pickle = []
        save_pickle.append(self.dict_event_to_synset)
        save_pickle.append(self.dict_synset_to_event)


        #并非只有动词的synset，raw_event的arg涉及到的synset都有
        for synset in self.dict_synset_to_event:
            self.all_used_synset.add(synset) 
        self.dict_synset_index = {}
        for i in self.all_used_synset:
            self.dict_synset_index[i] = len(self.dict_synset_index)

        save_pickle.append(self.dict_synset_index) 
        file = open(self.save_path + "synset_event_graph_info.pickle", 'wb')
        pickle.dump(save_pickle, file)
        file.close()

        logging.info("dict_event_to_synset, dict_synset_to_event, dict_synset_index are saved well.")

        return





def shuffle_dataset(file_name):

    line_list = []
    with open(file_name, 'r') as open_diff:  # 源文本文件
        diff_line = open_diff.readlines()
        
        for line in diff_line:
            line_list.append(line)
        count = len(line_list) # 文件行数
        print('源文件数据行数：',count)
        open_diff.close()
    
    random.shuffle(line_list)

    # 切分diff
    diff_match_split = [line_list[i:i+100000000] for i in range(0,len(line_list),100000000)]# 每个文件的数据行数

    
    if os.path.exists(file_name):
        # os.remove(file_name) 
        # 将切分的写入多个txt中
        for i,j in zip(range(0,int(count/100000000+1)),range(0,int(count/100000000+1))): # 写入txt，计算需要写入的文件数
            # with open(file_name[:-4] + "_shuffle" + ".txt", 'w+') as temp:
            with open(file_name, 'w+') as temp:
                for line in diff_match_split[i]:
                    temp.write(line)
        print('拆分后文件的个数：',i+1)




        
if __name__ == '__main__':

    if_other_hyper = False # 是否为event创造doc
    if_filter_retain_verb = True #是否过滤非v的synset
    path = "./results_save/no_other_hyper/" #no_other_hyper不为raw event创建doc
    # log_path = "./log/" + path.split("/")[-2] + "/"
    log_path = "./log/3+_dataset_sample.log"#!!!!!!!路径中不要有多个_下划线
    sample_max_num = 5


    logging.basicConfig(filename = log_path, format = '%(asctime)s | %(levelname)s | %(message)s', level = logging.INFO, filemode = 'w') #有filename是文件日志输出,filemode是’w’的话，文件会被覆盖之前生成的文件会被覆盖

    start_time = datetime.datetime.now()
    logging.info("The start time is {}".format(start_time))

    du = dataset_util(path, if_other_hyper = if_other_hyper, if_filter_retain_verb = if_filter_retain_verb)
    logging.info("加载模型成功")
    du.collect_event_synset_dict_and_data_sample(sample_max_num = 5, max_path_length = 3)

    # shuffle_dataset(path + "hard_hyper_verb_centric_triple_events.txt")
