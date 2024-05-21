import numpy as np
import networkx as nx
from random import random
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from multiprocessing import Process
import multiprocessing as mp

from tools import LOAD, apply_partition
import Partitions as P
import Visions as V

'''
This class lets you track entropy over time in a graph that evolves according to a given recommendation semantic
tracks entropy given a partition and a vision function
currently only supports baseline recommendation semantics and not diverse
supposedly multiprocesses the recomemndations but unsure whether it actually does
'''
class Evolution:
    def __init__(self, graph_name, entropy, nb_adds, nb_remove=1, partitioning = P.random_partition, vision=V.biased_PPR_vision, semantic='common', abandon_chance = 1, subset = False):
        self.data_name = graph_name
        self.data = LOAD(graph_name)
        self.pos = nx.spring_layout(self.data)
        
        self.partition_function = partitioning
        self.partition = self.partition_function(self.data)
        apply_partition(self.data, self.partition)

        self.nb_adds = nb_adds
        self.nb_remove = nb_remove
        self.nb_parts = 1
        if abandon_chance > 0 : self.init_history()

        self.abandon_chance = abandon_chance
        self.vision = vision
        self.semantic = semantic
        self.todistr = vision_distr[vision]
        self.entropy = entropy

        self.subset = subset
        self.time = 0
        self.init_record()

        if self.vision == V.PPR_vision:
            self.walk_database = {node:V.walk_segment(self.data,node) for node in self.data.nodes}
    
    def init_history(self):
        nx.set_node_attributes(self.data, {node: list(self.data[node]) for node in self.data.nodes}, "history")
    
    def init_record(self):
        self.record = {node: [self.entropy(self.todistr(self.data, self.vision(self.data,node)))] for node in random.sample(list(self.data.nodes), int(len(self.data.nodes)*0.10))}
    
    def record_entry(self):
        for k in self.record.keys():
            self.record[k].append(self.entropy(self.todistr(self.data, self.vision(self.data,k))))

    def use_vision(self,new_vision):
        self.vision = new_vision
        self.todistr = vision_distr[self.vision]

    def use_partition_function(self,new_partition_function):
        self.partition_function = new_partition_function
        self.partition = new_partition_function(self.data)
        global number_of_parts 
        number_of_parts = len(self.partition)
        apply_partition(self.data,self.partition)
    
    def use_partition(self, new_partition):
        apply_partition(self.data, new_partition)
    
    def use_semantic(self,new_semantic):
        self.semantic = new_semantic



    def reset(self):
        self.data = LOAD(self.data_name)
        apply_partition(self.data, self.partition)
        self.record = {node: [self.record[node][0]] for node in self.record.keys()}
        self.time = 0


    def plot_entropy(self):
        X = range(0,self.time+1)
        plt.figure()
        Y = [0 for _ in range(self.time+1)]
        for node,values in self.record.items():
            for i in range(len(values)):
                Y[i] += values[i]
        plt.plot(X,[y/20 for y in Y])
        plt.title(f'entropy in {self.data_name} at epoch {self.time} with {self.semantic} semantic')
        plt.xlabel('time')
        plt.ylabel('entropy')
        
    def draw(self,repos = False):
        plt.figure()
        if repos:
            p = nx.spring_layout(self.data)
        else:
            p = self.pos
        colors = []
        for node in self.data.nodes:
            colors.append(self.data.nodes[node]['partition'])
        nx.draw(self.data,pos = p, node_color=colors,with_labels=True)

    def epoch(self, nb, semantic = None):

        for _ in tqdm(range(nb), leave=False):
            processes = []
            recdic = mp.Manager().dict()
            
            adhd = []
            fordhd = []

            if self.subset:
                target_list = [k for k in self.record.keys()]
            else:
                target_list = self.data.nodes

            for target in tqdm(target_list, leave=False):
                nb = int(np.min([self.data.degree[target],self.nb_adds]))
                
                candidates = [component for component in nx.connected_components(self.data) if target in component][0]

                p = Process(target=self.recommend, args=(target, candidates, nb, recdic))
                processes.append(p)
                p.start()
                
                
                forgotten = self.forget(target)
                i = 0
                if self.data.degree[target] > self.nb_remove:
                    i = self.nb_remove
                else:
                    i = self.data.degree[target]
                
                for node in forgotten[:i]:
                    fordhd.append([target,node])

            for j in range(len(processes)):
                processes[j].join()

            trabajo = dict(recdic)
            for target in trabajo.keys():
                temp = [ [k,v] for k,v in trabajo[target].items()]
                temp.sort(key = lambda x: x[1], reverse = True)
                adds = temp[:self.nb_adds]
                #print(target, adds)
                
                [adhd.append([target,othernode]) for othernode,_ in adds]

            
            #print('------ \n',fordhd)
            self.data.add_edges_from(adhd)
            for u,v in fordhd:
                if self.data.degree[u] > 1 and self.data.degree[v] > 1 and [u,v] in self.data.edges:
                    self.data.remove_edge(u,v)
            
            self.time += 1
            self.record_entry()

            """
            for node in self.data.nodes:
                rec = self.recommend(node, self.data.nodes, self.nb_adds)
                forgotten = []
                if np.random.random_sample() < self.abandon_chance:
                    forgotten = self.forget(node)
                
                self.data.nodes[node]['history'] = self.data.nodes[node]['history'][len(forgotten):] + rec
                [self.data.remove_edge(node, othernode) for othernode in forgotten if [node,othernode] in self.data.edges]
                [self.data.add_edge(node,othernode) for othernode in rec]
            
            self.time += 1
            self.record_entry()

            if self.time % 20:from IPython.display import clear_output
                self.repartition()
            """

    def forget(self, node):
        forgotten = []

        if self.semantic in baseline_functions:
            temp = [
                    [c, self.semantic(self.data,node,c)] 
                        for c in self.data.neighbors(node) 
                            #if self.data.nodes()[c]['partition'] != self.data.nodes()[node]['partition']
                    ]
            temp.sort(key = lambda x: x[1], reverse = False)
            temp.sort(key = lambda x: nx.shortest_path_length(self.data, node, x[0]), reverse=True)
            forgotten = [x for x,_ in temp]

        return forgotten

    def recommend(self, target, candidates, nb_adds, dictionary):
        adds = []
        
        if self.semantic in baseline_functions:
            temp = [[c, self.semantic(self.data,target,c)] for c in candidates if c not in list(self.data.neighbors(target)) and c != target]
            temp.sort(key = lambda x: (x[1], self.data.nodes[x[0]]['partition'] == self.data.nodes[target]['partition'], nx.shortest_path_length(self.data,target,x[0])))
            #temp.sort(key = lambda x: self.data.nodes[x[0]]['partition'] - self.data.nodes[target]['partition'])
            #temp.sort(key = lambda x: nx.shortest_path_length(self.data,target,x[0]))
            adds = [(x,y) for x,y in temp[-nb_adds:]]
        
        for k,v in adds:
            if k != target:
                if target in dictionary:
                    temp = dictionary[target]
                    temp[k] = v
                    dictionary[target] = temp
                else:
                    dictionary[target] = dict({k:v})
