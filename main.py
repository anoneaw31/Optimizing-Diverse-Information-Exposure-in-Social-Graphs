import numpy as np
import networkx as nx
import pickle
import time
import datetime
import sys

import Functions.tools as ft
import Functions.Partitions as fp
import Functions.Matrix as fm
from Functions.Matrix import UniformQ, PreferentialQ
import Functions.Recommenders as fr
LOAD = ft.LOAD


partitions = [fp.louvain_communities_community_partition, 'ground-truth', 'continuous']


Wfunctions = [fm.WBFS, fm.WPPR, fm.WFJ]
strategies = ['greedy', 'PB', 'convexity'] #
Partition_strategies = ['MPB'] #, 'RMPB', 'BMPB'
Candidate_strategies = ['all'] #, 'ssbr', 'ssdb', 'sstb'
targetQs = [UniformQ, PreferentialQ]
models = fr.models


data = str(sys.argv[1])
n = int(sys.argv[2])
k = int(sys.argv[3])
if len(sys.argv) > 4:
    attr_name = str(sys.argv[4])
else:
    attr_name = ''
    partitions = partitions[:-1]

print(data, n, k, datetime.datetime.now())

if __name__ == '__main__':
    
        graph = LOAD(data)
        targets = fr.selectTargets(graph,n)
        print(targets)
        savedictionary = {'dataname':data, 'k':k, 'targets':targets}

        for partition in partitions:

            if partition == 'ground-truth':
                if 'ground-truth' in graph.nodes[0]:
                    print(partition)
                    name = partition
                else:
                    continue
            elif partition == 'continuous':
                if attr_name in graph.nodes[0]:
                    print(partition)
                    name = partition
                else:
                    continue
            else:
                print(partition.__name__)
                name = partition.__name__

            if partition == 'ground-truth' and 'ground-truth' in graph.nodes[0]:
                computed_partition = {}
                for node in graph.nodes(data=True):
                    if node[1]['ground-truth'] in computed_partition.keys():
                        computed_partition[node[1]['ground-truth']].append(node[0])
                    else:
                        computed_partition[node[1]['ground-truth']] = [node[0]]
                savedictionary[name] = {'partition':computed_partition}
                P = fm.partition_matrix(graph, computed_partition)
            elif partition == 'continuous':
                savedictionary[name] = {'partition':computed_partition}
                P = fm.continuous_P(graph, attr_name=attr_name)
            else:
                computed_partition = partition(graph)
                savedictionary[name] = {'partition': computed_partition}
                P = fm.partition_matrix(graph, computed_partition)

            for recommender in models:
                print(f'\t {recommender} : {datetime.datetime.now()}')
                if recommender.__name__ == 'recommendMain':
                    l = []
                    for Wfunction in Wfunctions:
                        print(f'\t\t {Wfunction}')
                        for Obj_Distr in targetQs:
                            print(f'\t\t\t {Obj_Distr.__name__}')
                            for strategy in strategies:
                                print(f'\t\t\t\t {strategy} : {datetime.datetime.now()}')
                                if strategy == 'PB':
                                    for Pstrategy in Partition_strategies:
                                        for Cstrategy in Candidate_strategies:
                                            strategy = Pstrategy + '-' + Cstrategy
                                            print(f'\t\t\t\t\t {strategy} : {datetime.datetime.now()}')
                                            recommendations = {u:[] for u in targets}
                                            times = {u:[] for u in targets}
                                            objs = {u:[] for u in targets}
                                            for u in targets:
                                                targetQ = Obj_Distr(P,fm.WBFS(nx.adjacency_matrix(graph), []), u)
                                                start = time.time()
                                                l = recommender(graph, u, k, Wfunction, P, strategy=strategy, targetQ=targetQ)
                                                end = time.time()
                                                recommendations[u] = l
                                                times[u] = end - start
                                                objs[u] = targetQ
                                            savedictionary[name][f'{Wfunction.__name__} {strategy} {Obj_Distr.__name__}'] = [recommendations,times,objs]
                                else:
                                    recommendations = {u:[] for u in targets}
                                    times = {u:[] for u in targets}
                                    objs = {u:[] for u in targets}
                                    for u in targets:
                                        targetQ = Obj_Distr(P,fm.WBFS(nx.adjacency_matrix(graph), []), u)
                                        start = time.time()
                                        l = recommender(graph, u, k, Wfunction, P, strategy=strategy, targetQ=targetQ)
                                        end = time.time()
                                        recommendations[u] = l
                                        times[u] = end - start
                                        objs[u] = targetQ
                                        savedictionary[name][f'{Wfunction.__name__} {strategy} {Obj_Distr.__name__}'] = [recommendations,times,objs]
                        #endfor strategy
                    #endfor Wfunction
                else:
                    recommendations = {u:[] for u in targets}
                    times = {u:[] for u in targets}
                    for u in targets:
                        start = time.time()
                        l = recommender(graph, u, k)
                        end = time.time()
                        recommendations[u] = l
                        times[u] = end - start
                        savedictionary[name][f'{recommender.__name__}'] = [recommendations,times]
                    
            #endfor recommender
        #endfor partition
        file_name = f'./Saves/{data}n{n}k{k}'
        with open(file_name+'.pkl','wb+') as f:
            pickle.dump(savedictionary, f)
        print('ended at : ',datetime.datetime.now())