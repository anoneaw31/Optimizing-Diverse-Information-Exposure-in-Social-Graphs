import numpy as np
import networkx as nx
import pickle
import time
import datetime
import os
import sys
import argparse
from tqdm import tqdm

import Functions.tools as ft
import Functions.Partitions as fp
import Functions.Matrix as fm
from Functions.Matrix import UniformQ, PreferentialQ
import Functions.Recommenders as fr
LOAD = ft.LOAD

parser = argparse.ArgumentParser(description='Description of your script')
parser.add_argument('--data', type=str, help='data name')
parser.add_argument('--p', type=str, help='partition')
parser.add_argument('--w', type=str, help='Wfunction')
parser.add_argument('--s', type=str, help='strategy')
parser.add_argument('--q', type=str, help='target distribution')
parser.add_argument('--m', type=str, help='model')
parser.add_argument('--c', type=str, help='continuous partition attribute name')

"""

python mainone.py --data polbooks --p louvain --w WPPR --m recommendMain --s convexity --q pref --c genre

--data  : ml-100k, polbooks
--p     : louvain, gr, c
--w     : WBFS, WPPR, WFJ
--m     : recommendMain, GlobalDemaine, SingleSourceDemaine, Neurips, common_neighbors_recommendation, EffectEstimation, random_recommendation
--s     : greedy, convexity
            MPB, RMPB, BMPB
                all, ssbr, ssdb, sstb
--q     : uni, pref
--c     : genre(ml-100k) 

"""
#strategies = ['greedy', 'PB', 'convexity']
#Partition_strategies = ['MPB', 'RMPB', 'BMPB']
#Candidate_strategies = ['all', 'ssbr', 'ssdb', 'sstb']
#targetQs = [UniformQ, PreferentialQ]



if __name__ == "__main__":

    args = parser.parse_args()

    dataname = args.data
    Wfunction = getattr(fm, args.w)
    model = getattr(fr, args.m)
    strategy = args.s
    k = 50

    graph = LOAD(dataname)
    file_name = dataname
    savedictionary = {'dataname':dataname, 'k':k}

    if args.p == 'louvain':
        partitioname = fp.louvain_communities_community_partition.__name__
    elif args.p == 'gr':
        partitioname = 'ground-truth'
    elif args.p == 'c' and args.c is not None:
        partitioname = 'continuous-'+args.c

    if args.q == 'uni':
        Qfunction = fr.UniformQ
    elif args.q == 'pref':
        Qfunction = fr.PreferentialQ   

    if file_name+'.pkl' in os.listdir('./Saves/'):
        print('found existing logs')
        with open(f'./Saves/{file_name}.pkl','rb') as f:
            baseDictionary = pickle.load(f)
            if 'targets' in baseDictionary.keys():
                targets = baseDictionary['targets']
            else:
                targets = fr.selectTargets(graph, 30)
            
            if partitioname in baseDictionary.keys():
                partition = baseDictionary[partitioname]['partition']
                if args.p == 'c' and args.c is not None:
                    P = fm.continuous_P(graph,args.c)
                else:
                    P = fm.partition_matrix(graph,partition)
            else:
                partition, P = fm.produce_partition(graph,partitioname,args.c)
        savedictionary['targets'] = targets
        savedictionary[partitioname] = {'partition': partition}
    else:
        print('no logs found')
        targets = fr.selectTargets(graph, 30)
        partition, P = fm.produce_partition(graph,partitioname,args.c)
        savedictionary['targets'] = targets
        savedictionary[partitioname] = {'partition': partition}
        with open(f'./Saves/{file_name}.pkl','wb+') as f:
            pickle.dump(savedictionary,f)

    print(dataname, Wfunction.__name__, model.__name__, strategy, k, Qfunction.__name__, partitioname)
    print(f'Starting at {datetime.datetime.now()}')
    #==================================================================================================
    #==================================================================================================
    #==================================================================================================

    if model.__name__ == 'recommendMain':
        recommendations = {u:[] for u in targets}
        times = {u:[] for u in targets}
        objs = {u:[] for u in targets}

        experiment_name = f'{Wfunction.__name__} {strategy} {Qfunction.__name__}'

        for u in tqdm(targets):
            targetQ = Qfunction(P,Wfunction(nx.adjacency_matrix(graph), []), u)
            start = time.time()
            l = model(graph, u, k, Wfunction, P, strategy=strategy, targetQ=targetQ)
            end = time.time()
            recommendations[u] = l
            times[u] = end - start
            objs[u] = targetQ
        experiment_results = [recommendations, times, objs]
    else:
        recommendations = {u:[] for u in targets}
        times = {u:[] for u in targets}
        experiment_name = f'{model.__name__}'
        for u in tqdm(targets):
            start = time.time()
            l = model(graph, u, k)
            end = time.time()
            recommendations[u] = l
            times[u] = end - start
        experiment_results = [recommendations, times]


    #==================================================================================================
    #==================================================================================================
    #==================================================================================================

    with open(f'./Saves/{file_name}.pkl','rb') as f:
        baseDictionary = pickle.load(f)
    
    if partitioname in baseDictionary.keys():
        baseDictionary[partitioname][experiment_name] = experiment_results
    else:
        baseDictionary[partitioname] = {}
        baseDictionary[partitioname][experiment_name] = experiment_results
    
    with open(f'./Saves/{file_name}.pkl','wb+') as f:
        pickle.dump(baseDictionary, f)
    print('ended at : ',datetime.datetime.now())