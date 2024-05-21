from Functions.Matrix import WPPR, WBFS, WFJ
import Functions.tools as ft
import Functions.Matrix as fm
import networkx as nx
import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt
from Functions.tools import LOAD
from tqdm import tqdm 
import pickle

"""
python plot.py

Change the parameters below to correspond to which experiment you want to plot
 - file_name :
    Corresponds to the dataset you want to plot
    only change the last element of the string
    values in : reddit, facebookfriend, epinions
 - W2show : 
    Corresponds to the exposure function you want to plot
    values in : WBFS, WPPR, WFJ
 - strat2show : 
    Corresponds to the Strategies you want to plot
    must be a list
    values inside the list in : 'greedy', ' MPB-all', 'convexity'
 - baseline2show :
    Corresponds to the baselines you want to plot
    must be a list 
    values inside the list in : 'GlobalDemaine', 'SingleSourceDemaine', 'Neurips', 'random_recommendation', 'common_neighbors_recommendation'
 - Q2show
    Corresponds to the target distribution you want to plot
    value in : UniformQ, PreferentialQ

for strat2show and baseline2show, if the experiments are not present nothing will be ploted
"""
file_name = f'./Results/epinions'
partition2show = 'louvain_communities_community_partition' #louvain_communities_community_partition, ground-truth
W2show = 'WBFS'
strat2show = ['greedy', ' MPB-all', 'convexity']
baseline2show = ['GlobalDemaine', 'SingleSourceDemaine', 'Neurips', 'random_recommendation', 'common_neighbors_recommendation']
Q2show = 'UniformQ' #PreferentialQ, UniformQ 


with open(file_name + '.pkl','rb') as f:
    savedictionary = pickle.load(f)



if W2show == 'WPPR':
    WF2show = WPPR
elif W2show == 'WFJ':
    WF2show = WFJ
elif W2show == 'WBFS':
    WF2show = WBFS



dataname = savedictionary['dataname']
targets = savedictionary['targets']
partition = savedictionary[partition2show]['partition']
graph = LOAD(savedictionary['dataname'])
ft.apply_partition(graph,partition)
if 'continuous' in partition2show:
    P = fm.continuous_P(graph, partition2show.split('-'))
else:
    P = fm.partition_matrix(graph,partition)
k = 50

for experiment in savedictionary[partition2show].keys():



    if W2show in experiment and Q2show in experiment and any([strat in experiment for strat in strat2show]) or\
        any([baseline in experiment for baseline in baseline2show]):
        
        if 'Neurips' in experiment:
            xplabel = 'SpGreedy'
        elif 'Global' in experiment:
            xplabel = 'Global Diameter Reduction'
        elif 'Single' in experiment:
            xplabel = 'Single Source Diameter Reduction'
        elif 'random' in experiment:
            xplabel = 'Random'
        elif 'neighbors' in experiment:
            xplabel = 'Triadic Closure'

        elif 'convexity' in experiment:
            xplabel = 'ConvexDiverse'
        elif 'greedy' in experiment:
            xplabel = 'GreedyDiverse'
        elif 'PB' in experiment:
            xplabel = 'MinBoostingDiverse'

        record = []
        with open(f'./Results/{dataname}-{partition2show}-{W2show}-{Q2show}.csv', 'w') as f:
            f.write('target,first,last\n')
        for target_node in tqdm(targets):
            COPY = graph.copy()
            Adj = nx.adjacency_matrix(COPY)
            Adj = fm.rowstochastic(Adj).tolil()
            Wm = WF2show(Adj, [target_node,target_node])

            targetarray = np.zeros(shape=(Adj.shape[0],1))
            targetarray[target_node,0] = 1 # np.random.randint(len(graph.nodes))
            targetarray = ss.csr_array(targetarray,dtype=np.int64)
            
            if any([baseline in experiment for baseline in baseline2show]):
                objectiveQ = savedictionary[partition2show][f'{W2show} {strat2show[0]} {Q2show}'][2][target_node]
            else:
                objectiveQ = savedictionary[partition2show][experiment][2][target_node]

            toplot = [fm.OBJECTIVE(P,Wm,targetarray, objectiveQ)[0]]
            fn = toplot[0]
            for recommended_node in savedictionary[partition2show][experiment][0][target_node]:
                COPY.add_edge(target_node,recommended_node)
            Adj = nx.adjacency_matrix(COPY)
            Adj = fm.rowstochastic(Adj).tolil()
            Wm = WF2show(Adj.tocsc(), [target_node, recommended_node])
                #o, _ = OBJECTIVE(P,Wm,targetarray, objectiveQ)
            o = np.linalg.norm(fm.distribution((P @ Wm @ targetarray).todense()) - objectiveQ)
            toplot.append(o)
            while len(toplot) < k+1:
                toplot.append(toplot[-1])
            ln = o
            with open(f'./Results/{dataname}-{partition2show}-{W2show}-{Q2show}.csv', 'a') as f:
                f.write(f'{target_node},{fn},{ln}\n')
            record.append(toplot)
        #break
        average = np.mean(record, axis=0)
        standard = np.std(record, axis=0)
        print (average)
        print (standard)
        plt.plot(average, label = xplabel)
        plt.xlabel('number of added edges')
        plt.ylabel('Objective Function')
        #plt.ylim(1.5,0)
        plt.legend()
        plt.savefig(f'./Results/{dataname}-{partition2show}-{W2show}-{Q2show}.pdf')
        