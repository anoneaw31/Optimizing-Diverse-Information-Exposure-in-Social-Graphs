import numpy as np
import networkx as nx
import re
import urllib.request
import io
from zipfile import ZipFile 
import os, shutil

'''
apply a given partition to a given graph
'''
def apply_partition(graph,partition):

  for part,nodelist in partition.items():
    for node in nodelist:
      graph.nodes[node]['partition'] = part

'''
Computes the segregation degree of a node in a partitioned graph
defined as the average distance of a random walk that leaves the home partition
'''
def segregation_degree(G,node, nb_walks = 100):

  home = G.nodes[node]['partition']
  probabilities = []

  for _ in range(nb_walks):
    steps = 0
    current_habitat = home
    next = node
    while current_habitat == home:

      surroundings = list(nx.neighbors(G,next))
      next = surroundings[np.random.randint(0,len(surroundings))]

      current_habitat = G.nodes[next]['partition']
      steps += 1
    probabilities.append(steps)
  
  return np.mean(probabilities)


def distribution(arr):
    total = np.sum(arr)
    return arr/total

'''
Data loader
currently usable only football and karate
'''
def LOAD(dataset):
    
    def date_to_int(stringl,start):
        timi = [365,30,1]
        result = 0
        if int(stringl[0]) <= start:
            return 0
        for i in range(3):
            if i == 0:
                result += (int(stringl[i]) - start) * timi[i]
            else:
                result += int(stringl[i]) * timi[i]
        return result

    def relabel(graph):
        translator = {}
        i = 0
        for node in graph.nodes:
            translator[node] = i
            i += 1

        return nx.relabel_nodes(graph, translator, copy=True)
        
    #Zachary's karate club
    if dataset == 'karate':
        return nx.karate_club_graph()
    
    #frrelabelom http://www-personal.umich.edu/~mejn/netdata/
    if dataset == 'polbooks':
        gml = "./Datasets/Polbooks/polbooks.gml"
        g = nx.read_gml(gml)

        translation = {'l':-1,
               'n':0,
               'c':1}

        i = 0
        update_attribute = {}
        update_id = {}
        for a in g.nodes():
            update_attribute[i] = {'title':a, 'value':translation[g.nodes[a]['value']]}
            update_id[a] = i
            i += 1 

        nx.relabel_nodes(g,update_id,copy=False)
        nx.set_node_attributes(g,update_attribute)

        for node in g.nodes():
            g.nodes[node]['ground-truth'] = g.nodes[node].pop('value')

        return g
    
    #football dataset from netowrkx
    if dataset == 'football':
        url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"

        sock = urllib.request.urlopen(url)  # open URL
        s = io.BytesIO(sock.read())  # read into BytesIO "file"
        sock.close()

        zf = ZipFile(s)  # zipfile object
        txt = zf.read("football.txt").decode()  # read info file
        gml = zf.read("football.gml").decode()  # read gml data
        # throw away bogus first line with # from mejn files
        gml = gml.split("\n")[1:]
        football = nx.parse_gml(gml)  # parse gml data
        
        #
        # In football nodes are indexed by their team names and not by an integer id
        #
        index_to_fname = list(football.nodes())
        fname_to_index = {}

        i = 0
        for name in index_to_fname:
            fname_to_index[name] = i
            i += 1

        for node in football.nodes():
            football.nodes[node]['ground-truth'] = football.nodes[node].pop('value')

        return nx.relabel_nodes(football, fname_to_index, copy=True)
    
    #email-eu-core from SNAP
    if dataset == 'emails':
        emals = []

        with open("./Datasets/Email/email-Eu-core.txt") as f:
            temp = f.readlines()

        for ff in temp:
            t = re.split(r' |\n', ff)[:2]
            emals.append([int(t[0]), int(t[1])])

        emails = nx.Graph()
        emails.add_edges_from(emals)

        # weird dataset phenomenon where some nodes are only connected to themselves
        # so we'll just remove them
        polluters = []
        for node in emails.nodes:
            if nx.degree(emails,node) <= 2 and [node,node] in emails.edges:
                polluters.append(node)
        emails.remove_nodes_from(polluters)

        emails = relabel(emails)
        with open("./Datasets/Email/email-Eu-core-department-labels.txt") as f:
            temp = f.readlines()
        for ff in temp:
            t = ff.replace('\n','').split(' ')
            nx.set_node_attributes(emails, {int(t[0]): {'ground-truth':int(t[1])}})

        return emails
    
    #facebook from SNAP
    if dataset == 'facebook':
        facbook = []

        with open("./Datasets/Facebook/facebook_combined.txt") as f:
            temp = f.readlines()

        for ff in temp:
            t = re.split(r' |\n|https', ff)[:2]
            facbook.append([int(t[0]), int(t[1])])

        facebook = nx.Graph()
        facebook.add_edges_from(facbook)

        return facebook
    
    #from SNAP
    if dataset == 'epinions':
        epipen = []

        with ZipFile('./Datasets/Epinions/epinions.zip', 'r') as zObject: 
            zObject.extract(
                "soc-Epinions1.txt", path='./Datasets/Epinions/')
        zObject.close() 

        with open("./Datasets/Epinions/soc-Epinions1.txt") as f:
            temp = f.readlines()

        for ff in temp[4:]:
            t = re.split(r' |\n|\t', ff)[:2]
            epipen.append([int(a) for a in t])

        Epinions = nx.Graph()
        Epinions.add_edges_from(epipen)
        Epinions = Epinions.subgraph(max(nx.connected_components(Epinions),key=len)).copy()
    
        return relabel(Epinions)
    
    #from SNAP
    if dataset == 'college':
        kindergarten = []

        with open("./Datasets/CollegeMsg/CollegeMsg.txt") as f:
            temp = f.readlines()

        for ff in temp:
            t = re.split(r' |\n', ff)[:3]
            kindergarten.append([int(a) for a in t])

        kindergarten = [[a,b,np.floor(c/(3600*24))] for a,b,c in kindergarten]
        m = min([c for _,_,c in kindergarten])
        #node ids in the dataset start at 1, the -1 below just brings the start back at 0 to be in line with the other datasets
        kindergarten = [[a-1,b-1,c - m] for a,b,c in kindergarten]

        College = nx.Graph()
        College.add_weighted_edges_from(kindergarten,weight='timestamp')
        College = College.subgraph(max(nx.connected_components(College),key=len))
        
        return relabel(College)
    
    #from SNAP
    if dataset == 'reddit':
        readit = []

        with ZipFile('./Datasets/Reddit/soc-redditHyperlinks-body.zip', 'r') as zObject: 
            zObject.extract(
                "soc-redditHyperlinks-body.tsv", path='./Datasets/Reddit')
        zObject.close() 


        with open("./Datasets/Reddit/soc-redditHyperlinks-body.tsv") as f:
            temp = f.readlines()

        for ff in temp[1:]:
            t = re.split(r' |\n|\t', ff)
            readit.append([t[0],t[1],t[3]])


        readit = [[a,b,date_to_int(re.split(r'-',c),start=2013)] for a,b,c in readit]

        Reddit = nx.Graph()
        Reddit.add_weighted_edges_from(readit,weight='timestamp')
        Reddit = Reddit.subgraph(max(nx.connected_components(Reddit),key=len))

        #
        # In reddit nodes are indexed by their subreddit names and not by an integer id
        #
        reddit_index_to_fname = list(Reddit.nodes())
        reddit_fname_to_index = {}

        i = 0
        for name in reddit_index_to_fname:
            reddit_fname_to_index[name] = i
            i += 1

        Reddit = nx.relabel_nodes(Reddit, reddit_fname_to_index, copy=True)
        return Reddit
    
    # fb-wosn-friend from network repository
    if dataset == 'facebookfriend':
        facbookfren = []

        with ZipFile('./Datasets/Facebook-friends/fb-wosn-friends.zip', 'r') as zObject: 
            zObject.extract(
                "fb-wosn-friends.edges", path='./Datasets/Facebook-friends')
        zObject.close() 

        with open("./Datasets/Facebook-friends/fb-wosn-friends.edges") as f:
            temp = f.readlines()

        for ff in temp[2:]:
            t = re.split(r' |\n|\t', ff)[:4]
            facbookfren.append([int(a) for a in t])

        facbookfren = [[a,b,np.floor(c/(3600*24))] for a,b,_,c in facbookfren]
        m = sorted([c for _,_,c in facbookfren])[1]
        facbookfren = [[a,b,c - m] for a,b,c in facbookfren]

        Facefriends = nx.Graph()
        Facefriends.add_weighted_edges_from(facbookfren,weight='timestamp')
    
        return relabel(Facefriends.subgraph(max(nx.connected_components(Facefriends),key=len)))
    
    # soc-flickr-growth from netowkr repository
    if dataset == 'flickr':
        flic = []

        with open("./Datasets/Flickr/soc-flickr-growth.edges") as f:
            temp = f.readlines()

        for ff in temp[1:]:
            t = re.split(r' |\n|\t', ff)
            flic.append([int(a) for a in list(filter(None,t))])

        flic = [[a,b,np.floor(c/(3600*24))] for a,b,_,c in flic]
        m = sorted([c for _,_,c in flic])[0]
        flic = [[a,b,c - m] for a,b,c in flic]

        Flickr = nx.Graph()
        Flickr.add_weighted_edges_from(flic,weight='timestamp')
        return relabel(Flickr.subgraph(max(nx.connected_components(Flickr),key=len)))

    #Movie-lens 100k item-item graph
    if dataset == 'ml-100k':
        path = './Datasets/Movielens/'
        name = dataset.split('|')[0]
        
        with ZipFile(path + name + '.zip', 'r') as zObject: 
            zObject.extract(
                "ml-100k/u.item", path=path + 'data')
        zObject.close() 

        node_list = []
        with open(path + "data/" + name + '/u.item', encoding='ISO-8859-1') as f:
            temp = f.readlines()
            i = 1
            for ff in temp:
                t = ff.replace('\n','').split(sep='|')
                node = {'genre':[int(element) for element in t[5:]]}
                if node['genre'][0] == 1:
                    i += 1
                else:
                    node_list.append([int(t[0]) -i, node])

        shutil.rmtree(path + 'data')
        edgelist = []
        for noD_nb in range(len(node_list)):
            for othernoD_nb in range(noD_nb+1,len(node_list)):
                if np.sum(np.array(node_list[othernoD_nb][1]['genre']) * np.array(node_list[noD_nb][1]['genre'])) > np.floor(0.66*np.sum(node_list[othernoD_nb][1]['genre'])): #19 total number of gneres
                    edgelist.append([noD_nb,othernoD_nb])
        
        graph = nx.Graph()
        graph.add_nodes_from(node_list)
        graph.add_edges_from(edgelist)

        change = []
        for node in graph.nodes:
            genre_array = np.array(graph.nodes[node]['genre'])
            for neighbor in nx.neighbors(graph,node):
                genre_array += np.array(graph.nodes[neighbor]['genre'])
                graph
            
            change.append([node, distribution(genre_array)])

        for node,genre in change:
            graph.nodes[node]['genre'] = genre

        del change
        del node_list
        del edgelist

        return graph

    #Movie-lens 25m item-item graph
    if dataset == 'ml-25m':
        path = './Datasets/Movielens/'
        name = dataset.split('|')[0]
        
        with ZipFile(path + name + '.zip', 'r') as zObject: 
            zObject.extract(
                "ml-25m/u.item", path=path + 'data')
        zObject.close() 

        node_list = []
        with open(path + "data/" + name + '/u.item', encoding='ISO-8859-1') as f:
            temp = f.readlines()
            i = 1
            for ff in temp:
                t = ff.replace('\n','').split(sep='|')
                node = {'genre':[int(element) for element in t[5:]]}
                if node['genre'][0] == 1:
                    i += 1
                else:
                    node_list.append([int(t[0]) -i, node])

        shutil.rmtree(path + 'data')
        edgelist = []
        for noD_nb in range(len(node_list)):
            for othernoD_nb in range(noD_nb+1,len(node_list)):
                if np.sum(np.array(node_list[othernoD_nb][1]['genre']) * np.array(node_list[noD_nb][1]['genre'])) > np.floor(0.66*np.sum(node_list[othernoD_nb][1]['genre'])): #19 total number of gneres
                    edgelist.append([noD_nb,othernoD_nb])
        
        graph = nx.Graph()
        graph.add_nodes_from(node_list)
        graph.add_edges_from(edgelist)

        change = []
        for node in graph.nodes:
            genre_array = np.array(graph.nodes[node]['genre'])
            for neighbor in nx.neighbors(graph,node):
                genre_array += np.array(graph.nodes[neighbor]['genre'])
                graph
            
            change.append([node, distribution(genre_array)])

        for node,genre in change:
            graph.nodes[node]['genre'] = genre

        del change
        del node_list
        del edgelist

        return graph

    #Epinions dataset user user graph
    if dataset == 'epinions-uug_ratings':
        epipen = []

        with open("./Datasets/Epinions/user_rating.txt") as f:
            temp = f.readlines()

        for ff in temp[:]:
            t = re.split(r' |\n|\t', ff)[:3]
            epipen.append([int(a) for a in t])

        Epinions = nx.Graph()
        Epinions.add_weighted_edges_from(epipen)
        Epinions = Epinions.subgraph(max(nx.connected_components(Epinions),key=len)).copy()
        translator = {}
        i = 0
        for node in Epinions.nodes:
            translator[node] = i
            i += 1

        Epinions = nx.relabel_nodes(Epinions, translator, copy=True)
        del epipen

        for node in Epinions.nodes:
            opinion = {1:0, -1:0}
            for neighbor in nx.neighbors(Epinions, node):
                opinion[Epinions.edges[node,neighbor]['weight']] += 1
            
            nx.set_node_attributes(Epinions, {node: {'opinion':distribution(list(opinion.values()))}})
        return Epinions
    
    return None