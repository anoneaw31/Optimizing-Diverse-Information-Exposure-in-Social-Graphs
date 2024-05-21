import numpy as np
import networkx as nx
import scipy.sparse as ss
from scipy.special import softmax


""" Given a graph and a partition dictionary {parts:[nodes]} returns the P matrix nb parts x nb nodes"""
def partition_matrix(graph, partition):
    P = np.zeros(shape=(len(partition.keys()), len(graph.nodes)))
    for i, nodelist in partition.items():
        for node in nodelist:
            P[i,node] = 1
    return ss.csr_matrix(P,dtype=np.int64)

def continuous_P(graph, attr_name):
    Pnparray = []
    for node in graph.nodes():
        Pnparray.append(graph.nodes[node][attr_name])

    Pnparray = np.array(Pnparray).T
    return ss.csr_matrix(Pnparray)

def produce_partition(graph, partitioname, c_attr = None):
    if partitioname == 'louvain_communities_community_partition':
        from Functions.Partitions import louvain_communities_community_partition
        partition = louvain_communities_community_partition(graph)
        P = partition_matrix(graph, partition)
    elif partitioname == 'ground-truth':
        partition = {}
        for node in graph.nodes(data=True):
            if node[1]['ground-truth'] in partition.keys():
                partition[node[1]['ground-truth']].append(node[0])
            else:
                partition[node[1]['ground-truth']] = [node[0]]
        P = partition_matrix(graph, partition)
    elif 'continuous' in partitioname:
        partition = {}
        P = continuous_P(graph, attr_name=c_attr)
    
    return partition, P


"""given an array of numbers, returns the distribution"""
def distribution(arr):
    total = np.sum(arr)
    if total == 0:
        return arr
    else:
        return arr/total

def uniformise(matrix):
    return matrix/np.sum(matrix, axis=1)

"""Objective Distribution Functions"""
def UniformQ(PMatrix, WMatrix, u):
    return (np.ones(PMatrix.shape[0])/PMatrix.shape[0]).reshape((PMatrix.shape[0],1))

def PreferentialQ(PMatrix, WMatrix, u):
    target = np.zeros(shape=(WMatrix.shape[0],1))
    target[u,0] = 1 # np.random.randint(len(graph.nodes))
    target = ss.csr_array(target,dtype=np.int64)
    targetQ = UniformQ(PMatrix, WMatrix, u)
    home = np.argmax(PMatrix @ target)

    base = (PMatrix @ WMatrix @ target).multiply(targetQ).T.toarray().flatten()
    harmonizer = np.array([1 + 1/PMatrix.shape[0] * np.exp(-((x - home) ** 2) / (2 * (PMatrix.shape[0]/2) ** 2)) if x != home else 1 for x in range(PMatrix.shape[0])])
    baseliner = [base[home]/10 if i != home else 0 for i in range(PMatrix.shape[0])]
    return distribution((base + baseliner) * harmonizer).reshape(PMatrix.shape[0],1)

"""It returns zero if the input is less than zero otherwise it returns the given input"""
def RELU(x):
    x1=np.zeros(shape=x.shape)
    for i in range(len(x)):
        if x[i]<0:
            x1[i] = -x[i]
        else:
            x1[i] = 0

    return x1

"""Given a square matrix, returns the corresponding matrix where each line sums to 1"""
def rowstochastic(A):
    return A.multiply(1/A.sum(axis=1))

# ref Generalized inversion of modified matrices
def increment_matrix_inverse(A,edge):
    c = ss.csr_array(np.eye(A.shape[0],k=edge[0],dtype=int)[0]).T
    d = ss.csr_array(np.eye(A.shape[0],k=edge[1],dtype=int)[0])


    beta = 1 + (d @ A @ c).todense()[0][0]

    return A - (1/beta) * (A @ c @ d @ A)

"""W functions"""
def WBFS(A,u, d=3):
    #res = A.copy()
    #step_matrix = A.copy()
    #for _ in range(3):
    #    step_matrix = step_matrix @ A
    #    res += step_matrix
    #return res
    A2 = A@A
    return A + A2 + A2@A

def WPPR(A,u, alpha=0.05):
    #return (1- alpha) * ss.linalg.inv((ss.eye(A.shape[0]) - A.multiply(alpha)).tocsc())
    #return (1 - alpha) * ss.linalg.spsolve((ss.eye(A.shape[0])-A.multiply(alpha)).tocsc(),ss.eye(A.shape[0]).tocsc())
    return ss.csr_matrix((1- alpha) * np.linalg.inv((ss.eye(A.shape[0]) - A.multiply(alpha)).todense()))

def WFJ(A,u, alpha=0.05):
    #return (1- alpha) * ss.linalg.inv((ss.eye(A.shape[0]) - (A.T).multiply(alpha)).tocsc())
    #return (1 - alpha) * ss.linalg.spsolve((ss.eye(A.shape[0])-(A.T).multiply(alpha)).tocsc(),ss.eye(A.shape[0]).tocsc())
    return ss.csr_matrix((1- alpha) * np.linalg.inv((ss.eye(A.shape[0]) - (A.T).multiply(alpha)).todense()))

"""Incremental Wfunctions"""
def increment_WBFS(A, edge, oldW):
    u,v = edge
    C = ss.lil_matrix(A.shape, dtype=int)
    C[u,v] = 1
    C[:, [v]] += A[[u], :].T
    C[[u], :] += A[:, [v]].T
    uv2 = np.sum(A[(A[[u],:] == 1).toarray().flatten()], axis=0)
    vv2 = np.sum(A[(A[:,[v]] == 1).toarray().flatten()], axis=0)
    C[[u],:] += vv2.reshape(C[[u],:].shape)
    C[:,[v]] += uv2.reshape(C[:,[v]].shape)
    check = np.sum(A[[u],:]==1)
    if check > 0:
        C[(A[[u],:]==1).toarray().flatten(),:] += ss.vstack([A[:, [v]].T for _ in range(np.sum(A[[u],:]==1))])

    return oldW + C

def increment_WPPR(A, edge, alpha = 0.05):
    matrix = (ss.eye(A.shape[0]) - A.multiply(alpha))
    return (1-alpha) * increment_matrix_inverse(matrix,edge)

def increment_WFJ(A, edge, alpha = 0.05):
    matrix = (ss.eye(A.shape[0]) - (A.T).multiply(alpha))
    return (1-alpha) * increment_matrix_inverse(matrix, edge)

""" Given 
    P partition matrix
    W matrix of transition
    targetarray indicator array of target node
    endQ the goal distribution
    returns the corresponding value of the objective functions
    """
def OBJECTIVE(PMatrix, W, targetarray, endQ):
    
    if W.shape[0] == W.shape[1]:
        disitr = distribution(((PMatrix @ W.T) @ targetarray).todense()) - endQ
    
    if W.shape[1] < W.shape[0]:
        disitr = distribution(PMatrix @ W) - endQ

    return np.linalg.norm(disitr, ord=2), disitr


""" Gradient Descent Algorithm
    Executes one step of the descent
    which corresponds to one recommendation"""
def GD_recommendation(graph, target_node, PMatrix, objectiveQ, WFunction, alpha=0.2, choice='max', mu=0.2):
    A = nx.adjacency_matrix(graph)
    A = rowstochastic(A).tolil()
    n = A.shape[0]

    target = np.zeros(shape=(A.shape[0],1))
    target[target_node,0] = 1 # np.random.randint(len(graph.nodes))
    target = ss.csr_array(target,dtype=np.int64)

    W = WFunction(A,[target_node,target_node])
    d = RELU((distribution(PMatrix @ W @ target) - objectiveQ)).T

    if 'BFS' in WFunction.__name__:
        A2 = ss.csr_matrix.power(A.tocsr(),2)
        dW = - ss.eye(n) - A - ss.eye(n).multiply(A2.T.tolil()[target_node,target_node]) - A2
    if 'PPR' in WFunction.__name__:
        Q = (ss.eye(n) - A.multiply(alpha)).tocsc()
        invQ = ss.linalg.inv(Q).tolil()
        dW = - alpha * (1 - alpha) * (invQ.T[target_node, target_node] * invQ)
    if 'FJ' in WFunction.__name__:
        Q = (ss.eye(A.shape[0]) - A.multiply(alpha).T).tocsc()
        invQ = ss.linalg.inv(Q).tolil()
        dW = - alpha * (1 - alpha) * (invQ.T[target_node, target_node] * invQ)

    dOu = 2*d @ PMatrix @ dW
    newA = (A[[target_node], :] - mu * dOu)[0]

    #projection
    newA[newA < 0] = 0
    newA[target_node] = 0
    newA[A[[target_node],:].todense()[0] > 0] = 0
    #newA /=np.sum(newA)

    if choice == 'max':
        return np.argmax(newA)
    if choice == 'random':
        return np.random.choice(range(len(newA)),size=1,replace=False,p=softmax(50*newA))[0]





