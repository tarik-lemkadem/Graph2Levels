#############################################################
#
#      graph embedding using path levels
#
#############################################################



import networkx as nx
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegressionCV
from tensorflow import keras
from networkx.linalg.graphmatrix import adjacency_matrix
from scipy.special import softmax
import networkx.generators.classic as Graphs
from scipy.sparse import csr_matrix
from scipy.sparse import identity
import time
from tqdm import tqdm
import random as rd




def normalise_weights(G,max_weight=None,convert=False):
    '''
    using softmax function


    '''
    n = len(G.nodes)
    
    #l = set(list(range(1,n+1)))
    
    #print(l)
    
    M = adjacency_matrix(G,nodelist=G.nodes())#,nodelist=l)
    
    #M = nx.convert_matrix.to_numpy_matrix(G)
    rows = M.nonzero()[0]
    cols = M.nonzero()[1]
    
    
    index = list(zip(rows,cols))
    #print(tuple(index))
    print("matrix shape :  " + str(M.shape[1]) +" X "  +  str(M.shape[0]) )
    count  = 0
    if convert:
        for i in index:
                M[i] = max_weight - M[i] 
                #print(str(i)+'      ' +str(M[0,i]))
                count = count +1 
                if count > 100:
                    break

    ## Softmax
    L = M.tolil()
    L = L.asfptype()
    #print(L.todense())
    #ind =M.indices
    #for i in ind:
     #   print(M[i])
    for i in range(len(L.rows)):
        X = []
        #print(L.rows[i])
        for j in L.rows[i]:
            X.append(L[i,j])
            
        if len(X) > 1:
            X = softmax( np.array(X))
        else:
            continue
        c = 0
        for j in L.rows[i]:
            #print(X[c])
            L[i,j] = X[c]
            #print(L[i,j])
            c=c+1
        #print (X)
        del(X)
    #print(L.todense())
    
    return L
    



def create_transaction_M(M):
    '''
    takes as input the normalized adjacency matrix of graph 
    '''
    for i in range(len(M.rows)):
        M[i,i] = 0
    
    return M  




def create_levels(G,M,T,level,alpha=.99):
    Multi_level = []
    matrix = M
    #G = nx.to_directed(G)
    N = M.shape[0]
    #Multi_level.append(identity(N).tolil())
    Multi_level.append(N)  
    Multi_level.append(M)
    #edges = G.edges()
    level_edges = G.edges()
    Path = {}
    Path[1] = level_edges 
    node_map = {}
    t = []
    leng = []
    leng.append(len(level_edges ))
    c = 0
    #with open('_file.txt', 'w',encoding='utf8') as f: 
    for i in list(G.nodes.items()):
        node_map[i[0]] = c
        c += 1
        

    del(c)
    #src = []
    #dst = []
    start =time.time()
    for i in range(2,level+1):
        temp = csr_matrix((N, N), dtype=np.float64).tolil()
        next_level = []
        #prev = Multi_level[i-1]
        
        print(f" {len(level_edges)}   edges in current level ' {i} '")
        for edge in tqdm(level_edges):
            last_elt = list(edge)[-1]
            node_nbr = G.edges(last_elt)
            for nbr in node_nbr:
                elt = list(edge)[0]
                next_elt = list(nbr)[-1]
                if next_elt not in list(edge):
                    a = list(edge)
                    a.append(next_elt)#.append(list(nbr)[-1])
                    next_level.append(a)
                    temp[node_map[elt],node_map[next_elt]] +=  alpha*Multi_level[i-1][node_map[elt],node_map[last_elt]]*T[node_map[last_elt],node_map[next_elt]]
                    #print(f"list : {edge} ,{next_elt}  ")
            #print(G.nodes())#.index(edge[0]))
            #print(G.get_edge_data(edge[1],edge[1]))
            #print(T[node_map[edge[0]],node_map[edge[1]]])
        #print(temp.todense())
        level_edges = next_level
        leng.append(len(level_edges))
        Path[i] = level_edges
        
        Multi_level.append(temp)
        t.append(time.time() - start)
        del(next_level)
        del(temp)
        #del()
        print(f"level {i} built  in {time.time() - start}")
    return Multi_level , t,leng




def embedding(nodes,Multi_level):
    
    
    level = len(Multi_level)
    E = {}
    #print(type(nodes))
    c = 1
    for l in Multi_level[c:]:
        rep = {}
        #print(l)
        for v in range(len(nodes)):
        
            
            coor = 0
            #print(type(l))
            for val in l.rows[v]:
                #print(l[v,val])
                coor = coor +  l[v,val]
    
            rep[nodes[v]] = coor 
        E[c] = rep
        c+=1
    '''
    for v in range(len(nodes)):
        
        rep = []
        
        for l in Multi_level[1:]:
            
            coor = 0
            #print(type(l))
            for val in l.rows[v]:
                #print(l[v,val])
                coor = coor +  l[v,val]
    
            rep.append(coor)
        E[l][nodes[v]] = rep
        #print(rep)
    #for elt in E:
     #   print(elt)
    '''
    return E

def embedding_g(nodes,Multi_level):
    
    
    level = len(Multi_level)
    E = []
    #print(type(nodes))
    c = 0
    for l in Multi_level[c:]:    
        
        
        #print(type(l))#.toarray())
        C = sum(l.toarray())/len(nodes)
        #print(sum(C))
        E.append(sum(C))
    
    return E





def nodes_embedding(G,level=2,alpha=.99):
    
  
    #sorted_list = sorted(G.edges(data=True),key= lambda x: x[2]['weight'])
    
    #max_weight = sorted_list[-1][2]['weight']

    #print(type(max_weight))
    level = level
    alpha = alpha

    L = adjacency_matrix(G,nodelist=G.nodes()).tolil().asfptype()

    start = time.time()
    #############
    M = normalise_weights(G)#,max_weight)
    print("   matrix is normalized         ")
    T = create_transaction_M(M)
    print("   matrix of transaction built       ")
    level_M = create_levels(G,L,T,level,alpha)
    print("   begin embedding         ")
    Embed = embedding(list(G.nodes),level_M[0])
    #print(M.todense())
    print(f"Embeding done in  {time.time()-start}")
    return Embed





def graph_embedding(G,level=2,alpha=.99):
    
  
    #sorted_list = sorted(G.edges(data=True),key= lambda x: x[2]['weight'])
    
    #max_weight = sorted_list[-1][2]['weight']

    #print(type(max_weight))
    level = level
    alpha = alpha
    
    L = adjacency_matrix(G,nodelist=G.nodes()).tolil().asfptype()

    start = time.time()
    #############
    
    M = normalise_weights(G)#,max_weight)
    print("   matrix is normalized         ")
    T = create_transaction_M(M)
    print("   matrix of transaction built       ")
    level_M, roll ,leng = create_levels(G,L,T,level,alpha)
    
    print("   begin embedding         ")
    Embed = embedding_g(list(G.nodes),level_M)
    print(f"Embeding done in  {time.time()-start}")

    return Embed , roll,leng