from Embed import *
import json
import networkx as nx
from networkx.readwrite import json_graph
import pandas as pd
import time



f = "./data/git_stargazers/git_edges.json"
c = 0 
df = pd.read_csv('./data/git_stargazers/git_target.csv')
print(df.head(10))


with open(f) as f:
    for obj in f:
        gr = json.loads(obj)
        #gr = sorted(gr)
        #k = gr
        #print(k[0:5])
        for id,g in gr.items():#.values():
            lines = []
            print(id)
            print(g)
            for edge in g:
                lines.append(f"{edge[0]}  {edge[1]}")
            
        #js_graph = json.load(gr)
        #G = json_graph.node_link_graph(gr)
            G = nx.parse_edgelist(lines)
            embedding = graph_embedding(G,level=3,alpha=.9)
            print(embedding)
            #Stime.sleep(30)
        c+=1
        print(c)
        time.sleep(30)
        
                

        

        #pprint(js_graph)

    #return json_graph.node_link_graph(js_graph)



G = nx.read_edgelist('karate-mirrored.edgelist')
#G = nx.read_edgelist('./data/wiki/Wiki_edgelist.txt') 
#G = nx.read_edgelist('europe-airports.edgelist') 






