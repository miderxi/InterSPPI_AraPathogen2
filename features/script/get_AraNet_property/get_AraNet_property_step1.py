import numpy as np
import networkx as nx
import numpy as np
import pickle

# read
#(1) ppis
ppis_ara = np.genfromtxt("./arabidopsis_thaliana_ppis.txt",str)
ara_ids = set([j for i in ppis_ara for j in i])

#(2)target protein of train
# we hyposis that different effector allways target same arabidopsis protein.
# this is about 10 train set 

target_of_train = []
for i in range(10):
    target_of_train.append( set(np.genfromtxt(f"./target_of_train/train_known_target{i}.txt",str)))


#(3) target of experiment veried arabidopsis thaliana disease resistence protein
target_rgene_reference = set([i.upper() for i in np.genfromtxt("./target_rgene_reference.txt",str)])

#(4) target of all possible arabidopsis thaliana disease resistence protein 
target_rgene_uniprot = set(np.genfromtxt("./target_rgene_uniprot.txt",str))

#2. Building network of arabidopsis thaliana to calculate degree,betweenness,closeness....
G = nx.Graph()
G.add_edges_from(ppis_ara)

#    compute feature
#(1) degree,betweenness, closeness, transitivity, pagerank, eccentricity and eigenvector.
#(2) minimum network distance to know target, average network distance to know targets, 
#(3) minimum network distances experimentally verified R-proteins, average network distances to experimentally verified R-protein,
#(4) minimum network distances predicted R-proteins, average network distances to predicted R-protein,

sp1 = dict(nx.all_pairs_shortest_path_length(G))
degree = G.degree   #compute degree
betweenness = nx.betweenness.betweenness_centrality(G)  #compute betweenness
closeness = nx.closeness.closeness_centrality(G)    #compute closeness
#transitivity = nx.transitivity(G)
pagerank = nx.pagerank(G)   #compute pagerank
eigenvector = nx.eigenvector.eigenvector_centrality(G)  #compute eigenvector

eccentricity = {}   #compute eccentricity
for comp in nx.connected_components(G):
    sub_g = G.subgraph(comp)
    sub_ecc = nx.eccentricity(sub_g,sp=sp1)
    for k,v in sub_ecc.items():
        eccentricity[k]=v

#
for train_i in range(10):
    net_features = dict()
    for ara_id in ara_ids:
        feature = []
        feature.append(degree[ara_id])
        feature.append(betweenness[ara_id])
        feature.append(closeness[ara_id])
        #feature.append(transitivity[ara_id])
        feature.append(pagerank[ara_id])
        feature.append(eccentricity[ara_id])
        feature.append(eigenvector[ara_id])

        #(1) add target of train min and max distance
        t_dist = []
        for j in target_of_train[train_i]:
            if j in G.nodes and j in sp1[ara_id]:
                t_dist.append(sp1[ara_id][j])
        if len(t_dist) >= 1:
            feature.append(np.array(t_dist).min())
            feature.append(np.array(t_dist).max())
        else:
            feature.append(20)
            feature.append(20)

        #(2) add target of experiment verified disease resistence gene
        t_dist = []
        for j in target_rgene_reference:
            if j in G.nodes and j in sp1[ara_id]:
                t_dist.append(sp1[ara_id][j])
        if len(t_dist) >= 1:
            feature.append(np.array(t_dist).min())
            feature.append(np.array(t_dist).max())
        else:
            feature.append(20)
            feature.append(20)
        
        #(3) add target of all possible disease resistence gene
        # come from uniprot
        t_dist = []
        for j in target_rgene_uniprot:
            if j in G.nodes and j in sp1[ara_id]:
                t_dist.append(sp1[ara_id][j])
        if len(t_dist) >= 1:
            feature.append(np.array(t_dist).min())
            feature.append(np.array(t_dist).max())
        else:
            feature.append(20)
            feature.append(20)

        net_features[ara_id] = np.array(feature)

    with open(f'../../AraNet_property/AraNet_property_train_{train_i}.pkl',"wb") as f:
        for (k,v) in net_features.items():
            net_features[k] = np.array(net_features[k],dtype=np.float32)
        pickle.dump(net_features,f)


