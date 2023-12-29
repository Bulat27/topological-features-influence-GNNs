import networkx as nx
from utilities import * 
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data
from node2vec import run_training

# compute structural features of graph G specified in feats
# usage example: structural_features(G,['cc','bc','dc'])
def structural_features(G, feats):
    num_nodes = G.number_of_nodes()
    features = torch.zeros(num_nodes,0)
    for feat in feats:
        vals = dict()
        if feat == 'cc':
            vals = nx.clustering(G)
        elif feat == 'bc':
            vals = nx.betweenness_centrality(G, 30)
        elif feat == 'dc':
            vals = nx.degree_centrality(G)
        elif feat == 'ec':
            vals = nx.eigenvector_centrality(G)
        elif feat == 'pr':
            vals = nx.pagerank(G)
        elif feat == 'cn':
            vals = nx.node_clique_number(G)
        elif feat == 'lc':
            lc_part = nx.community.louvain_communities(G)
            lc = dict()
            for i in range(len(lc_part)):
                community = list(lc_part[i])
                for node in community:
                    lc[node] = len(community)
            vals = lc
        elif feat == 'nd':
            vals = nx.average_neighbor_degree(G)
        elif feat == 'kc':
            vals = nx.core_number(G)

        vals_tensor = dict_to_tensor(vals)
        features = concatenate(features,vals_tensor)
    
    return features

# compute positional features of data
def positional_features(data):
    return run_training(data)

# create new data with the concatenation of additional features
def create_data_with_features(data,features):
    X = concatenate(data.x,features)
    data_features = Data(
        x=X, 
        edge_index=data.edge_index, 
        y=data.y, 
        train_mask=data.train_mask, 
        val_mask=data.val_mask, 
        test_mask=data.test_mask, 
        name=data.name, 
        num_classes=data.num_classes)

    return data_features