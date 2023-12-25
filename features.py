import networkx as nx
from utilities import * 
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data

# convert the directed graph to undirected (better performance)
def data_to_undirected(data):
    G = to_networkx(data, to_undirected=True)
    data_unidirected = from_networkx(G)

    # restore the attributes
    data_unidirected.name = data.name
    data_unidirected.num_classes = data.num_classes
    data_unidirected.x = data.x
    data_unidirected.y = data.y
    data_unidirected.train_mask = data.train_mask
    data_unidirected.val_mask = data.val_mask
    data_unidirected.test_mask = data.test_mask

    return G, data_unidirected

# compute the topological features out of the graph
def compute_features(G):
    cc = nx.clustering(G)
    bc = nx.betweenness_centrality(G, 30)
    dc = nx.degree_centrality(G)
    ec = nx.eigenvector_centrality(G)
    pr = nx.pagerank(G)
    cn = nx.node_clique_number(G)
    lc_part = nx.community.louvain_communities(G)
    lc = dict()
    for i in range(len(lc_part)):
        community = list(lc_part[i])
        for node in community:
            lc[node] = len(community)
    lc = MinMaxNormalization(lc)
    nd = nx.average_neighbor_degree(G)
    nd = MinMaxNormalization(nd)
    kc = nx.core_number(G)
    kc = MinMaxNormalization(kc)

    return list([cc,bc,dc,ec,pr,cn,lc,nd,kc])

# concatenate the topological features to the corresp. node features in X
def concatenate_features(X,features):
    X_list = X.tolist()
    for i in range(len(X_list)):
        for feature in features:
            X_list[i].append(feature[i])

    return torch.tensor(X_list)

# 1. convert graph to undirected
# 2. compute topological features
# 3. return data with additional features 
def create_data_with_features(data):
    G, data = data_to_undirected(data)
    features = compute_features(G)
    X = concatenate_features(data.x,features)
    data_features = Data(x=X, edge_index=data.edge_index, y=data.y, train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask, name=data.name, num_classes=data.num_classes)

    return data_features

