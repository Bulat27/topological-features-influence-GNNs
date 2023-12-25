import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import index_to_mask
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def load_data(dataset_name):
    if dataset_name == 'ogbn-arxiv':
        return load_arxiv()
    else:
        return load_cora()

# preprocess and load ogbn-arxiv dataset
def load_arxiv():
    target_dataset = 'ogbn-arxiv'
    dataset = PygNodePropPredDataset(name=target_dataset, root='networks')

    data = dataset[0]
    data.name = dataset.name
    data.num_classes = dataset.num_classes

    split_idx = dataset.get_idx_split()

    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']

    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)

    data.y = torch.flatten(data.y)

    return data

def load_cora():
    dataset = Planetoid(root='data/Planetoid', name='Cora')
    data = dataset[0]
    data.name = dataset.name
    data.num_classes = dataset.num_classes
    
    return data

def dataset_stats(data):
    print('Name:', data.name)
    print('Number of graphs:', 1)
    print('Nodes:', data.num_nodes)
    print('Edges:', data.num_edges // 2)
    print('Features:', data.num_features)
    print('Classes:', data.num_classes)
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print('Is undirected:', data.is_undirected())
