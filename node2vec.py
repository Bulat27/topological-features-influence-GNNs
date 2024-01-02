from torch_geometric.nn import Node2Vec
from tqdm.notebook import tqdm
from datasets import *
from features import *
import random
import numpy as np

def train(model,optimizer,loader,device):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in tqdm(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))  # compute the loss for the batch
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# return a tensor with the positional features of size: (num_nodes, embedding_dim)
def run_training(data,n_iterations, embedding_dim):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = []
    losses = np.empty(n_iterations)
    for i in range(n_iterations):
        print(f"\nRANDOM SEARCH ITERATION: {i+1}")
        model = Node2Vec(
            data.edge_index,
            embedding_dim=embedding_dim,
            walk_length=random.randint(30,100), 
            context_size=random.randint(1,30), 
            walks_per_node=random.randint(1,30),
            num_negative_samples=1, 
            p=random.random(), 
            q=random.random(), 
            sparse=True).to(device)
    
        loader = model.loader(batch_size=128, shuffle=True, num_workers=4)  # data loader to speed the train 
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
        for epoch in range(1,30+1):
            loss = train(model,optimizer,loader,device)
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        
        losses[i] = loss
        models.append(model)
    
    best_model_idx = np.argmin(losses)
    best_model = models[best_model_idx]
    
    return best_model, best_model(torch.arange(data.num_nodes, device=device)).detach()
