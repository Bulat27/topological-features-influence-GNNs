from torch_geometric.nn import Node2Vec
from tqdm.notebook import tqdm
from datasets import *
from features import *

def train(model,optimizer,loader,device):
    model.train()  # put model in train model
    total_loss = 0
    for pos_rw, neg_rw in tqdm(loader):
        optimizer.zero_grad()  # set the gradients to 0
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))  # compute the loss for the batch
        loss.backward()
        optimizer.step()  # optimize the parameters
        total_loss += loss.item()
    return total_loss / len(loader)

# return a tensor with the positional features of size: (num_nodes, num_features)
def run_training(data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # check if cuda is available to send the model and tensors to the GPU
    model = Node2Vec(
        data.edge_index,
        embedding_dim=32, 
        walk_length=40, 
        context_size=10, 
        walks_per_node=10,
        num_negative_samples=1, p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)  # data loader to speed the train 
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)  # initzialize the optimizer 
    for epoch in range(1,6):
        loss = train(model,optimizer,loader,device)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    
    return model(torch.arange(data.num_nodes, device=device)).detach()
