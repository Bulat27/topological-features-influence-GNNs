import torch
import time

# Either add the variance among the accuracies in different runs here or remember to measure it!
def run_experiments(model, data, n_runs, n_epochs, optimizer, criterion, device):
  data = data.to(device)
  model = model.to(device)
  test_accs, run_times = [], []

  for i in range(1 , n_runs + 1):
    print(f"\n RUN: {i}\n")

    reset_weights(model)

    start_time = time.time()
    train_losses, train_accs, val_losses, val_accs, best_epoch = model_training(n_epochs, model, data, optimizer, criterion)
    end_time = time.time()

    run_times.append(end_time - start_time)

    model.load_state_dict(torch.load('best_model.pth'))
    test_acc, _  = eval(model, data, data.test_mask, criterion)
    test_accs.append(test_acc)
    avg_acc = sum(test_accs) / len(test_accs)

  return avg_acc, test_accs, train_losses, train_accs, val_losses, val_accs, run_times, best_epoch

def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        else:
            raise AttributeError(f'The layer {layer.__class__.__name__} does not have a reset_parameters method.')    


def model_training(n_epochs, model, data, optimizer, criterion):
  train_losses = []
  train_accs = []
  val_losses = []
  val_accs = []
  best_val_acc = 0.0
  best_epoch = 0

  for epoch in range(1, n_epochs + 1):
    train_acc, train_loss = train(model, data, optimizer, criterion)
    val_acc, val_loss = eval(model, data, data.val_mask, criterion)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_model.pth')

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

  return train_losses, train_accs, val_losses, val_accs, best_epoch

def train(model, data, optimizer, criterion):
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      pred = out.argmax(dim=1)
      correct = pred[data.train_mask] == data.y[data.train_mask]
      acc = int(correct.sum()) / int(data.train_mask.sum())
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return acc, loss.item()

def eval(model, data, data_mask, criterion):
      # We need torch.nograd here!!!
      model.eval()

      with torch.no_grad():
          out = model(data.x, data.edge_index)
          pred = out.argmax(dim=1)  # Use the class with highest probability.
          correct = pred[data_mask] == data.y[data_mask]  # Check against ground-truth labels.
          acc = int(correct.sum()) / int(data_mask.sum())  # Derive ratio of correct predictions.
          loss = criterion(out[data_mask], data.y[data_mask])
          
      return acc, loss.item()

def eval_raw(model, data, data_mask):
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        out = out[data_mask]

    return out    
        

