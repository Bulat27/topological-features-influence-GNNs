import torch

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
        

