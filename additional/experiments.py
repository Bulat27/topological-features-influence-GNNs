import time
from models.model import *
from additional.utilities import *

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
        
   

