import time
from model import *
from utilities import *

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
        
def run_feature_combinations(original_features, structural_features, positional_features, file_name, normalization=lambda x: x):
    features_combinations = [
      original_features, 
      structural_features, 
      positional_features, 
      concatenate(original_features,structural_features), 
      concatenate(original_features,positional_features),
      concatenate(structural_features,positional_features),
      concatenate(original_features,structural_features,positional_features)]

    file_names = [
      'original', 
      'structural', 
      'positional', 
      'original-structural', 
      'original-positional', 
      'structural-positional', 
      'original-structural-positional']

    basic_models = dict()
    for curr_features, curr_file_name in zip(features_combinations, file_names):
        data.x = curr_features
        data.x = normalization(data.x)
        results = dict()
        results['avg_acc'], results['test_accs'], results['train_losses'], results['train_accs'], results['val_losses'], results['val_accs'], results['run_times'],results['best_epoch'] = run_experiments(model, data, n_runs, n_epochs, optimizer, criterion, device) # These should be "global variables"
        results['model'] = model
     
        basic_models[curr_file_name] = results

    save_results(basic_models, file_name)
   

