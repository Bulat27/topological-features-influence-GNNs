import torch

def get_meta_model_features(models, features, data_mask, edge_index, original_features=True):
    # Ensure that the number of models and features match
    assert len(models) == len(features), "Number of models and features must be the same"

    # List to store raw predictions for each model
    raw_predictions_list = []

    # Obtain raw predictions for each model
    for model, data_x in zip(models, features):
        raw_predictions = eval_raw(model, data_x, edge_index, data_mask)
        raw_predictions_list.append(raw_predictions)

    # Concatenate raw predictions along the last dimension
    concatenated_predictions = torch.cat(raw_predictions_list, dim=-1)

    # If original_features is True, concatenate original features with predictions
  
    if original_features:
        concatenated_features = torch.cat(features, dim=-1)
        concatenated_features = concatenated_features[data_mask]
        concatenated_features = torch.cat((concatenated_features, concatenated_predictions), dim=-1)
    else:
        concatenated_features = concatenated_predictions

    return concatenated_features


def eval_raw(model, features, edge_index, data_mask):
    model.eval()

    with torch.no_grad():
        out = model(features, edge_index)
        out = out[data_mask]

    return out  

# split_data
# Here, we could also parametrize the split percentage that can be fine-tuned depending on the complexity
# of the base models and the meta model.
def get_data_split(data):
    val_mask = data.val_mask

    # Calculate the number of true values in val_mask
    num_true_values = val_mask.sum().item()

    # Calculate the number of true values for each split
    num_true_values_20_percent = int(0.2 * num_true_values)
    
    # Generate indices of true values in val_mask
    true_indices = torch.nonzero(val_mask).view(-1)

    # Shuffle the true indices randomly once
    shuffled_indices = true_indices[torch.randperm(num_true_values)]

    # Split the shuffled indices into two parts
    indices_20_percent = shuffled_indices[:num_true_values_20_percent]
    indices_80_percent = shuffled_indices[num_true_values_20_percent:]

    new_mask_20_percent = torch.zeros_like(val_mask, dtype=torch.bool)
    new_mask_80_percent = torch.zeros_like(val_mask, dtype=torch.bool)

    new_mask_20_percent[indices_20_percent] = True
    new_mask_80_percent[indices_80_percent] = True

    data.val_mask = new_mask_20_percent
    data.ensemble_val_mask = new_mask_80_percent

    return data

# def split_data(indices_20_percent, indices_80_percent, data):
#     val_mask = data.val_mask

#     new_mask_20_percent = torch.zeros_like(val_mask, dtype=torch.bool)
#     new_mask_80_percent = torch.zeros_like(val_mask, dtype=torch.bool)

#     new_mask_20_percent[indices_20_percent] = True
#     new_mask_80_percent[indices_80_percent] = True

#     data.val_mask = new_mask_20_percent
#     data.ensemble_val_mask = new_mask_80_percent

#     return data



    
    






   