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





   