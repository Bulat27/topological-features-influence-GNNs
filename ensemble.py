import torch
from experiments import eval_raw


def get_meta_model_features(models, features, original_features=True):
    # Ensure that the number of models and features match
    assert len(models) == len(features), "Number of models and features must be the same"

    # List to store raw predictions for each model
    raw_predictions_list = []

    # Obtain raw predictions for each model
    for model, data in zip(models, features):
        raw_predictions = eval_raw(model, data, data.val_mask)
        raw_predictions_list.append(raw_predictions)

    # Concatenate raw predictions along the last dimension
    concatenated_predictions = torch.cat(raw_predictions_list, dim=-1)

    # If original_features is True, concatenate original features with predictions
    if original_features:
        concatenated_features = torch.cat([concatenated_predictions] + features, dim=-1)
    else:
        concatenated_features = concatenated_predictions

    return concatenated_features






   