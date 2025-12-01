import torch
import torch.nn as nn


def multiclass_classify_samples(outputs):
    return torch.argmax(outputs, dim=1).cpu().numpy()


def get_classification_function():
    """Returns the classification function for multiclass mode."""
    return multiclass_classify_samples


def get_loss_criterion() -> nn.CrossEntropyLoss:
    """Returns the loss criterion for multiclass classification."""
    return nn.CrossEntropyLoss()
