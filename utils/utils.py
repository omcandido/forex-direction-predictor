from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix
import torch.nn as nn
import numpy as np

def metrics_batch(preds_batch, labels_batch, confidence=0.5):
    """
    Calculate metrics and counts for a batch of predictions.

    Args:
        preds_batch (torch.Tensor): Batch of predicted values.
        labels_batch (torch.Tensor): Batch of true labels.
        confidence (float, optional): Confidence threshold for considering a prediction as confident. Defaults to 0.5.

    Returns:
        tuple: A tuple containing the following:
            - metrics (tuple): Precision, recall, F1-score, and support for each label.
            - counts (numpy.ndarray): Confusion matrix for each label.

    """
    probs_batch = nn.Softmax(dim=1)(preds_batch)
    choices_batch = probs_batch.argmax(dim=1)
    confident = (probs_batch >= confidence).sum(dim=1) > 0
    choices_batch[~confident] = 0
    metrics = precision_recall_fscore_support(labels_batch.cpu(), choices_batch.cpu(), labels=(1,2), zero_division=np.nan) # type: ignore
    counts = multilabel_confusion_matrix(labels_batch.cpu(), choices_batch.cpu(), labels=(1,2))
    return metrics, counts