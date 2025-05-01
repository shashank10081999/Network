from networksecurity.entity.artifact_entity import ClassificationArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os
import sys
from sklearn.metrics import f1_score, precision_score, recall_score


def get_classification_metric(y_true, y_pred) -> ClassificationArtifact:
    """
    Calculate classification metrics and return them as an artifact.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        ClassificationArtifact: Object containing classification metrics.
    """
    try:
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        classification_artifact = ClassificationArtifact(
            f1_score=f1,
            precision=precision,
            recall=recall
        )
        return classification_artifact

    except Exception as e:
        raise NetworkSecurityException(e, sys)