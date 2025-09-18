# src/evaluation.py
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def exact_match_accuracy(predictions, targets, task_type='sst'):
    """
    Calculate exact match accuracy between predictions and targets.

    Args:
        predictions: torch.Tensor of predicted labels or list of predicted decoded strings
        targets: torch.Tensor of target labels or list of target decoded strings
        task_type: str indicating the task type ('sst' for text comparison, others for tensor comparison)

    Returns:
        float: Accuracy value
    """
    # print('exact_match_accuracy, task_type:', task_type)
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions and targets must have the same length. Got {len(predictions)} and {len(targets)}")

    if task_type == 'sst':
        # Normalize text for comparison (e.g., strip, lowercase)
        predictions = [pred.strip().lower() for pred in predictions]
        targets = [target.strip().lower() for target in targets]

        # Calculate exact matches
        matches = sum(p == t for p, t in zip(predictions, targets)) # todo: check if there are extra char by tokenizer here
        return matches / len(targets)
    else:
        # Original implementation for non-text tasks
        if isinstance(predictions, torch.Tensor) and isinstance(targets, torch.Tensor):
            return (predictions == targets).float().mean().item()
        else:
            # Convert to tensors if they're not already
            if not isinstance(predictions, torch.Tensor):
                predictions = torch.tensor(predictions, device=targets.device)
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets, device=predictions.device)
            return (predictions == targets).float().mean().item()


def top_k_accuracy(true_labels: str, predictions: List[str], k: int) -> float:
    correct_predictions = 0
    for true_label, prediction in zip(true_labels, predictions):
        true_label = true_label.strip()  # Strip extra spaces from true label
        # top_k_pred = prediction.strip()
        top_k_pred = [pred.strip() for pred in prediction[:k]]  # Strip spaces from predictions

        if true_label in top_k_pred:
            correct_predictions += 1
    return correct_predictions / len(true_labels)


def top_k_precision_recall_f1(true_labels: List[str], predictions: List[List[str]], k: int):
    true_positives = 0
    total_predictions = 0
    total_actual = len(true_labels)

    for true_label, prediction in zip(true_labels, predictions):
        true_label = true_label.strip()  # Strip extra spaces from true label
        # top_k_pred = prediction.strip()
        top_k_pred = [pred.strip() for pred in prediction[:k]]  # Strip spaces from top-k predictions
        # print('true_label', true_label)
        # print('top_k_pred', top_k_pred)

        if true_label in top_k_pred:
            true_positives += 1
        total_predictions += k

    precision = true_positives / total_predictions if total_predictions > 0 else 0
    recall = true_positives / total_actual if total_actual > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def cosine_semantic_similarity(sent1: str, sent2: str, model: SentenceTransformer) -> float:
    embeddings = model.encode([sent1, sent2])
    return 1 - cosine(embeddings[0], embeddings[1])


def mean_reciprocal_rank(true_label: str, predictions: List[str]) -> float:
    if true_label in predictions:
        return 1.0 / (predictions.index(true_label) + 1)
    return 0.0


def calculate_metrics(true_labels: List[str], predicted_labels: List[List[str]], k: int):
    accuracy = top_k_accuracy(true_labels, predicted_labels, k)
    precision, recall, f1 = top_k_precision_recall_f1(true_labels, predicted_labels, k)

    sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    semantic_similarities = [cosine_semantic_similarity(true, pred[0], sentence_model) for true, pred in
                             zip(true_labels, predicted_labels)]
    mrr = np.mean([mean_reciprocal_rank(true, pred) for true, pred in zip(true_labels, predicted_labels)])

    return {
        'accuracy': accuracy,
        'precision_at_k': precision,
        'recall_at_k': recall,
        'f1_at_k': f1,
        'mean_semantic_similarity': np.mean(semantic_similarities),
        'mean_reciprocal_rank': mrr
    }


def summary_metrics(combined_results):
    # generate summary statistics of the results
    clean_accuracies = [result['clean_metrics']['accuracy'] for result in combined_results]
    grouped_accuracies = [result['grouped_metrics']['accuracy'] for result in combined_results]
    clean_precision = [result['clean_metrics']['precision_at_k'] for result in combined_results]
    grouped_precision = [result['grouped_metrics']['precision_at_k'] for result in combined_results]
    clean_recall = [result['clean_metrics']['recall_at_k'] for result in combined_results]
    grouped_recall = [result['grouped_metrics']['recall_at_k'] for result in combined_results]
    clean_f1 = [result['clean_metrics']['f1_at_k'] for result in combined_results]
    grouped_f1 = [result['grouped_metrics']['f1_at_k'] for result in combined_results]
    # clean_map = [result['clean_metrics']['map'] for result in combined_results]
    # grouped_map = [result['grouped_metrics']['map'] for result in combined_results]
    clean_mrr = [result['clean_metrics']['mean_reciprocal_rank'] for result in combined_results]
    grouped_mrr = [result['grouped_metrics']['mean_reciprocal_rank'] for result in combined_results]
    clean_semantic_similarity = [result['clean_metrics']['mean_semantic_similarity'] for result in combined_results]
    grouped_semantic_similarity = [result['grouped_metrics']['mean_semantic_similarity'] for result in combined_results]
    # average them and log them
    avg_clean_accuracy = np.mean(clean_accuracies)
    avg_grouped_accuracy = np.mean(grouped_accuracies)
    avg_clean_precision = np.mean(clean_precision)
    avg_grouped_precision = np.mean(grouped_precision)
    avg_clean_recall = np.mean(clean_recall)
    avg_grouped_recall = np.mean(grouped_recall)
    avg_clean_f1 = np.mean(clean_f1)
    avg_grouped_f1 = np.mean(grouped_f1)
    # avg_clean_map = np.mean(clean_map)
    # avg_grouped_map = np.mean(grouped_map)
    avg_clean_mrr = np.mean(clean_mrr)
    avg_grouped_mrr = np.mean(grouped_mrr)
    avg_clean_semantic_similarity = np.mean(clean_semantic_similarity)
    avg_grouped_semantic_similarity = np.mean(grouped_semantic_similarity)
    # calculate the standard deviation
    std_clean_accuracy = np.std(clean_accuracies)
    std_grouped_accuracy = np.std(grouped_accuracies)
    std_clean_precision = np.std(clean_precision)
    std_grouped_precision = np.std(grouped_precision)
    std_clean_recall = np.std(clean_recall)
    std_grouped_recall = np.std(grouped_recall)
    std_clean_f1 = np.std(clean_f1)
    std_grouped_f1 = np.std(grouped_f1)
    # std_clean_map = np.std(clean_map)
    # std_grouped_map = np.std(grouped_map)
    std_clean_mrr = np.std(clean_mrr)
    std_grouped_mrr = np.std(grouped_mrr)
    std_clean_semantic_similarity = np.std(clean_semantic_similarity)
    std_grouped_semantic_similarity = np.std(grouped_semantic_similarity)

    summary = {
        'avg_clean_accuracy': avg_clean_accuracy,
        'avg_grouped_accuracy': avg_grouped_accuracy,
        'avg_clean_precision': avg_clean_precision,
        'avg_grouped_precision': avg_grouped_precision,
        'avg_clean_recall': avg_clean_recall,
        'avg_grouped_recall': avg_grouped_recall,
        'avg_clean_f1': avg_clean_f1,
        'avg_grouped_f1': avg_grouped_f1,
        # 'avg_clean_map': avg_clean_map,
        # 'avg_grouped_map': avg_grouped_map,
        'avg_clean_mrr': avg_clean_mrr,
        'avg_grouped_mrr': avg_grouped_mrr,
        'avg_clean_semantic_similarity': avg_clean_semantic_similarity,
        'avg_grouped_semantic_similarity': avg_grouped_semantic_similarity,
        'std_clean_accuracy': std_clean_accuracy,
        'std_grouped_accuracy': std_grouped_accuracy,
        'std_clean_precision': std_clean_precision,
        'std_grouped_precision': std_grouped_precision,
        'std_clean_recall': std_clean_recall,
        'std_grouped_recall': std_grouped_recall,
        'std_clean_f1': std_clean_f1,
        'std_grouped_f1': std_grouped_f1,
        # 'std_clean_map': std_clean_map,
        # 'std_grouped_map': std_grouped_map,
        'std_clean_mrr': std_clean_mrr,
        'std_grouped_mrr': std_grouped_mrr,
        'std_clean_semantic_similarity': std_clean_semantic_similarity,
        'std_grouped_semantic_similarity': std_grouped_semantic_similarity
    }
    return summary


def calculate_metrics_testing(predictions, targets, task_type='sst', average='weighted'):
    """
    Calculate precision, recall, and F1 score between predictions and targets.

    Args:
        predictions: torch.Tensor or list of predicted labels/strings
        targets: torch.Tensor or list of target labels/strings
        task_type: str indicating the task type ('sst' for text comparison, others for tensor comparison)
        average: str indicating the averaging method for metrics ('micro', 'macro', 'weighted')

    Returns:
        dict: Dictionary containing precision, recall, and F1 score
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions and targets must have the same length. Got {len(predictions)} and {len(targets)}")

    if task_type == 'sst':
        # Normalize text for comparison
        predictions = [pred.strip().lower() for pred in predictions]
        targets = [target.strip().lower() for target in targets]

    # If task is not 'sst', assume tensor-based metrics
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # Calculate metrics
    precision = precision_score(targets, predictions, average=average)
    recall = recall_score(targets, predictions, average=average)
    f1 = f1_score(targets, predictions, average=average)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
