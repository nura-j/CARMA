import json
import os
import torch
import numpy as np
from tqdm import tqdm
from transformer_lens import HookedTransformer
from src.dataset import TokenSpan
from src.utils import get_last_token_logits
from src.evaluation import exact_match_accuracy, calculate_metrics_testing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from scipy.stats import chi2_contingency
import pandas as pd


def calculate_next_token_metrics(logits, labels, predictions, input_tokens=None):
    """Calculate metrics specific to next token prediction."""
    probs = torch.softmax(logits, dim=1)

    # Calculate perplexity
    cross_entropy = torch.nn.functional.cross_entropy(logits, labels)
    perplexity = torch.exp(cross_entropy).item()

    # Calculate token entropy
    entropy = -torch.sum(probs * torch.log_softmax(logits, dim=1), dim=1).mean().item()

    # Calculate prediction ranks
    ranks = []
    for prob, label in zip(probs, labels):
        rank = (prob.argsort(descending=True) == label).nonzero().item() + 1
        ranks.append(rank)

    # Mean Reciprocal Rank
    mrr = np.mean([1 / rank for rank in ranks])

    # Vocabulary diversity
    vocab_diversity = len(set(predictions)) / len(predictions)

    return {
        "perplexity": float(perplexity),
        "token_entropy": float(entropy),
        "mean_reciprocal_rank": float(mrr),
        "vocabulary_diversity": float(vocab_diversity),
        "prediction_ranks": ranks
    }


def calculate_calibration_metrics(confidences, correct_predictions, n_bins=10):
    """Calculate Expected Calibration Error (ECE) and other calibration metrics."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    calibration_stats = []
    ece = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        if any(in_bin):
            bin_conf = np.mean(confidences[in_bin])
            bin_acc = np.mean(correct_predictions[in_bin])
            bin_size = np.sum(in_bin) / len(confidences)
            ece += bin_size * abs(bin_acc - bin_conf)

            calibration_stats.append({
                'bin_lower': float(bin_lower),
                'bin_upper': float(bin_upper),
                'confidence': float(bin_conf),
                'accuracy': float(bin_acc),
                'samples': int(np.sum(in_bin))
            })

    return {
        'ece': float(ece),
        'calibration_bins': calibration_stats
    }


def calculate_statistical_metrics(predictions, labels, confidences):
    """Calculate statistical significance metrics."""
    try:
        chi_square = float(chi2_contingency(
            pd.crosstab(pd.Series(labels), pd.Series(predictions))
        )[0])
    except:
        chi_square = None

    correlation = float(np.corrcoef(
        confidences,
        [1 if p == l else 0 for p, l in zip(predictions, labels)]
    )[0, 1])

    return {
        'chi_square': chi_square,
        'confidence_correlation': correlation
    }


def test_model(
        model,
        tokenizer=None,
        test_dataloader=None,
        model_name: str = 'GPT2',
        device = 'cpu',
        save_results: bool = True,
        save_path: str = None,
        task_name: str = 'idm',
        intervention_type: str = 'original',
        seed: int = 42,
        replacement_percentage=0.4,
        k: int = 5
):
    """Test the model on the test set."""
    model = model.eval()
    print('Testing model...')
    print(f'Intervention type: {intervention_type}')

    # Initialize storage
    all_predictions = []
    all_labels = []
    all_inputs = []
    all_attention_masks = []
    all_annotations = []
    all_reduction_percentages = []
    all_top_k_predictions = []
    all_confidences = []
    all_top_k_confidences = []
    all_logits = []  # Add this to store logits

    with torch.no_grad():
        progress_bar = tqdm(test_dataloader, desc="Testing")

        for batch in progress_bar:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

            if isinstance(model, HookedTransformer):
                logits = model(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    return_type="logits",
                )
            else:
                logits = model(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                )['logits']

            labels = batch['labels'].to(logits.device)
            last_token_logits = get_last_token_logits(logits, batch['attention_mask'])
            # all_logits.append(last_token_logits)
            all_logits.append(last_token_logits.detach().cpu())  # Detach and move to CPU

            # Get predictions and confidences
            probs = torch.softmax(last_token_logits, dim=1)
            top_k_values, top_k_indices = torch.topk(probs, k=k, dim=1)
            predictions = top_k_indices[:, 0]

            # Store confidences
            batch_confidences = top_k_values[:, 0].cpu().tolist()
            batch_top_k_confidences = top_k_values.cpu().tolist()

            if task_name == 'sst':
                pred_labels_text = tokenizer.batch_decode(predictions.tolist(), skip_special_tokens=True)
                labels_text = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)
                pred_labels_text = [label.strip().lower() for label in pred_labels_text]

                top_k_text = [
                    [tokenizer.decode(idx, skip_special_tokens=True).strip().lower()
                     for idx in pred_k]
                    for pred_k in top_k_indices.cpu().tolist()
                ]

                all_predictions.extend(pred_labels_text)
                all_labels.extend(labels_text)
                all_top_k_predictions.extend(top_k_text)
            else:
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_top_k_predictions.extend(top_k_indices.cpu().tolist())

            all_inputs.extend(batch['input_ids'].cpu().tolist())
            all_attention_masks.extend(batch['attention_mask'].cpu().tolist())
            all_annotations.extend(batch['annotations'])
            all_confidences.extend(batch_confidences)
            all_top_k_confidences.extend(batch_top_k_confidences)

            if intervention_type != 'original':
                all_reduction_percentages.extend(batch['reduction_percentage'])

            # Update progress bar with batch accuracy
            if task_name == 'sst':
                batch_accuracy = exact_match_accuracy(pred_labels_text, labels_text, task_type='sst')
            else:
                batch_accuracy = exact_match_accuracy(predictions, labels, task_type=task_name)

            progress_bar.set_postfix(batch_accuracy=batch_accuracy)

        # Calculate metrics
        if task_name == 'sst':
            overall_accuracy = exact_match_accuracy(all_predictions, all_labels, task_type='sst')
        else:
            overall_accuracy = accuracy_score(all_labels, all_predictions)

        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        # Calculate top-k accuracy
        correct_in_top_k = sum(
            1 for preds, label in zip(all_top_k_predictions, all_labels)
            if label in preds
        )
        top_k_accuracy = correct_in_top_k / len(all_labels)

        # Calculate confidence metrics
        confidence_metrics = {
            'exact_match': {
                'mean_confidence': float(np.mean(all_confidences)),
                'median_confidence': float(np.median(all_confidences)),
                'std_confidence': float(np.std(all_confidences)),
                'confidence_percentiles': {
                    '25': float(np.percentile(all_confidences, 25)),
                    '75': float(np.percentile(all_confidences, 75)),
                    '90': float(np.percentile(all_confidences, 90))
                }
            },
            'top_k': {
                'mean_confidence': float(np.mean([c for confs in all_top_k_confidences for c in confs])),
                'median_confidence': float(np.median([c for confs in all_top_k_confidences for c in confs])),
                'std_confidence': float(np.std([c for confs in all_top_k_confidences for c in confs])),
                'confidence_percentiles': {
                    '25': float(np.percentile([c for confs in all_top_k_confidences for c in confs], 25)),
                    '75': float(np.percentile([c for confs in all_top_k_confidences for c in confs], 75)),
                    '90': float(np.percentile([c for confs in all_top_k_confidences for c in confs], 90))
                }
            }
        }

        # Calculate calibration and statistical metrics
        calibration_metrics = calculate_calibration_metrics(
            np.array(all_confidences),
            np.array(all_predictions) == np.array(all_labels)
        )

        statistical_metrics = calculate_statistical_metrics(
            all_predictions, all_labels, all_confidences
        )

        # Serialize annotations
        if intervention_type == 'original':
            serialized_annotations = [
                {
                    "token_spans": [
                        {
                            "start_idx": span.start_idx,
                            "end_idx": span.end_idx,
                            "token_ids": span.token_ids,
                            "token_texts": span.token_texts,
                            "original_word": span.original_word
                        }
                        for span in annotation["token_spans"]
                    ],
                    "reduction_percentage": annotation["reduction_percentage"]
                }
                for annotation in all_annotations
            ]
        else:
            serialized_annotations = [
                {
                    "token_spans": [
                        {
                            "start_idx": span.start_idx,
                            "end_idx": span.end_idx,
                            "token_ids": span.token_ids,
                            "token_texts": span.token_texts,
                            "original_word": span.original_word
                        }
                        for span in annotation
                    ],
                    "reduction_percentage": reduction_percentage
                }
                for annotation, reduction_percentage in zip(all_annotations, all_reduction_percentages)
            ]

        # Create correct predictions list (original format)
        if task_name == 'sst':
            correct_predictions = [
                {
                    "input_ids": inp,
                    "label": lbl,
                    "prediction": pred,
                    "attention_mask": attn_mask,
                    "annotations": annotation,
                }
                for inp, lbl, pred, attn_mask, annotation in zip(
                    all_inputs, all_labels, all_predictions, all_attention_masks, serialized_annotations
                )
                if lbl.strip().lower() == pred.strip().lower()
            ]
        else:
            correct_predictions = [
                {
                    "input_ids": inp,
                    "label": lbl,
                    "prediction": pred,
                    "attention_mask": attn_mask,
                    "annotations": annotation,
                }
                for inp, lbl, pred, attn_mask, annotation in zip(
                    all_inputs, all_labels, all_predictions, all_attention_masks, serialized_annotations
                )
                if lbl == pred
            ]

        print('Number of correct predictions:', len(correct_predictions))
        print(f"Accuracy: {overall_accuracy:.4f}")
        print(f"Top-{k} Accuracy: {top_k_accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Save results
        if save_results:
            print('Saving results...')
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            directory = os.path.join(base_dir, f'results/{model_name}/{task_name}/test_results')
            os.makedirs(directory, exist_ok=True)

            if intervention_type == 'original':
                save_path = save_path or f"{directory}/{model_name}_{task_name}_test_results.json"
                metrics_save_path = save_path.replace('test_results.json', 'metrics.json')
            else:
                replacement_percentage_str = str(replacement_percentage * 100).replace('.', '_')
                save_path = save_path or f"{directory}/{model_name}_{task_name}_{intervention_type}_seed_{seed}_rep_{replacement_percentage_str}_test_results.json"
                metrics_save_path = save_path.replace('test_results.json', 'metrics.json')

            # Save original format results (correct predictions only)
            print(f"Saving correct predictions to {save_path}")
            with open(save_path, "w") as f:
                json.dump(correct_predictions, f, indent=4)
            # all_logits = torch.cat(all_logits, dim=0)
            all_logits_tensor = torch.cat(all_logits, dim=0).to(device)

            # Save all metrics to separate file
            metrics_dict = {
                "metadata": {
                    "model_name": model_name,
                    "task_name": task_name,
                    "intervention_type": intervention_type,
                    "seed": seed,
                    "k": k,
                    "replacement_percentage": replacement_percentage
                },
                "basic_metrics": {
                    "total_samples": len(all_labels),
                    "correct_predictions": len(correct_predictions),
                    "accuracy": float(overall_accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                },
                "extended_metrics": {
                    "top_k_accuracy": float(top_k_accuracy),
                    "correct_in_top_k": correct_in_top_k,
                    "confidence_analysis": confidence_metrics,
                    "calibration": calibration_metrics,
                    "statistical_tests": statistical_metrics,
                },
                "next_token_metrics": calculate_next_token_metrics(
                    logits=all_logits_tensor,
                    labels=torch.tensor(all_labels).to(device),
                    predictions=all_predictions
                )
            }

            print(f"Saving metrics to {metrics_save_path}")
            with open(metrics_save_path, "w") as f:
                json.dump(metrics_dict, f, indent=4)

        # Convert tensor metrics to Python scalars
        if isinstance(overall_accuracy, torch.Tensor):
            overall_accuracy = overall_accuracy.item()
        if isinstance(precision, torch.Tensor):
            precision = precision.item()
        if isinstance(recall, torch.Tensor):
            recall = recall.item()
        if isinstance(f1, torch.Tensor):
            f1 = f1.item()
        if isinstance(top_k_accuracy, torch.Tensor):
            top_k_accuracy = top_k_accuracy.item()

        return overall_accuracy, precision, recall, f1, len(correct_predictions), save_path, top_k_accuracy
