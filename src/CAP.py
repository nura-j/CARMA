import json
import logging
import os
from datetime import datetime
import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer
from src.evaluation import calculate_metrics, summary_metrics
from src.utils import get_last_token_logits, NumpyEncoder
from src.model_setup import set_model_name
from src.dataset import group_batches_by_length, create_batch_ranges
from src.load_saved_predictions import load_saved_predictions


def cap(model: HookedTransformer,
        tokenizer=None,
        test_dataloader=None,
        correct_predictions_path: str = None,
        model_name: str = 'GPT2',
        supervision_type: str = '',
        task_type: str = 'inverse_dictionary',
        component: str = None,
        start_layer: int = 4,
        seed: int = 42,
        k: int = 1,  # top k predictions
        grouping_protocol: str = 'mean',
        granularity: str = 'tw',  # the level of words grouping
        batch_size: int = 2,
        device: str = 'cpu'):

    # Assert the task type is valid
    assert task_type in ['idm', 'sst', 'mp'], 'Invalid task type'

    # Set the model to evaluation mode
    model.eval()
    # Set the model name and logging path and file
    model_name = set_model_name(model_name)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    log_path = os.path.join(base_dir, f'results/{model_name}/{task_type}/cap_results')
    # log_path = f'results/{model_name}/{task_type}'
    os.makedirs(log_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    granularity = 'tp_logs' if granularity == 'token_to_phrases' else 'tw_logs'
    log_file_json_path = os.path.join(base_dir, f'results/{model_name}/{task_type}/cap_results/{granularity}/')
    log_file_json = f'{log_file_json_path}/llm_evaluation_results_{model_name}_{supervision_type}_{task_type}_seed_{seed}_{grouping_protocol}_{timestamp}_starting_{start_layer}.json'
    os.makedirs(log_file_json_path, exist_ok=True)

    print('Start the loading of the data')
    if correct_predictions_path:
        test_dataloader = load_saved_predictions(correct_predictions_path, batch_size=batch_size)

    print('Data loaded')
    grouped_batches = group_batches_by_length(test_dataloader, batch_size=batch_size)
    batches_ranges = create_batch_ranges(grouped_batches)
    model_number_of_layers = model.cfg.n_layers
    result_collection = []

    def create_hook_filter(components, layer_range, hook_types):
        return lambda name: name in components and any(
            [f".{i}." in name for i in range(layer_range[0], layer_range[1])]) and any(
            hook in name for hook in hook_types)

    for length, batch_groups in grouped_batches.items():
        for batch_idx, batch in enumerate(batch_groups):
            clean_predictions, clean_labels, grouped_predictions, grouped_labels, reduction_percentage = [], [], [], [], []
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            batch_inputs = batch['input_ids']  # batch_inputs = batch['input_ids'].to(device)
            batch_labels = batch['labels']  # batch_labels = batch['labels'].to(device)
            batch_attention_mask = batch['attention_mask']  # batch_attention_mask = batch['attention_mask'].to(device)
            batch_annotations = batch['annotations']
            batch_reduction_percentage = batch['reduction_percentage']

            # Run the model on the batch
            logits, cache = model.run_with_cache(batch_inputs, attention_mask=batch_attention_mask,
                                                 return_type="logits")  # Clean run

            # Process clean predictions
            for i, annotations in enumerate(batch_annotations):
                last_token_logits = get_last_token_logits(logits, batch_attention_mask) # last token before padding, only top 1 not k predictions
                pred_labels = last_token_logits.argmax(dim=-1)
                # predicted_tokens_c = [model.to_string(pred_labels[i])]
                predicted_tokens_c = [tokenizer.decode(pred_labels[i], skip_special_tokens=True)]
                # label_tokens = model.to_string(batch_labels[i])
                label_tokens = tokenizer.decode(batch_labels[i], skip_special_tokens=True)
                clean_predictions.append(predicted_tokens_c)
                clean_labels.append(label_tokens)

            # for debugging purposes - print the logits shape or predictions
            # print(f'logits shape: {logits.shape}')
            # print(f'Predicted: {predicted_tokens_c}, Label: {label_tokens}')
            # print(f'Labels: {label_tokens}')

            # Define components and hook filters
            components = [name for name in cache.keys() if 'embed' not in name] if not component else [component]
            layer_range = (start_layer, model_number_of_layers)

            mlp_hook_filter = create_hook_filter(components, layer_range, ['mlp'])
            resid_hook_filter = create_hook_filter(components, layer_range, ['resid', 'hook_normalized', 'hook_scale'])
            attn_kqvz_hook_filter = create_hook_filter(components, layer_range,
                                                       ['hook_k', 'hook_q', 'hook_v', 'hook_z'])
            attn_scores_pattern_hook_filter = create_hook_filter(components, layer_range,
                                                                 ['hook_attn_scores', 'hook_pattern'])
            attn_out_hook_filter = create_hook_filter(components, layer_range, ['hook_attn_out'])
            final_hook_filter = create_hook_filter(components, layer_range, ['ln_final'])

            batch_range = batches_ranges[length][batch_idx]
            print(f"Starting the grouping run for batch {batch_idx} of length {length}")

            # Define grouping functions
            def group_mlp_resid_attn_out_layers(tensor: Float, protocol: str = grouping_protocol,
                                                ranges_r: list = batch_range, **kwargs):
                batch_size, sequence_length, hidden_size = tensor.shape
                new_seq_len = len(ranges_r[0])
                if tensor.shape == (batch_size, new_seq_len, hidden_size):
                    return tensor
                new_tensor = torch.zeros((batch_size, new_seq_len, hidden_size), device=tensor.device)
                for batch_idx in range(batch_size):
                    for i, (start, end) in enumerate(ranges_r[batch_idx]):
                        if protocol == 'sum':
                            new_tensor[batch_idx, i] = tensor[batch_idx, start:end].sum(dim=0)
                        elif protocol == 'mean':
                            new_tensor[batch_idx, i] = tensor[batch_idx, start:end].mean(dim=0)
                        elif protocol == 'max':
                            new_tensor[batch_idx, i] = tensor[batch_idx, start:end].max(dim=0).values
                        else:
                            raise ValueError(f"Invalid protocol: {protocol}")
                return new_tensor

            def group_attn_scores_pattern(tensor: Float, protocol: str = grouping_protocol,
                                          ranges_r: list = batch_range, **kwargs):
                batch_size, num_heads, seq_len, _ = tensor.shape
                new_seq_len = len(ranges_r[0])
                new_tensor = torch.zeros((batch_size, num_heads, new_seq_len, new_seq_len), device=tensor.device)
                if tensor.shape == (batch_size, num_heads, new_seq_len, new_seq_len):
                    return tensor
                for batch_idx, flattened_batch in enumerate(tensor):
                    r = ranges_r[batch_idx]
                    new_part = torch.zeros((num_heads, new_seq_len, new_seq_len), device=tensor.device)
                    for i, (start, end) in enumerate(r):
                        for j, (start_, end_) in enumerate(r):
                            if protocol == 'sum':
                                new_part[:, i, j] = flattened_batch[:, start:end, start_:end_].sum(dim=1).sum(dim=1)
                            elif protocol == 'mean':
                                new_part[:, i, j] = flattened_batch[:, start:end, start_:end_].mean(dim=1).mean(dim=1)
                            elif protocol == 'max':
                                new_part[:, i, j] = flattened_batch[:, start:end, start_:end_].max(dim=1).values.max(
                                    dim=1).values
                            else:
                                raise ValueError(f"Invalid protocol: {protocol}")
                    new_tensor[batch_idx] = new_part
                return new_tensor

            def group_qkvz(tensor: Float, protocol: str = grouping_protocol, ranges_r: list = batch_range,
                           **kwargs):
                batch_size, seq_len, no_heads, hidden_size = tensor.shape
                new_seq_len = len(ranges_r[0])
                if tensor.shape == (batch_size, new_seq_len, no_heads, hidden_size):
                    return tensor
                new_tensor = torch.zeros((batch_size, new_seq_len, no_heads, hidden_size), device=tensor.device)
                for batch_idx in range(batch_size):
                    for i, (start, end) in enumerate(ranges_r[batch_idx]):
                        if protocol == 'sum':
                            new_tensor[batch_idx, i] = tensor[batch_idx, start:end].sum(dim=0)
                        elif protocol == 'mean':
                            new_tensor[batch_idx, i] = tensor[batch_idx, start:end].mean(dim=0)
                        elif protocol == 'max':
                            new_tensor[batch_idx, i] = tensor[batch_idx, start:end].max(dim=0).values
                        else:
                            raise ValueError(f"Invalid protocol: {protocol}")
                return new_tensor

            # Run model with hooks
            logits = model.run_with_hooks(
                    batch_inputs,
                    return_type='logits',
                    fwd_hooks=[
                        (mlp_hook_filter, group_mlp_resid_attn_out_layers),
                        (resid_hook_filter, group_mlp_resid_attn_out_layers),
                        (attn_scores_pattern_hook_filter, group_attn_scores_pattern),
                        (attn_kqvz_hook_filter, group_qkvz),
                        (attn_out_hook_filter, group_mlp_resid_attn_out_layers),
                        (final_hook_filter, group_mlp_resid_attn_out_layers)
                    ]
            )
            # print(f'Grouped logits shape: {logits.shape}')
            # Process grouped predictions
            for i, token_end_idx in enumerate(batch_annotations):
                top_k_values, top_k_indices = torch.topk(logits[i, -1, :], k)
                # predicted_tokens = [model.to_string(top_k_indices[0])]
                predicted_tokens = [tokenizer.decode(top_k_indices[0], skip_special_tokens=True)]
                # label_tokens = model.to_string(batch_labels[i])
                label_tokens = tokenizer.decode(batch_labels[i], skip_special_tokens=True)
                grouped_predictions.append(predicted_tokens)
                reduction_percentage.append(token_end_idx[-1])
                grouped_labels.append(label_tokens)

            # for debugging purposes - print the logits shape or predictions
            # print(f'logits shape: {logits.shape}')
            # print(f'Predicted: {predicted_tokens}, Label: {label_tokens}')
            # print(f'Labels: {label_tokens}')


            # Calculate metrics
            clean_metrics = calculate_metrics(clean_labels, clean_predictions, k)
            grouped_metrics = calculate_metrics(grouped_labels, grouped_predictions, k)
            print(f"Finished the grouping run for batch {batch_idx} of length {length}")
            # Prepare results
            results = {
                'model_name': model_name,
                'supervision_type': supervision_type,
                'task_type': task_type,
                # 'original_shape': model.cfg.n_positions,
                'sequence_length': length,
                'granularity': granularity,
                'start_layer': start_layer,
                'component': component,
                'grouping_protocol': grouping_protocol,
                'grouped_shape': len(batch_range[0]),
                'seed': seed,
                'k': k,
                'clean_metrics': clean_metrics,
                'grouped_metrics': grouped_metrics,
                'grouped_predictions': grouped_predictions,
                'original_labels': grouped_labels,
                'batch_size': len(batch_inputs),
                'reduction_ration': batch_reduction_percentage,
            }
            result_collection.append(results)
            logging.info(json.dumps(results, indent=2, cls=NumpyEncoder))
            model.reset_hooks()
        print(f"Evaluation complete for length {length}. Results have been logged to '{log_file_json}'.")
    # Log the summary results
    results_summary = summary_metrics(result_collection)
    results_summary['granularity'] = granularity
    results_summary['start_layer'] = start_layer
    results_summary['component'] = component
    results_summary['grouping_protocol'] = grouping_protocol
    results_summary['seed'] = seed
    results_summary['k'] = k
    results_summary['model_name'] = model_name
    logging.info(json.dumps(results_summary, indent=2, cls=NumpyEncoder))
    result_collection.append({"summary": results_summary})
    print("Summary results added to log.")

    # Ensure all directories in the path exist
    os.makedirs(os.path.dirname(log_file_json), exist_ok=True)
    # Write the collected results to the JSON file
    with open(log_file_json, 'w') as json_file:
        json.dump(result_collection, json_file, indent=4, cls=NumpyEncoder)
    print(f"Results have been saved to '{log_file_json}'.")
    print('CAP evaluation complete.')