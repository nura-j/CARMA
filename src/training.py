# src/training.py
from collections import defaultdict
from datetime import datetime
from typing import Optional
from transformers import get_scheduler
from tqdm import tqdm
from src.optm_losses import StabilityLoss, MILoss
from transformer_lens import HookedTransformer
import torch
from src.evaluation import exact_match_accuracy
import os
from src.model_setup import ActivationCache, filter_layers, generate_hook_keys
from src.utils import set_model_name

torch.set_grad_enabled(True)


def fine_tune_transformer(
        model,
        tokenizer,
        train_dataloader,
        val_dataloader,
        model_name: str = 'GPT2',
        learning_rate: float = 1e-4,
        num_epochs: int = 3,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
        warmup_steps: int = 100,
        stability_weight: float = 0.3,
        mi_weight: float = 0.2,
        carma_weight: float = 0.2,
        stability_start_layer: int = 0,
        stability_end_layer: Optional[int] = None,
        mi_start_layer: int = 0,
        mi_end_layer: Optional[int] = None,
        save_model: bool = True,
        save_path: str = None,
        task_name: str = 'idm',
        granularity: str = 'tw',
        seed: int = 42,
):
    model = model.train()
    model_type = None
    if 'gpt' in model_name.lower():
        model_type = 'GPT'
    elif 'llama' in model_name.lower():
        model_type = 'Llama'
    elif 'gemma' in model_name.lower():
        model_type = 'Gemma'
    elif 'qwen' in model_name.lower():
        model_type = 'Qwen'
    else:
        raise ValueError(f"Model type not recognised: {model_name}")

    resid_post_keys = filter_layers(
        model,
        model_type=model_type,
        # include_substrings=include_keywords,
        # exclude_substrings=exclude_keywords
    )
    activation_cache = ActivationCache()
    activation_cache.register_hooks(model, resid_post_keys)  # Register hooks on the model
    total_training_duration = 0  # in seconds
    if save_path is None and save_model:
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        save_dir = os.path.join(parent_dir, 'models', model_name.split('/')[-1].replace('-', '_'))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = set_model_name(model_name)

        if carma_weight == 1:
            # filename = f"{model_name}_{granularity}_carma_pure_{timestamp}.pt"
            m = str(mi_weight).replace('.', '_')
            s = str(stability_weight).replace('.', '_')
            c = str(carma_weight).replace('.', '_')
            filename = f"{model_name}_{granularity}_carma_pure_{c}_st_{s}_mi_{m}_seed_{seed}.pt"
        elif carma_weight > 0:
            # filename = f"{model_name}_{granularity}_carma_tuned_{timestamp}.pt"
            m = str(mi_weight).replace('.','_')
            s = str(stability_weight).replace('.','_')
            c = str(carma_weight).replace('.','_')
            filename = f"{model_name}_{granularity}_carma_tuned_{c}_st_{s}_mi_{m}_seed_{seed}.pt"
        else:
            filename = f"{model_name}_{granularity}_finetuned.pt"

        save_path = os.path.join(save_dir, task_name, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Model will be saved to: {save_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # Set up optimizer
    cross_entropy_loss = torch.nn.CrossEntropyLoss() # Task loss

    if carma_weight > 0:
        stability_loss = StabilityLoss(stability_start_layer, stability_end_layer, model_type)
        mi_loss = MILoss(mi_start_layer, mi_end_layer, model_type=model_type)

    total_training_steps = len(train_dataloader) * num_epochs
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)

    history = {
        'train_loss': [],
        'train_task_loss': [],
        'train_stability_loss': [],
        'train_mi_loss': [],
        'train_accuracy': [],
        'val_loss': [] if val_dataloader else None,
        'val_task_loss': [] if val_dataloader else None,
        'val_stability_loss': [] if val_dataloader else None,
        'val_mi_loss': [] if val_dataloader else None,
        'val_accuracy': [] if val_dataloader else None
    }

    best_val_loss = float('inf')
    best_accuracy = 0.0  # Track the best accuracy for model saving
    stability_loss_value = 0.0
    mi_loss_value = 0.0

    def get_last_token_logits(logits, attention_mask):
        """Helper function to get logits at last non-padding position"""
        last_non_pad = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        last_non_pad = last_non_pad.to(logits.device)
        return logits[batch_indices, last_non_pad]

    print('Starting training...')
    for epoch in range(num_epochs):
        metrics = defaultdict(float)
        num_batches = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for batch in progress_bar:
            model.train()
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            input_ids = batch['input_ids']
            labels = batch['labels']
            attention_mask = batch['attention_mask'] if 'attention_mask' in batch else None
            token_spans = batch['annotations']
            token_spans = [t['token_spans'] for t in token_spans]
            epoch_start_time = datetime.now()
            if isinstance(model, HookedTransformer):
                logits = model(
                    batch['input_ids'],
                    attention_mask=attention_mask,
                    return_type="logits",
                )
            else:
                logits = model(
                    batch['input_ids'],
                    attention_mask=attention_mask,
                )['logits']

            # Access the cache
            cache = activation_cache.cache
            # print('cache:', cache.keys())

            # Get the last token predictions
            # last_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # Find the last non-padding position for each sequence
            # Inspired by: https://colab.research.google.com/drive/1KgrEwvCKdX-8DQ1uSiIuxwIiwzJuQ3Gw?usp=sharing#scrollTo=t8chMxqel6mF
            labels = labels.to(logits.device)  # Ensure labels are on the same device as logits
            last_token_logits = get_last_token_logits(logits, attention_mask)

            # Debug prints for shapes (only first batch)
            if num_batches == 0:
                print(f"Input IDs shape: {batch['input_ids'].shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Logits shape: {logits.shape}")
                print(f"Last token logits shape: {last_token_logits.shape}")

            # Task-specific Cross-Entropy Loss
            if carma_weight != 1:
                ce_loss = cross_entropy_loss(last_token_logits, labels)
            if carma_weight == 0:
                total_loss = ce_loss
            else:
                # Stability loss
                stability_loss_value = stability_loss.calculate_loss(cache)
                # Mutual Information (MI) Regularisation Loss
                mi_loss_value = mi_loss.calculate_mi_loss(cache, token_spans)

                # Combine losses
                total_loss = ((1-carma_weight) * ce_loss
                          + carma_weight * (stability_weight * stability_loss_value + mi_weight * mi_loss_value))

            # Calculate accuracy
            pred_labels = last_token_logits.argmax(dim=-1)
            if task_name == 'sst':
                #pred_labels_text = model.tokenizer.batch_decode(pred_labels.tolist(), skip_special_tokens=True)
                pred_labels_text = tokenizer.batch_decode(pred_labels.tolist(), skip_special_tokens=True)
                #labels_text = model.tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)
                labels_text = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)
                pred_labels_text = [label.strip().lower() for label in pred_labels_text]
                accuracy = exact_match_accuracy(pred_labels_text, labels_text, task_type=task_name)
            else:
                accuracy = exact_match_accuracy(pred_labels, labels, task_type=task_name)

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            epoch_end_time = datetime.now()
            epoch_duration = epoch_end_time - epoch_start_time
            total_training_duration += epoch_duration.total_seconds()
            cache.clear()
            # Update history
            metrics['train_loss'] += total_loss.item()
            metrics['train_task_loss'] += ce_loss.item()
            if carma_weight > 0:
                metrics['train_stability_loss'] += stability_loss_value if isinstance(stability_loss_value, float) else stability_loss_value.item() #stability_loss_value.item()
                metrics['train_mi_loss'] += mi_loss_value if isinstance(mi_loss_value, float) else mi_loss_value.item()
            else:
                metrics['train_stability_loss'] += 0.0
                metrics['train_mi_loss'] += 0.0
            metrics['train_accuracy'] += accuracy if isinstance(accuracy, float) else accuracy.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({k: v / num_batches for k, v in metrics.items()})

        # Calculate average losses for epoch
        for k in metrics:
            metrics[k] /= num_batches
            history[k].append(metrics[k])

        # Validation
        if val_dataloader:
            model.eval()
            val_metrics = defaultdict(float)
            num_val_batches = 0
            cache.clear()
            with torch.no_grad():
                for batch in val_dataloader:
                    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
                    input_ids = batch['input_ids']
                    labels = batch['labels']
                    attention_mask = batch['attention_mask'] if 'attention_mask' in batch else None
                    token_spans = batch['annotations']
                    token_spans = [t['token_spans'] for t in token_spans]
                    if isinstance(model, HookedTransformer):
                        logits = model(
                            batch['input_ids'],
                            attention_mask=attention_mask,
                            return_type="logits"
                        )
                    else:
                        logits = model(
                            batch['input_ids'],
                            attention_mask=attention_mask,
                        )['logits']

                    cache = activation_cache.cache
                    # Get last token predictions
                    # last_token_logits = logits[:, -1, :]
                    labels = labels.to(logits.device)
                    last_token_logits = get_last_token_logits(logits, attention_mask)

                    valid_task_loss = cross_entropy_loss(last_token_logits, labels)
                    if carma_weight == 0:
                        valid_loss = valid_task_loss
                    else:
                        valid_stab_loss = stability_loss.calculate_loss(cache)
                        valid_mi_loss = mi_loss.calculate_mi_loss(cache, token_spans)
                        valid_loss = ((1-carma_weight) * valid_task_loss
                                  + carma_weight * (stability_weight*valid_stab_loss + mi_weight*valid_mi_loss))

                    # Calculate validation accuracy using exact_match_accuracy
                    pred_labels = last_token_logits.argmax(dim=-1)
                    if task_name == 'sst':
                        pred_labels_text = tokenizer.batch_decode(pred_labels.tolist(), skip_special_tokens=True)
                        labels_text = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)
                        #pred_labels_text = model.tokenizer.batch_decode(pred_labels.tolist(), skip_special_tokens=True)
                        #labels_text = model.tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)

                        pred_labels_text = [label.strip().lower() for label in pred_labels_text]
                        val_accuracy = exact_match_accuracy(pred_labels_text, labels_text, task_type='sst')
                    else:
                        val_accuracy = exact_match_accuracy(pred_labels, labels, task_type=task_name)

                    #val_accuracy = exact_match_accuracy(pred_labels, labels)

                    val_metrics['val_loss'] += valid_loss.item()
                    val_metrics['val_task_loss'] += valid_task_loss.item()
                    if carma_weight > 0:
                        val_metrics['val_stability_loss'] += valid_stab_loss if isinstance(valid_stab_loss, float) else valid_stab_loss.item()
                        val_metrics['val_mi_loss'] += valid_mi_loss if isinstance(valid_mi_loss, float) else valid_mi_loss.item()
                    else:
                        val_metrics['val_stability_loss'] += 0.0
                        val_metrics['val_mi_loss'] += 0.0
                    val_metrics['val_accuracy'] += val_accuracy if isinstance(val_accuracy, float) else val_accuracy.item()
                    # val_metrics['val_accuracy'] += val_accuracy.item()
                    num_val_batches += 1


                # Calculate validation averages
            for k in val_metrics:
                val_metrics[k] /= num_val_batches
                history[k].append(val_metrics[k])

            current_accuracy = val_metrics['val_accuracy']
            current_val_loss = val_metrics['val_loss']
            # Save the best model
            if val_metrics['val_loss'] < best_val_loss or current_accuracy > best_accuracy:
                best_val_loss = min(best_val_loss, val_metrics['val_loss'])
                best_accuracy = max(best_accuracy, current_accuracy)
                if save_model:
                    # Save the model state dict
                    training_config = {
                        "model_name": model_name,
                        "learning_rate": learning_rate,
                        "num_epochs": num_epochs,
                        "max_grad_norm": max_grad_norm,
                        "device": str(device),
                        "warmup_steps": warmup_steps,
                        "stability_weight": stability_weight,
                        "mi_weight": mi_weight,
                        "carma_weight": carma_weight,
                        "stability_start_layer": stability_start_layer,
                        "stability_end_layer": stability_end_layer,
                        "mi_start_layer": mi_start_layer,
                        "mi_end_layer": mi_end_layer,
                        "task_name": task_name,
                        "batch_size": train_dataloader.batch_size
                    }
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'best_accuracy': best_accuracy,
                        'training_history': history,
                        'training_config': training_config
                    }, save_path)

                # Print epoch summary
            print(f'\nEpoch {epoch + 1} Summary:')
            for k, v in metrics.items():
                print(f'  {k}: {v:.4f}')
            if val_dataloader:
                for k, v in val_metrics.items():
                    print(f'  {k}: {v:.4f}')

        else:
            #saving the model after each epoch
            training_config = {
                "model_name": model_name,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "max_grad_norm": max_grad_norm,
                "device": str(device),
                "warmup_steps": warmup_steps,
                "stability_weight": stability_weight,
                "mi_weight": mi_weight,
                "carma_weight": carma_weight,
                "stability_start_layer": stability_start_layer,
                "stability_end_layer": stability_end_layer,
                "mi_start_layer": mi_start_layer,
                "mi_end_layer": mi_end_layer,
                "task_name": task_name,
                "batch_size": train_dataloader.batch_size
            }
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_accuracy': best_accuracy,
                'training_history': history,
                'training_config': training_config
            }, save_path)

        print()
    model_name = save_path.split('/')[-1].split('.')[0]
    return history, model, model_name, total_training_duration
