import json
import os

import nltk
import torch
import numpy as np
import random
import transformers


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_model_name(model_name):
    model_name = model_name.replace('.pt', '')
    model_name = model_name.split('/')[-1]
    model_name = model_name.replace('.', '_')
    # model_name = model_name.split('.')[0]
    model_name = model_name.replace('-', '_')
    return model_name


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    transformers.set_seed(seed)
    #transformer_lens.HookedTransformerConfig.set_seed_everywhere(seed=seed) # transformer_lens.HookedTransformerConfig


def get_last_token_logits(logits, attention_mask):
    """Helper function to get logits at last non-padding position"""
    last_non_pad = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(logits.size(0), device=logits.device)
    return logits[batch_indices, last_non_pad]


def construct_save_path(parent_dir, model_name, task_name, granularity, carma_weight, timestamp):
    model_name = set_model_name(model_name, 'fine_tuned')
    save_dir = os.path.join(parent_dir, 'models', model_name.split('/')[-1].replace('-', '_'))
    if carma_weight > 0:
        filename = f"{model_name}_{granularity}_carma_tuned_{timestamp}.pt"
    else:
        filename = f"{model_name}_{granularity}_tuned_{timestamp}.pt"
    return os.path.join(save_dir, task_name, filename)


# Custom encoder for JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def ensure_nltk_resources():
    datasets = ['wordnet', 'punkt', 'averaged_perceptron_tagger', 'stopwords']
    for dataset in datasets:
        try:
            nltk.data.find(f'corpora/{dataset}')
            print(f"NLTK dataset '{dataset}' is already available.")
        except LookupError:
            print(f"Downloading NLTK dataset '{dataset}'...")
            nltk.download(dataset)
            print(f"Downloaded '{dataset}'.")