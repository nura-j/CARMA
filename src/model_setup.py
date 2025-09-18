# src/model_setup.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer


def load_model_transformers(model_name="gpt2", device='cpu'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype='auto')
    return model, tokenizer


def load_model(args, device, CARMA=False):
    """Load the model based on supervision type."""
    if args.supervision_type == 'fine-tuned':
        assert args.model_path is not None, "Model path must be provided for fine-tuned models"
        if CARMA:
            model = HookedTransformer.from_pretrained(
                args.model_name,
                device=device,
            )
            model.load_state_dict(torch.load(args.model_path)['model_state_dict'])
        else:
            # hf_model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype="auto")
            hf_model = AutoModelForCausalLM.from_pretrained(args.model_path)
            model = HookedTransformer.from_pretrained(
                args.model_name,
                hf_model=hf_model,
                center_unembed=args.model_name.startswith('GPT2'),
                center_writing_weights=args.model_name.startswith('GPT2'),
                fold_ln=args.model_name.startswith('GPT2'),
                refactor_factored_attn_matrices=args.model_name.startswith('GPT2'),
            )
    elif args.supervision_type == 'original':
        model = HookedTransformer.from_pretrained(
            args.model_name,
            center_unembed=args.model_name.startswith('GPT2'),
            center_writing_weights=args.model_name.startswith('GPT2'),
            fold_ln=args.model_name.startswith('GPT2'),
            refactor_factored_attn_matrices=args.model_name.startswith('GPT2'),
            # n_devices=torch.cuda.device_count()
        )
    elif args.supervision_type == 'instruct':
        raise NotImplementedError("Instruction-based supervision is not yet implemented")
    tokenizer = model.tokenizer
    return model, tokenizer


def load_model_tl(model_name="GPT2", device='cpu', model_path=None, path_type='tl'):
    if model_path and path_type == 'HF':
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cpu')
        state_dict = torch.load(model_path, map_location='cpu')['model_state_dict']
        hf_model.load_state_dict(state_dict)
        model = HookedTransformer.from_pretrained(
            model_name,
            hf_model=hf_model,
            device=device,
            # center_unembed=args.model_name.startswith('GPT2'),
            # center_writing_weights=args.model_name.startswith('GPT2'),
            # fold_ln=args.model_name.startswith('GPT2'),
            # refactor_factored_attn_matrices=args.model_name.startswith('GPT2'),
        )
    elif model_path and path_type == 'tl':
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
    else:
        model = HookedTransformer.from_pretrained(
            model_name,
            #device=device,
            # fold_ln=False,
            # center_writing_weights=False,
            # center_unembed=False,
            #n_devices=torch.cuda.device_count() if device == 'cuda' else 1
        )
    tokenizer = model.tokenizer
    # todo: padding size, padding token, add special tokens
    return model, tokenizer


class ActivationCache:
    """
    A utility to cache intermediate activations using forward hooks.
    """
    def __init__(self):
        self.cache = {}

    def hook_fn(self, module, input, output):
        """
        The hook function to store the module's output in the cache.
        """
        self.cache[module.fullname] = output

    def register_hooks(self, model, keys):
        """
        Register hooks on the specified layers/modules.

        Args:
            model (torch.nn.Module): The PyTorch model.
            keys (List[str]): List of layer/module names to hook.
        """
        for name, module in model.named_modules():
            if name in keys:
                module.fullname = name  # Add a reference name to the module
                module.register_forward_hook(self.hook_fn)


def filter_layers(model, model_type="GPT", block_prefix_gpt="transformer.h", block_prefix_gemma="model.layers"):
    """
    Filter layers to include only the outputs of transformer blocks, adapted for GPT and Gemma models.

    Args:
        model (torch.nn.Module): The PyTorch model.
        model_type (str): The type of model ("GPT" or "Gemma").
        block_prefix_gpt (str): The prefix for GPT transformer blocks (default: "transformer.h").
        block_prefix_gemma (str): The prefix for Gemma transformer blocks (default: "model.layers").

    Returns:
        List[str]: A list of layer names corresponding to block outputs.
    """
    filtered_layers = []

    if model_type == "GPT":
        # Include only the full block outputs for GPT models
        for name, _ in model.named_modules():
            if name.startswith(block_prefix_gpt) and not any(sub in name for sub in [".ln_", ".attn", ".mlp", ".dropout"]):
                filtered_layers.append(name)
    elif model_type == "Gemma" or model_type == "Llama" or model_type == 'Qwen':
        # Include only the full block outputs for Gemma models
        for name, _ in model.named_modules():
            #if name.startswith(block_prefix_gemma) and name.endswith("post_attention_layernorm"):
            if name.startswith(block_prefix_gemma) and name.count('.') == 2:
                filtered_layers.append(name)

    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types are 'GPT', 'LLama' and 'Gemma'.")

    return filtered_layers

def filter_layers_tl(model, include_substrings=None, exclude_substrings=None, layer_types=None):
    """
    Filter layers based on substrings, layer types, or exclusions.

    Args:
        model (torch.nn.Module): The PyTorch model.
        include_substrings (List[str]): Substrings to include in layer names.
        exclude_substrings (List[str]): Substrings to exclude from layer names.
        layer_types (Tuple[type]): Tuple of layer types to include (e.g., (nn.Linear, nn.Conv2d)).

    Returns:
        List[str]: A list of filtered layer names.
    """
    filtered_layers = []

    for name, module in model.named_modules():
        # Check substrings to include
        if include_substrings and not any(sub in name for sub in include_substrings):
            continue

        # Check substrings to exclude
        if exclude_substrings and any(sub in name for sub in exclude_substrings):
            continue

        # Check layer types
        if layer_types and not isinstance(module, layer_types):
            continue

        filtered_layers.append(name)

    return filtered_layers


def generate_hook_keys(model, base_keywords=None, layer_keywords=None):
    """
    Automatically generate keys for hooks by analyzing the model structure.

    Args:
        model (torch.nn.Module): The PyTorch model.
        base_keywords (List[str]): Keywords for base layers (e.g., 'embed', 'pos_embed').
        layer_keywords (List[str]): Keywords for per-layer components (e.g., 'resid', 'ln', 'attn').

    Returns:
        List[str]: A list of dynamically generated hook keys.
    """
    if base_keywords is None:
        base_keywords = ['hook_embed', 'hook_pos_embed']
    if layer_keywords is None:
        layer_keywords = ['hook_resid', 'hook_scale', 'hook_normalized',
                          'hook_attn', 'hook_mlp', 'hook_post']

    hook_keys = []
    for name, _ in model.named_modules():
        if any(kw in name for kw in base_keywords + layer_keywords):
            hook_keys.append(name)

    return hook_keys


def set_model_name(model_name, supervision_type=None):
    model_name = model_name.replace('.pt', '')
    model_name = model_name.split('/')[-1]
    model_name = model_name.replace('.', '_')
    model_name = model_name.replace('-', '_')
    if supervision_type:
        model_name += f"_{supervision_type}"
    return model_name