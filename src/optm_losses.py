# src/optm_losses.py
from typing import Dict, List, Tuple

import random
from src.dataset import TokenSpan
import torch.nn.functional as F
import torch
import torch.nn as nn


# Cross-Entropy Loss for the task
def task_loss(logits, labels):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))


class MILoss(nn.Module):
    def __init__(self, start_layer: int = 0, end_layer: int = 11, temperature: float = 0.1, n_negatives: int = 5,
                 similarity_metric: str = "cosine", max_length: int = 58, model_type='GPT'):
        """
        InfoNCE-based Mutual Information Loss with cosine or dot product similarity
        Args:
            start_layer (int): Starting layer index for MI calculation.
            end_layer (int): Ending layer index for MI calculation.
            temperature (float): Temperature scaling factor for InfoNCE.
            n_negatives (int): Number of negative samples to use per pair.
            similarity_metric (str): Similarity metric to use. Options: "cosine", "dot_product".
            max_length (int): Maximum length in tokens.
        """
        super(MILoss, self).__init__()
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.temperature = temperature
        self.n_negatives = n_negatives
        self.similarity_metric = similarity_metric
        self.max_length = max_length
        self.index_combs = [torch.combinations(torch.arange(n_tokens), r=2) for n_tokens in range(max_length + 1)]
        self.model_type = model_type

    def compute_similarity(self, anchors, positives, negatives):
        """
        Compute similarity based on the chosen metric.
        Args:
            anchors: torch.Tensor of shape [N, D]
            positives: torch.Tensor of shape [N, D]
            negatives: torch.Tensor of shape [N, K, D] where K is n_negatives
        Returns:
            tuple of (pos_similarity [N], neg_similarity [N, K])
        """
        if self.similarity_metric == "dot_product":
            # Normalize vectors for numerical stability
            # anchors = F.normalize(anchors, dim=-1) # v / ||v||
            # positives = F.normalize(positives, dim=-1)
            # negatives = F.normalize(negatives, dim=-1)

            pos_similarity = torch.sum(anchors * positives, dim=-1)  # dot product = cos(theta) = a.b / ||a|| ||b||
            neg_similarity = torch.bmm(anchors.unsqueeze(1), negatives.transpose(-2, -1)).squeeze(
                1)  # anchor is a vector, negatives are vectors

        elif self.similarity_metric == "cosine":  # most used in the literature
            pos_similarity = F.cosine_similarity(anchors, positives, dim=-1)
            neg_similarity = F.cosine_similarity(anchors.unsqueeze(1).expand(-1, negatives.size(1), -1), negatives,
                                                 dim=-1)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        return pos_similarity / self.temperature, neg_similarity / self.temperature

    @staticmethod
    def get_anchors(activations: torch.Tensor, indices: List[Tuple[torch.Tensor, Tuple[int, int]]]) -> torch.Tensor:
        anchors = torch.stack([
            activations[batch_idx][idxs[1][0]:idxs[1][1]][idx_pair[0]]
            for batch_idx in range(activations.shape[0]) for idxs in indices for idx_pair in idxs[0]
        ])

        return anchors

    @staticmethod
    def get_positives(activations: torch.Tensor, indices: List[Tuple[torch.Tensor, Tuple[int, int]]]) -> torch.Tensor:
        positives = torch.stack([
            activations[batch_idx][idxs[1][0]:idxs[1][1]][idx_pair[1]]
            for batch_idx in range(activations.shape[0]) for idxs in indices for idx_pair in idxs[0]
        ])

        return positives

    @staticmethod
    def get_negatives(activations: torch.Tensor,
                      indices: List[Tuple[torch.Tensor, Tuple[int, int]]],
                      n_negatives: int) -> torch.Tensor:
        batch_size = activations.shape[0]
        neg_sample_size = sum([len(idxs[0]) for batch_idx in range(batch_size) for idxs in indices])
        neg_indices = list()

        for batch_idx in range(batch_size):
            for idxs in indices:
                for idx_pair in idxs[0]:
                    neg_indices.extend(
                        sorted(random.choices(
                            [(batch_idx, i)
                             for i in range(activations[batch_idx].shape[0])
                             if (i < idxs[1][0] or i >= idxs[1][1])
                             ],
                            k=n_negatives
                        ), key=lambda x: x[1])
                    )

        negatives_alt = activations[torch.tensor(neg_indices).long().T.tolist()]

        return negatives_alt.view((neg_sample_size, n_negatives, activations.shape[-1]))

    def compute_layer_mi(self, activations: torch.Tensor, token_spans: List[List['TokenSpan']]) -> torch.Tensor:
        """
        Compute mutual information for a single layer
        Args:
            activations: torch.Tensor of shape [batch_size, seq_len, hidden_dim]
            token_spans: List of lists of TokenSpan objects
        Returns:
            torch.Tensor: MI estimate for the layer
        """

        indices = [
            (self.index_combs[ws.end_idx - ws.start_idx], (ws.start_idx, ws.end_idx))
            for word_spans in token_spans for ws in word_spans[:-1]
            if (ws.end_idx - ws.start_idx > 1)
        ]

        anchors = MILoss.get_anchors(activations, indices)
        positives = MILoss.get_positives(activations, indices)
        negatives = MILoss.get_negatives(activations, indices, self.n_negatives)

        # Compute InfoNCE loss
        # pos_similarity, neg_similarity = self.compute_similarity(anchors, positives, negatives)
        pos_similarity, neg_similarity = self.compute_similarity(anchors, positives, negatives)

        # InfoNCE loss: -log(exp(pos_sim) / (exp(pos_sim) + sum(exp(neg_sim))))
        numer = torch.exp(pos_similarity)
        denom = numer + torch.sum(torch.exp(neg_similarity), dim=1)
        mi_estimate = torch.mean(torch.log(numer / denom))  # average over batches

        return mi_estimate

    def calculate_mi_loss(self, cache, token_spans: List[List['TokenSpan']]) -> torch.Tensor:
        """
        Forward pass to calculate MI loss across layers
        """
        # resid_keys = sorted([k for k in cache.keys() if 'resid_post' in k])
        if self.model_type == 'GPT': # Since we are using hugface transformers, we need to manually check the model type and get the correct keys
            resid_keys = sorted([k for k in cache.keys() if k.startswith("transformer.h.")],
                                key=lambda x: int(x.split('.')[2]))
        elif self.model_type == 'Gemma' or self.model_type == 'Llama':
            resid_keys = sorted([k for k in cache.keys() if k.startswith("model.layers.")],
                                key=lambda x: int(x.split('.')[2]))
        else: # for other models and future models can be added here
            resid_keys = sorted([k for k in cache.keys() if k.startswith("model.layers.")],
                                key=lambda x: int(x.split('.')[2]))
        resid_keys = resid_keys[self.start_layer:self.end_layer + 1]

        total_mi = 0.0
        n_valid_layers = 0

        for key in resid_keys:
            layer_activations = cache[key]
            if isinstance(layer_activations, tuple):
                layer_activations = layer_activations[0]
            layer_mi = self.compute_layer_mi(layer_activations, token_spans)
            # print('layer_mi:', layer_mi)
            if layer_mi is not None:
                # total_mi += layer_mi.item()
                total_mi += layer_mi
                n_valid_layers += 1

        if n_valid_layers == 0:
            return torch.tensor(0.0, device=cache[resid_keys[0]].device)

        return -total_mi / n_valid_layers  # Negative because we want to maximize MI


class StabilityLoss:
    """
    Stability loss for the hidden states of the model
    It runs as a forward hook on the model
    loss = sum(layer_stability_loss(activations_l, activations_l_plus_1) for l in layers)
    layer_stability_loss = (activations_l - activations_l_plus_1)**2
    """

    def __init__(self, start_layer=0, end_layer=11, model_type='GPT'):
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.model_type = model_type
        # must check the start and end layer are in correct ranges

    def calculate_loss(self, cache) -> torch.Tensor:
        """
        Calculate the layer-wise stability loss using cached activations.

        Args:
            cache: TransformerLens activation cache

        Returns:
            torch.Tensor: Stability loss
        """
        # loss = torch.tensor(0.0)
        loss = 0.0
        # resid_keys = sorted([k for k in cache.keys() if 'resid_post' in k])
        # resid_keys = sorted([k for k in cache.keys() if k.startswith("transformer.h.")],
        #                     key=lambda x: int(x.split('.')[2]))
        if self.model_type == 'GPT':
            resid_keys = sorted([k for k in cache.keys() if k.startswith("transformer.h.")],
                                key=lambda x: int(x.split('.')[2]))
        elif self.model_type == 'Gemma' or self.model_type == 'Llama':
            resid_keys = sorted([k for k in cache.keys() if k.startswith("model.layers.")],
                                key=lambda x: int(x.split('.')[2]))
        else:
            resid_keys = sorted([k for k in cache.keys() if k.startswith("model.layers.")],
                                key=lambda x: int(x.split('.')[2]))

        if self.end_layer is None:
            self.end_layer = len(resid_keys) - 1

        for i in range(self.start_layer, self.end_layer):
            current = cache[resid_keys[i]]
            if isinstance(current, tuple):
                current = current[0]
            # print('current shape:', current.requires_grad)
            next_layer = cache[resid_keys[i + 1]]
            if isinstance(next_layer, tuple):
                next_layer = next_layer[0]
            # print('next_layer shape:', next_layer.requires_grad)
            # layer_loss = torch.mean((next_layer - current) ** 2) # should I convert this by softmax?
            next_layer = next_layer.to(current.device)
            layer_loss = torch.mean((next_layer - current) ** 2) / (
                    torch.mean(current ** 2) + torch.mean(next_layer ** 2) + 1e-8)
            #layer_loss.to(loss.device)
            # loss += layer_loss.item()
            loss += layer_loss
        return loss
