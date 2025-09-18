import json
from torch.utils.data import DataLoader, Dataset
import torch
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from copy import deepcopy
import random
from .synonym_generator import SynonymGenerator
synonym_generator = SynonymGenerator()

def clean_synonym(word):
    """Clean a synonym by removing underscores and checking for valid words."""
    # Remove underscores and split into words
    cleaned = word.replace('_', ' ')

    # Check if it's a single word (no spaces)
    if ' ' in cleaned:
        return None

    # Check if it contains only letters
    if not cleaned.isalpha():
        return None

    return cleaned


def get_word_pos(word, context):
    """Get the part of speech of a word in context."""
    # Tag the whole context to get better POS accuracy
    tagged = pos_tag(context.split())

    # Find the target word in the tagged context
    for w, pos in tagged:
        if w == word:
            return pos
    return None


def get_synonyms(word, context, num_synonyms=5):
    """
    Get more natural synonyms for a word considering its context.

    Args:
        word: The word to find synonyms for
        context: The full text context where the word appears
        num_synonyms: Maximum number of synonyms to return
    """
    if len(word) <= 3:  # Skip very short words
        return []

    # Skip common words that shouldn't be replaced
    common_words = {'the', 'and', 'that', 'this', 'but', 'they', 'have', 'with', 'from', 'will'} # todo: replace with stopwords
    if word.lower() in common_words:
        return []

    # Get POS tag for the word in context
    context_pos = get_word_pos(word, context)

    synonyms = set()
    for syn in wn.synsets(word):
        # Skip if POS doesn't match context (if we have context POS)
        if context_pos:
            if context_pos.startswith('JJ') and syn.pos() != 'a':  # adjective
                continue
            if context_pos.startswith('NN') and syn.pos() != 'n':  # noun
                continue
            if context_pos.startswith('VB') and syn.pos() != 'v':  # verb
                continue
            if context_pos.startswith('RB') and syn.pos() != 'r':  # adverb
                continue

        for lemma in syn.lemmas():
            if len(synonyms) >= num_synonyms:
                break

            synonym = lemma.name()
            if synonym == word:
                continue

            # Clean and validate the synonym
            cleaned = clean_synonym(synonym)
            if not cleaned:
                continue

            if cleaned == word:
                continue

            # Check word frequency to avoid rare words
            if lemma.count() < 3:  # Skip rare words
                continue

            synonyms.add(cleaned)

    return list(synonyms)


class SynonymReplacementDataset(Dataset):
    """Dataset for creating synonym replacements of the original data."""

    def __init__(self, original_dataset, tokenizer, model_name, dataset_name, prompts, replacement_percentage=0.9):
        """
        Initialize the synonym replacement dataset.

        Args:
            original_dataset: Original SavedPredictionsDataset instance
            tokenizer: Tokenizer for decoding/encoding text
            model_name: Name of the model (e.g., 'GPT2', 'Meta_Llama_3_70B')
            dataset_name: Name of the dataset (e.g., 'idm', 'sst')
            prompts: Dictionary of prompt templates by model and dataset
            replacement_percentage: Percentage of words to replace with synonyms
        """
        self.original_dataset = original_dataset
        self.tokenizer = tokenizer
        self.replacement_percentage = replacement_percentage

        # Extract the prefix and suffix from the prompt template
        # todo: automate this process
        if 'gemma' in model_name:
            model_name = 'gemma_2b'
        elif 'GPT2' in model_name:
            model_name = 'GPT2'
        elif 'GPT2_medium' in model_name:
            model_name = 'GPT2_medium'
        elif 'GPT2_large' in model_name:
            model_name = 'GPT2_large'
        elif 'Meta_Llama_3_8B' in model_name:
            model_name = 'Meta_Llama_3_8B'
        elif 'Meta_Llama_3_70B' in model_name:
            model_name = 'Meta_Llama_3_70B'
        elif 'Llama_3_2_1B' in model_name:
            model_name = 'Llama_3_2_1B'
        elif 'Llama_3_2_3B' in model_name:
            model_name = 'Llama_3_2_3B'
        elif 'Llama_3_2_70B' in model_name:
            model_name = 'Llama_3_2_70B'
        elif 'Llama_3_1_8B' in model_name:
            model_name = 'Llama_3_1_8B'
        elif 'Qwen2_0_5B' in model_name:
            model_name = 'Qwen2_0_5B'
        elif 'Qwen2_1_5B' in model_name:
            model_name = 'Qwen2_1_5B'
        elif 'Qwen2_7B' in model_name:
            model_name = 'Qwen2_7B'
        elif 'Qwen2_5_0_5B':
            model_name = 'Qwen2_5_0_5B'
        elif 'Qwen2_5_1_5B':
            model_name = 'Qwen2_5_1_5B'
        elif 'Qwen2_5_3B':
            model_name = 'Qwen2_5_3B'
        template = prompts[model_name][dataset_name]
        parts = template.split('{}')


        # Handle templates with or without placeholders
        if len(parts) == 2:
            self.prompt_prefix = parts[0]
            if 'gemma' in model_name:
                self.prompt_prefix = '<bos>' + self.prompt_prefix
            self.prompt_suffix = parts[1]
        else:
            self.prompt_prefix = ""
            self.prompt_suffix = template

    def extract_content_from_text(self, text, attention_mask):
        """
        Extract content while keeping track of padding token positions.
        Returns content and positions of padding tokens.
        """
        content = text

        # Remove prefix if it exists
        if self.prompt_prefix and text.startswith(self.prompt_prefix):
            #content = content[len(self.prompt_prefix):]
            content = content.split(self.prompt_prefix)[1]

        # Split on the suffix and get padding info
        if self.prompt_suffix:
            parts = content.split(self.prompt_suffix)
            content = parts[0]
            # Keep track of what comes after content (suffix + padding)
            remaining = self.prompt_suffix + ''.join(parts[1:])
        else:
            remaining = ""
        # print(f'content: {content}')
        # print(f'remaining: {remaining}')
        return content.strip(), remaining

    def replace_with_synonyms(self, text):
        """Replace words with synonyms in the content part only."""
        # print(f'Original Text: {text}')
        words = text.split()
        num_replacements = max(1, int(len(words) * self.replacement_percentage))

        # Create a list of indices for words that could be replaced
        valid_indices = [
            i for i, word in enumerate(words)
            if len(synonym_generator.get_synonyms(word, text)) > 0
            # if len(get_synonyms(word, text)) > 0  # Pass full text as context
        ]
        # print(f'Valid Indices: {valid_indices}')
        if not valid_indices:
            return text

        # Randomly select indices to replace
        indices_to_replace = random.sample(
            valid_indices,
            min(num_replacements, len(valid_indices))
        )
        # print(f'O Words: {(words)}, Replacements: {num_replacements}')
        # Replace selected words with synonyms
        for idx in indices_to_replace:
            synonyms = synonym_generator.get_synonyms(words[idx], text)
            # synonyms = get_synonyms(words[idx], text)
            if synonyms:
                words[idx] = random.choice(synonyms)
        # print(f'S Words: {(words)}, Replacements: {num_replacements}')
        return ' '.join(words)

    def get_attention_mask(self, text):
        """Generate attention mask by identifying padding tokens."""
        # Convert text to tokens to find padding tokens
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        padding_token_id = self.tokenizer.pad_token_id

        # Create attention mask (0 for padding tokens, 1 for others)
        attention_mask = [1 if token != padding_token_id else 0 for token in tokens]
        return torch.tensor(attention_mask)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_item = self.original_dataset[idx]
        item = deepcopy(original_item)

        # Decode input_ids without skipping special tokens
        full_text = self.tokenizer.decode(item['input_ids'], skip_special_tokens=False)

        # Extract content and remaining text (suffix + padding)
        content, remaining = self.extract_content_from_text(full_text, item['attention_mask'])

        # Replace words with synonyms only in the content
        modified_content = self.replace_with_synonyms(content)

        # Reconstruct the full text
        modified_text = (self.prompt_prefix + modified_content + remaining)

        # Encode while preserving padding token information
        encoded = self.tokenizer(
            modified_text,
            padding=False,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False  # Don't add additional special tokens
        )

        # Update the item
        item['input_ids'] = encoded['input_ids'][0]

        # Create proper attention mask (0 for padding tokens)
        item['attention_mask'] = self.get_attention_mask(modified_text)

        # Ensure lengths match
        if len(item['input_ids']) != len(item['attention_mask']):
            # Trim or pad attention mask to match input_ids length
            if len(item['attention_mask']) < len(item['input_ids']):
                pad_length = len(item['input_ids']) - len(item['attention_mask'])
                item['attention_mask'] = torch.cat([item['attention_mask'], torch.zeros(pad_length)])
            else:
                item['attention_mask'] = item['attention_mask'][:len(item['input_ids'])]
        # print(f'item["input_ids"]:', self.tokenizer.decode(item["input_ids"]))
        # print(f'item["attention_mask"]:', item["attention_mask"])
        return item


def create_synonym_replacement_test_loader_json(
        original_loader,
        tokenizer,
        model_name,
        dataset_name,
        prompts,
        replacement_percentage=0.2
):
    """
    Create a DataLoader for synonym replacement intervention.

    Args:
        original_loader: Original DataLoader instance
        tokenizer: Tokenizer for decoding/encoding text
        model_name: Name of the model (e.g., 'GPT2', 'Meta_Llama_3_70B')
        dataset_name: Name of the dataset (e.g., 'idm', 'sst')
        prompts: Dictionary of prompt templates by model and dataset
        replacement_percentage: Percentage of words to replace with synonyms

    Returns:
        DataLoader: A DataLoader for the modified dataset
    """
    # Create the synonym replacement dataset
    print(f'Creating synonym replacement dataset with {replacement_percentage * 100}% replacement')
    # print('original_loader.dataset:', original_loader.dataset)
    synonym_dataset = SynonymReplacementDataset(
        original_dataset=original_loader.dataset,
        tokenizer=tokenizer,
        model_name=model_name,
        dataset_name=dataset_name,
        prompts=prompts,
        replacement_percentage=replacement_percentage
    )

    # Create a new DataLoader using the same batch size and collate function
    return DataLoader(
        synonym_dataset,
        batch_size=original_loader.batch_size,
        shuffle=False,
        collate_fn=original_loader.collate_fn
    )
