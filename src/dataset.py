from collections import Counter, defaultdict
from datasets import load_dataset, DatasetDict, concatenate_datasets
from typing import Dict, List, Optional, Tuple, Any
import spacy
from dataclasses import dataclass
import random
import torch
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from torch.utils.data import DataLoader, Dataset as TorchDataset
import re
stops = set(stopwords.words('english'))


@dataclass
class TokenSpan:
    """Store token span information"""
    start_idx: int
    end_idx: int
    token_ids: List[int]
    token_texts: List[str]
    original_word: str
    # reduction_percentage: Optional[float] = None
    # linguistic_features: Dict[str, Any]


additional_prompts = {
    'GPT2': {
        'idm': '{} is called a "', # 87
        'hypernyms': '{} is a type of "',
        'synonyms': '{} is a synonym of "',
        'sst': 'The sentiment of {} is \"',
    },
    'GPT2_medium': {
        'idm': '{} is called a "',
        'hypernyms': '{} is a type of "',
        'synonyms': '{} is a synonym of "',
        'sst': 'The sentiment of {} is \"',
    },
    'GPT2_large': {
        'idm': '{} is called a "',
        'hypernyms': '{} is a type of "',
        'synonyms': '{} is a synonym of "',
        'sst': 'The sentiment of {} is \"',
    },
    'gemma_2b': {
        'idm': '{} is called a "', 
        'hypernyms': '{} is a type of "', 
        'synonyms': '{} is a synonym of "', 
        'sst': "Determine if the sentiment of the headline in brackets is positive, neutral, or negative. Return the label: [{}] ="
    },
    'Meta_Llama_3_8B': {
        'idm': '{} is called a "', 
        'hypernyms': '{} is a type of "', 
        'synonyms': '{} is a synonym of "', 
        # 'sst': """
        #     Analyze the sentiment of the news headline enclosed in square brackets,
        #     determine if it is positive, neutral, or negative, and return the answer as
        #     the corresponding sentiment label positive or neutral or negative
        #
        #     [{}] =
        #
        #     """.strip()
        'sst': "Label the sentiment of the headline in brackets: positive, neutral, or negative. [{}] ="
    },
    'mistral_7b': {
        'idm': '{} is called a ', 
        'hypernyms': '{} is a type of', 
        'synonyms': '{} is a synonym of', 
    },
    'Llama-2-7b-chat-hf': {
        'idm': '{} is called a ', 
        'hypernyms': '{} is a type of', 
        'synonyms': '{} is a synonym of', 
        'sst': "Label the sentiment of the headline in brackets: positive, neutral, or negative. [{}] ="
    },
    'Meta_Llama_3_70B': {
            'idm': '{} is called a "', 
            'hypernyms': '{} is a type of "', 
            'synonyms': '{} is a synonym of "', 
            # 'sst': """
            #             Analyze the sentiment of the news headline enclosed in square brackets,
            #             determine if it is positive, neutral, or negative, and return the answer as
            #             the corresponding sentiment label positive or neutral or negative
            #
            #             [{}] =
            #
            # """.strip(),
        'sst': "Label the sentiment of the headline in brackets: positive, neutral, or negative. [{}] ="
    },
    'Llama_3_2_3B': {
        'idm': '{} is called a "',  
        'hypernyms': '{} is a type of "',  
        'synonyms': '{} is a synonym of "',  
        # 'sst': """
        #                 Analyze the sentiment of the news headline enclosed in square brackets,
        #                 determine if it is positive, neutral, or negative, and return the answer as
        #                 the corresponding sentiment label positive or neutral or negative
        #
        #                 [{}] = \"
        #
        #     """.strip(),
        'sst': "Label the sentiment of the headline in brackets: positive, neutral, or negative. [{}] ="
    },
    'Llama_3_2_1B': {
        'idm': '{} is called a "',  
        'hypernyms': '{} is a type of "',  
        'synonyms': '{} is a synonym of "',  
        # 'sst': """
        #                 Analyze the sentiment of the news headline enclosed in square brackets,
        #                 determine if it is positive, neutral, or negative, and return the answer as
        #                 the corresponding sentiment label positive or neutral or negative
        #
        #                 [{}] = \"
        #
        #     """.strip(),
        'sst': "Label the sentiment of the headline in brackets: positive, neutral, or negative. [{}] ="
    },
    'Llama_3_1_8B': {
        'idm': '{} is called a "',  
        'hypernyms': '{} is a type of "',  
        'synonyms': '{} is a synonym of "',  
        # 'sst': """
        #                 Analyze the sentiment of the news headline enclosed in square brackets,
        #                 determine if it is positive, neutral, or negative, and return the answer as
        #                 the corresponding sentiment label positive or neutral or negative
        #
        #                 [{}] = \"
        #
        #     """.strip()
        'sst': "Label the sentiment of the headline in brackets: positive, neutral, or negative. [{}] =",
    },
    'Llama_2_7b_hf': {
        'idm': '{} is called a "', 
        'hypernyms': '{} is a type of "', 
        'synonyms': '{} is a synonym of "', 
        'sst': "Label the sentiment of the headline in brackets: positive, neutral, or negative. [{}] ="
    },
    'Llama_2_7b': {
        'idm': '{} is called a "', 
        'hypernyms': '{} is a type of "', 
        'synonyms': '{} is a synonym of "', 
        'sst': "Label the sentiment of the headline in brackets: positive, neutral, or negative. [{}] ="
    },
    'Qwen2_0_5B': {
        'idm': '{} is called a "',
        'hypernyms': '{} is a type of "', 
        'synonyms': '{} is a synonym of "', 
        'sst': 'The sentiment of {} is \"',
    },
    'Qwen2_1_5B': {
        'idm': '{} is called a "', 
        'hypernyms': '{} is a type of "', 
        'synonyms': '{} is a synonym of "', 
        'sst': 'The sentiment of {} is \"',
    },
    'Qwen2_7B': {
        'idm': '{} is called a "', 
        'hypernyms': '{} is a type of "', 
        'synonyms': '{} is a synonym of "', 
        'sst': 'The sentiment of {} is \"',
    },
    'Qwen2_5_0_5B':{
        'idm': '{} is commonly known as "',
        'hypernyms': '{} is a type of "',
        'synonyms': '{} is a synonym of "',
        'sst': 'The sentiment of {} is \"',
    },
    'Qwen2_5_1_5B':{
        'idm': '{} is commonly known as "',
        'hypernyms': '{} is a type of "',
        'synonyms': '{} is a synonym of "',
        'sst': 'The sentiment of {} is \"',
    },
    'Qwen2_5_3B':{
        'idm': '{} is commonly known as "',
        'hypernyms': '{} is a type of "',
        'synonyms': '{} is a synonym of "',
        'sst': 'The sentiment of {} is \"',
    },
}


def clean_text(text: str) -> str:
    """Clean SST text with various preprocessing steps."""
    # Convert to lowercase
    text = text.lower()

    # Fix spacing around punctuation
    text = re.sub(r'\s+([?.!,"])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'(["])\s+', r'\1', text)  # Remove space after quotes
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

    # Clean specific patterns
    text = re.sub(r'\.\.+', '.', text)  # Replace multiple dots with single dot
    text = re.sub(r'\s*\.\s*$', '', text)  # Remove trailing dots and spaces
    text = re.sub(r'\s*\!\s*$', '!', text)  # Fix exclamation mark spacing
    text = re.sub(r'\s*\?\s*$', '?', text)  # Fix question mark spacing

    # Remove specific artifacts
    text = text.replace('`', "'")  # Replace backticks with apostrophes
    text = text.replace('\'', "'")               # Normalize quotes
    text = text.replace('--', '-')               # Normalize dashes

    # Fix contractions spacing
    text = re.sub(r'\s\'s', "'s", text)         # Fix 's contractions
    text = re.sub(r'\s\'t', "'t", text)         # Fix 't contractions
    text = re.sub(r'\s\'re', "'re", text)       # Fix 're contractions
    text = re.sub(r'\s\'ve', "'ve", text)       # Fix 've contractions
    text = re.sub(r'\s\'ll', "'ll", text)       # Fix 'll contractions

    # Remove extra spaces and strip
    text = ' '.join(text.split())
    text = text.strip()

    return text

class CustomDataset(TorchDataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DatasetLoader:
    """Dataset loader with preprocessing and annotation capabilities."""

    VALID_DATASETS = {"idm", "mp", "sst"}

    def __init__(self,
                 dataset_name: str,
                 path: str,
                 tokenizer=None,
                 max_length: Optional[int] = None,
                 batch_size: int = 16,
                 spacy_model: str = "en_core_web_sm",
                 model_name: str = "GPT2",
                 augment_test: bool = False):
        """
        Initialize the dataset loader.
        Args:
            dataset_name: Name of the dataset to load
            path: Path to the dataset
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            batch_size: Batch size for DataLoader
            spacy_model: Name of spaCy model to use for linguistic analysis
        """
        if dataset_name not in self.VALID_DATASETS:
            raise ValueError(f"Invalid dataset name. Must be one of {self.VALID_DATASETS}")

        self.dataset_name = dataset_name
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.dataset = None
        self.encodings = {}
        self.nlp = spacy.load(spacy_model)
        self.model_name = model_name
        self.augment_test = augment_test

        if tokenizer is None:
            raise ValueError("Tokenizer must be provided")

        self.load_data()
        print(f"Dataset '{dataset_name}' loaded successfully.")

        if self.max_length is None:
            self.max_length = self._compute_max_length()

    def load_data(self) -> None:
        """Load the specified dataset."""
        loading_methods = {
            "idm": self._load_idm,
            "morphological_production": self._load_morphological_production,
            "sst": self._load_sst # Stanford Sentiment Treebank
        }

        if self.dataset_name not in loading_methods:
            raise ValueError(f"Dataset '{self.dataset_name}' not supported")

        self.dataset = loading_methods[self.dataset_name](self.path)

    def _compute_max_length(self, splits=('train', 'valid', 'test'), ratio=0.95) -> int:
        """
        Compute the maximum sequence length for tokenized prompts based on a threshold ratio.

        Args:
            ratio (float): Percentage (0-1 or 0-100) of samples to include in the length computation.

        Returns:
            int: Computed maximum length for tokenization.
        """
        if ratio > 1:
            ratio /= 100.  # Normalize to [0, 1] if given as a percentage

        if not (0 <= ratio <= 1):
            raise ValueError("Ratio must be between 0 and 1 (inclusive).")

        lengths = Counter()
        max_len = 0

        for split in splits:
            if split not in self.dataset:
                raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(self.dataset.keys())}")

            for item in self.dataset[split]:
                prompt = self.format_prompt(item)
                tokenized_prompt = self.tokenizer(prompt, return_tensors="pt").input_ids
                length = tokenized_prompt.shape[1]
                lengths[length] += 1
                max_len = max(max_len, length)

        if ratio == 1:  # Return the absolute maximum length
            return max_len
        # print(f"Lenghts: {sorted(lengths.items())}")
        # Compute the cumulative distribution of lengths
        total = sum(lengths.values())
        current = 0
        for length, count in sorted(lengths.items()):  # Process lengths in ascending order - could be improved tp avoid starting from shortest
            current += count
            if current / total >= ratio:
                return length

        # Fallback: If no lengths meet the criteria (shouldn't happen)
        print(f'Computed max length: {max_len}')
        return max_len

    def _load_idm(self, path: str) -> DatasetDict:
        """Load IDM dataset with proper train/valid/test splits."""
        try:
            dataset = load_dataset("json", data_files=path)
            if 'train' not in dataset:
                raise ValueError("Dataset must contain 'train' split")

            train_test = dataset['train'].train_test_split(test_size=0.2, seed=42)
            test_valid = train_test['test'].train_test_split(test_size=0.5, seed=42)

            if self.augment_test:
                augment_path = './data/webster_augmented_test_seed_42.json'
                augment_dataset = load_dataset("json", data_files=augment_path)['train']
                test_valid['test'] = concatenate_datasets([test_valid['test'], augment_dataset])

            return DatasetDict({
                'train': train_test['train'],
                'valid': test_valid['train'],
                'test': test_valid['test']
            })
        except Exception as e:
            raise RuntimeError(f"Error loading IDM dataset: {str(e)}")

    def _load_morphological_production(self, path: str) -> DatasetDict:
        """Load morphological production dataset."""
        raise NotImplementedError("Morphological production dataset loading not implemented")

    def _load_sst(self, path: str="stanfordnlp/sst") -> DatasetDict:
        """Load IDM dataset with proper train/valid/test splits."""
        try:
            dataset = load_dataset(path, trust_remote_code=True)
            if 'train' not in dataset:
                raise ValueError("Dataset must contain 'train' split")
            result_dataset = DatasetDict({
                'train': dataset['train'],
                'valid': dataset['validation'],
                'test': dataset['test']
            })
            if self.augment_test:
                try:
                    # Determine augmentation dataset
                    augment_path = 'stanfordnlp/imdb'
                    print(f"Loading augmentation dataset from {augment_path}")

                    # Load and process augmentation dataset
                    augment_dataset = load_dataset(augment_path)['test']

                    # Rename text column to sentence and convert labels
                    augment_dataset = augment_dataset.rename_column('text', 'sentence')

                    # Convert the IMDB dataset to match SST features
                    from datasets import Features, Value

                    def convert_features(batch):
                        # Keep the original label values but change the type to float32
                        return {
                            'sentence': batch['sentence'],
                            'label': [float(l) for l in batch['label']]
                        }

                    # Cast to new feature schema
                    augment_dataset = augment_dataset.map(
                        convert_features,
                        batched=True,
                        features=Features({
                            'sentence': Value('string'),
                            'label': Value('float32')
                        })
                    )
                    # Verify required columns exist
                    required_columns = {'sentence', 'label'}
                    missing_columns = required_columns - set(augment_dataset.column_names)
                    if missing_columns:
                        raise ValueError(f"Augmentation dataset missing required columns: {missing_columns}")
                    print(augment_dataset[0])
                    print(result_dataset['test'][0])
                    # Combine datasets
                    result_dataset['test'] = concatenate_datasets([result_dataset['test'],augment_dataset])
                    print(f"Successfully augmented test set. New size: {len(result_dataset['test'])}")

                except Exception as e:
                    print(f"Warning: Failed to augment test set: {str(e)}")
                    print("Continuing with original test set")

            return result_dataset
        except Exception as e:
            raise RuntimeError(f"Error loading SST dataset: {str(e)}")

    def format_prompt(self, prompt: str) -> str:
        """Format the prompt according to task requirements."""
        pmpt = additional_prompts[self.model_name][self.dataset_name]
        # print(pmpt.format(prompt))
        return pmpt.format(prompt)

    # def get_token_spans(self, text: str, tokens: List[str], token_ids: List[int], constituency_level: str) -> List[
    #     TokenSpan]:
    #     """
    #     Get spans of tokens that belong to the same word.
    #     """
    #     if constituency_level == 'tok_word':
    #         current_span_start = 0
    #         token_idx = 0
    #         splitted_text = text.split()
    #         current_word_idx = 0
    #         spans = []
    #
    #         # Handle empty or invalid input
    #         if not tokens or not text:
    #             return spans
    #
    #         while token_idx < len(tokens):
    #             try:
    #                 # Check for word boundaries using tokenizer-specific prefixes
    #                 if token_idx < len(tokens) and  tokens[token_idx][0] in ['Ġ', '▁', '##']:
    #                     if current_span_start < token_idx and current_word_idx < len(splitted_text):
    #                         spans.append(TokenSpan(
    #                             start_idx=current_span_start,
    #                             end_idx=token_idx,
    #                             token_ids=token_ids[current_span_start:token_idx],
    #                             token_texts=tokens[current_span_start:token_idx],
    #                             original_word=splitted_text[current_word_idx],
    #                         ))
    #                         current_word_idx += 1
    #                     current_span_start = token_idx
    #                     token_idx += 1
    #
    #                 elif tokens[token_idx] == self.tokenizer.pad_token:
    #                     # Save the last word before padding
    #                     if current_span_start < token_idx and current_word_idx < len(splitted_text):
    #                         spans.append(TokenSpan(
    #                             start_idx=current_span_start,
    #                             end_idx=token_idx,
    #                             token_ids=token_ids[current_span_start:token_idx],
    #                             token_texts=tokens[current_span_start:token_idx],
    #                             original_word=splitted_text[current_word_idx],
    #
    #                         ))
    #                         current_word_idx += 1
    #                     spans.append(TokenSpan(
    #                         start_idx=current_span_start+1,
    #                         end_idx=len(tokens),
    #                         token_ids=token_ids[current_span_start:],
    #                         token_texts=tokens[current_span_start:],
    #                         original_word=self.tokenizer.pad_token
    #                     ))
    #
    #                     # # Add all padding tokens as a single span
    #                     # pad_start = token_idx
    #                     # while token_idx < len(tokens) and tokens[token_idx] == self.tokenizer.pad_token:
    #                     #     token_idx += 1
    #                     # spans.append(TokenSpan(
    #                     #     start_idx=pad_start,
    #                     #     end_idx=token_idx,
    #                     #     token_ids=token_ids[pad_start:token_idx],
    #                     #     token_texts=tokens[pad_start:token_idx],
    #                     #     original_word=self.tokenizer.pad_token
    #                     # ))
    #                     break
    #
    #                 else:
    #                     token_idx += 1
    #
    #             except IndexError:
    #                 print(f"Warning: Index error processing tokens at position {token_idx}")
    #                 break
    #
    #         # Add final span if there's remaining tokens
    #         if token_idx > current_span_start and current_word_idx < len(splitted_text):
    #             spans.append(TokenSpan(
    #                 start_idx=current_span_start,
    #                 end_idx=token_idx,
    #                 token_ids=token_ids[current_span_start:token_idx],
    #                 token_texts=tokens[current_span_start:token_idx],
    #                 original_word=splitted_text[current_word_idx]
    #             ))
    #         # reduction percentage
    #         # for span in spans:
    #         return spans
    #
    #     raise ValueError(f"Unsupported constituency level: {constituency_level}")
    def get_token_spans(self, text: str, tokens: List[str], token_ids: List[int], constituency_level: str) -> List[
        TokenSpan]:
        """
        Get spans of tokens that belong to the same word.
        """
        if constituency_level == 'tok_word':
            current_span_start = 0
            token_idx = 0
            splitted_text = text.split()
            current_word_idx = 0
            spans = []

            # Handle empty or invalid input
            if not tokens or not text:
                return spans

            # Loop through tokens to create spans
            while token_idx < len(tokens):
                try:
                    # Check for word boundaries using tokenizer-specific prefixes
                    if token_idx < len(tokens) and tokens[token_idx][0] in ['Ġ', '▁', '##']:
                        if current_span_start < token_idx and current_word_idx < len(splitted_text):
                            spans.append(TokenSpan(
                                start_idx=current_span_start,
                                end_idx=token_idx,
                                token_ids=token_ids[current_span_start:token_idx],
                                token_texts=tokens[current_span_start:token_idx],
                                original_word=splitted_text[current_word_idx]
                            ))
                            current_word_idx += 1
                        current_span_start = token_idx
                        token_idx += 1

                    elif tokens[token_idx] == self.tokenizer.pad_token:
                        # Handle padding tokens
                        if current_span_start < token_idx and current_word_idx < len(splitted_text):
                            spans.append(TokenSpan(
                                start_idx=current_span_start,
                                end_idx=token_idx,
                                token_ids=token_ids[current_span_start:token_idx],
                                token_texts=tokens[current_span_start:token_idx],
                                original_word=splitted_text[current_word_idx]
                            ))
                            current_word_idx += 1
                        break
                    else:
                        token_idx += 1

                except IndexError:
                    print(f"Warning: Index error processing tokens at position {token_idx}")
                    break

            # Add final span if there's remaining tokens
            if token_idx > current_span_start and current_word_idx < len(splitted_text):
                spans.append(TokenSpan(
                    start_idx=current_span_start,
                    end_idx=token_idx,
                    token_ids=token_ids[current_span_start:token_idx],
                    token_texts=tokens[current_span_start:token_idx],
                    original_word=splitted_text[current_word_idx]
                ))

            return spans

        raise ValueError(f"Unsupported constituency level: {constituency_level}")

    def annotate_input(self, text: str, token_ids: List[int]) -> Dict[str, Any]:
        """
        Annotate tokens with their relationships to original words and calculate reduction percentage.
        """
        # Get tokens from IDs
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        # Exclude prompt and padding tokens
        valid_tokens = [t for t in tokens if t not in ['<pad>', '<|endoftext|>', '<eos>']]
        original_length = len(valid_tokens)

        # Get token spans
        token_spans = self.get_token_spans(text=text, tokens=tokens, token_ids=token_ids, constituency_level='tok_word')

        # Calculate reduced length (number of spans)
        reduced_length = len(token_spans)

        # Calculate reduction percentage
        reduction_percentage = ((original_length - reduced_length) / original_length) * 100 if original_length > 0 else 0

        # Add reduction percentage to each span for tracking
        for span in token_spans:
            span.reduction_percentage = reduction_percentage

        return {
            'token_spans': token_spans,
            'reduction_percentage': reduction_percentage
        }

    def tokenize_data(self, data) -> List[Dict]:
        """Tokenize and annotate the dataset examples."""
        tokenized_data = []

        for example in data:
            # print('example:', example)
            try:
                # print()
                # Format and clean the input
                if self.dataset_name == 'idm':
                    formatted_definition = self.format_prompt(example['definition'])
                    cleaned_word = example['word'].replace("_", "")
                elif self.dataset_name == 'sst':
                    example['sentence'] = example['sentence'].lower().replace(" .", "").strip() # todo: more cleaning
                    example['sentence'] = clean_text(example['sentence'])

                    formatted_definition = self.format_prompt(example['sentence'])
                    # cleaned_word = 'positive' if example['label'] == 1 else 'negative' # double check 1 and 0
                    cleaned_word = (
                        'positive' if example['label'] > 0.6 else
                        'neutral' if 0.4 <= example['label'] <= 0.6 else
                        'negative'
                    )

                else:
                    raise ValueError("Unsupported dataset")

                # Tokenize input
                input_encoding = self.tokenizer(
                    formatted_definition,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                # Tokenize label
                label_encoding = self.tokenizer(
                    cleaned_word,
                    padding=False,
                    add_special_tokens=False,
                    return_tensors='pt'
                )

                if label_encoding.input_ids.size(1) == 1:
                    # Get annotations with spans
                    try:
                        annotations = self.annotate_input(
                            formatted_definition,
                            input_encoding.input_ids.squeeze().tolist()
                        )

                        tokenized_data.append({
                            'input_ids': input_encoding.input_ids.squeeze(),
                            'attention_mask': input_encoding.attention_mask.squeeze(),
                            'labels': label_encoding.input_ids.squeeze(),
                            'annotations': annotations
                        })
                    except Exception as e:
                        print(f"Warning: Failed to create annotations: {str(e)}")
                        continue

            except Exception as e:
                print(f"Warning: Failed to tokenize example: {str(e)}")
                continue

        if not tokenized_data:
            raise ValueError("No valid examples after tokenization")

        return tokenized_data

    def create_data_loader(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoader instances for train, validation, and test sets."""
        try:
            tokenized_train = self.tokenize_data(self.dataset['train'])
            tokenized_valid = self.tokenize_data(self.dataset['valid'])
            tokenized_test = self.tokenize_data(self.dataset['test'])
            print('Tokenized data')

            # Custom collate function to handle variable-sized annotations
            def collate_fn(batch):
                batch_dict = {
                    'input_ids': torch.stack([item['input_ids'] for item in batch]),
                    'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
                    'labels': torch.stack([item['labels'] for item in batch]),
                    'annotations': [item['annotations'] for item in batch]
                }
                return batch_dict

            train_dataset = CustomDataset(tokenized_train)
            valid_dataset = CustomDataset(tokenized_valid)
            test_dataset = CustomDataset(tokenized_test)
            print('Created datasets')

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )

            return train_loader, valid_loader, test_loader

        except Exception as e:
            raise RuntimeError(f"Error creating data loaders: {str(e)}")


def group_batches_by_length(dataloader: DataLoader, batch_size: int) -> Dict[int, List[Dict[str, torch.Tensor]]]:
    """
    Groups batches from a dataloader based on annotation list lengths.

    Args:
        dataloader: Original dataloader containing batches
        batch_size: Desired size for the new batches

    Returns:
        Dictionary mapping lengths to lists of grouped batches
    """
    # Collect all items and their lengths
    all_items = []
    lengths = []

    for batch in dataloader:
        batch_annotations = batch['annotations']
        batch_input_ids = batch['input_ids']
        batch_attention_mask = batch['attention_mask']
        batch_labels = batch['labels']
        batch_reduction_percentage = batch['reduction_percentage']

        # Process each item in the batch
        for i in range(len(batch_annotations)):
            # Simply use the length of the annotations list
            length = len(batch_annotations[i])

            # Create individual item
            item = {
                'input_ids': batch_input_ids[i],
                'attention_mask': batch_attention_mask[i],
                'labels': batch_labels[i],
                'annotations': batch_annotations[i],
                'reduction_percentage': batch_reduction_percentage[i]
            }

            all_items.append(item)
            lengths.append(length)

    # Group items by length
    length_groups = defaultdict(list)
    for idx, length in enumerate(lengths):
        length_groups[length].append(all_items[idx])
    # Create new batches for each length group
    grouped_batches = {}

    for length, items in length_groups.items():
        # Split items into batches of specified size
        num_batches = (len(items) + batch_size - 1) // batch_size
        batches = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(items))
            batch_items = items[start_idx:end_idx]

            # Create batch tensors
            batch_dict = {
                'input_ids': torch.stack([item['input_ids'] for item in batch_items]),
                'attention_mask': torch.stack([item['attention_mask'] for item in batch_items]),
                'labels': torch.stack([item['labels'] for item in batch_items]),
                'annotations': [item['annotations'] for item in batch_items],
                'reduction_percentage': [item['reduction_percentage'] for item in batch_items]
            }

            batches.append(batch_dict)

        grouped_batches[length] = batches

    return grouped_batches


def create_batch_ranges(grouped_batches):
    """
    Create batch ranges using TokenSpan start_idx and end_idx

    Args:
        grouped_batches: Dictionary of grouped batches from previous implementation

    Returns:
        Dictionary of batch ranges for each length and batch
    """
    batches_ranges = {}

    for length, batch_groups in grouped_batches.items():
        batches_ranges[length] = []

        for batch in batch_groups:
            batch_annotations = batch['annotations']
            batch_ranges = []

            for annotations in batch_annotations:
                # Create ranges directly from TokenSpan start_idx and end_idx
                ranges = [
                    (ann.start_idx, ann.end_idx)
                    for ann in annotations
                    if not (
                            ann.token_texts == ['<|endoftext|>'] or
                            all(tid == 50256 for tid in ann.token_ids)
                    )
                ]

                batch_ranges.append(ranges)

            batches_ranges[length].append(batch_ranges)

    return batches_ranges

class SynonymReplacementDataset(CustomDataset):
    def __init__(self, data, tokenizer, model_name, dataset_name, replacement_percentage=0.1, stops=None, prompts=None):
        """
        Args:
            data: Original dataset
            tokenizer: Tokenizer for the model
            model_name: Name of the model (e.g., GPT2, Llama-2)
            dataset_name: Name of the task (e.g., idm, sst)
            replacement_percentage: Percentage of words in the content to replace with synonyms
            stops: Stopwords to exclude from replacement
            prompts: Dictionary mapping model/task to their respective prompts
        """
        super().__init__(data)
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.replacement_percentage = replacement_percentage
        self.stops = stops or set(stopwords.words('english'))
        self.prompts = prompts or additional_prompts

    def extract_content(self, input_text: str) -> Tuple[str, str]:
        """
        Split input text into the prompt and content for the specific task/model.
        """
        prompt_format = self.prompts.get(self.model_name, {}).get(self.dataset_name, "")
        if prompt_format and prompt_format in input_text:
            prompt, content = input_text.split(prompt_format, 1)
            return prompt + prompt_format, content.strip()
        return "", input_text

    def synonym_replacement(self, content: str) -> str:
        """
        Replace a percentage of words in the content with synonyms.
        """
        words = content.split(' ')
        num_words_to_replace = max(1, int(len(words) * self.replacement_percentage))
        random_word_list = list(set([word for word in words if word.lower() not in self.stops]))
        random.shuffle(random_word_list)

        replaced_content = words.copy()
        num_replaced = 0

        for word in random_word_list:
            synonyms = get_synonyms(word)
            if synonyms:
                synonym = random.choice(synonyms).replace("_", " ")
                replaced_content = [synonym if w == word else w for w in replaced_content]
                num_replaced += 1
            if num_replaced >= num_words_to_replace:
                break

        return ' '.join(replaced_content)

    def __getitem__(self, idx):
        item = self.data[idx]
        if 'input_ids' in item:
            input_text = self.tokenizer.decode(item['input_ids'], skip_special_tokens=True)

            # Extract prompt and content dynamically
            prompt, content = self.extract_content(input_text)

            # Apply synonym replacement only to the content
            modified_content = self.synonym_replacement(content)
            modified_text = f"{prompt}{modified_content}"

            # Re-tokenize the modified text
            input_encoding = self.tokenizer(
                modified_text,
                truncation=True,
                padding='max_length',
                max_length=len(item['input_ids']),
                return_tensors='pt'
            )

            item['input_ids'] = input_encoding.input_ids.squeeze()
            item['attention_mask'] = input_encoding.attention_mask.squeeze()

        return item


def get_synonyms(word, match_pos=True, num_synonyms=5):
    synonyms = []
    for syn in wn.synsets(word):
        for lm in syn.lemmas():
            if len(synonyms) >= num_synonyms:
                break
            if lm.name() != word:
                if match_pos and lm.synset().pos() != syn.pos():
                    continue
                synonyms.append(lm.name())
    return synonyms


def create_synonym_replacement_test_loader(original_test_loader, tokenizer, model_name, dataset_name, replacement_percentage=0.2):
    """
    Create a DataLoader for synonym replacement intervention on the test dataset.

    Args:
        original_test_loader: DataLoader containing the original test dataset.
        tokenizer: Tokenizer associated with the model.
        model_name: Name of the model (e.g., GPT2, Llama-2).
        dataset_name: Name of the task (e.g., idm, sst).
        replacement_percentage: Percentage of words to replace with synonyms in the content.

    Returns:
        DataLoader: A DataLoader for the modified dataset.
    """
    # Extract the raw data from the original DataLoader
    test_data = original_test_loader.dataset.data

    # Create the synonym replacement dataset
    synonym_dataset = SynonymReplacementDataset(
        data=test_data,
        tokenizer=tokenizer,
        model_name=model_name,
        dataset_name=dataset_name,
        replacement_percentage=replacement_percentage,
        stops=stops,
        prompts=additional_prompts
    )

    # Create a new DataLoader for the synonym-replaced dataset
    return DataLoader(
        synonym_dataset,
        batch_size=original_test_loader.batch_size,
        shuffle=False,
        collate_fn=original_test_loader.collate_fn
    )