# import nltk
# from nltk.corpus import wordnet as wn
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.tag import pos_tag
# from textblob import TextBlob
# import spacy
# from collections import defaultdict
# import numpy as np
#
#
# class SynonymGenerator:
#     def __init__(self):
#         self.nlp = spacy.load('en_core_web_md')
#         self.stop_words = set(stopwords.words('english'))
#
#     def get_word_pos(self, word, context):
#         """Get part of speech tag for a word in context."""
#         if not context:
#             return None
#
#         # Tokenize and POS tag the context
#         tokens = word_tokenize(context)
#         pos_tags = pos_tag(tokens)
#
#         # Find the word in context and get its POS tag
#         word_lower = word.lower()
#         for token, pos in pos_tags:
#             if token.lower() == word_lower:
#                 return pos
#         return None
#
#     def get_sentiment(self, word):
#         """Get sentiment polarity of a word."""
#         return TextBlob(word).sentiment.polarity
#
#     def clean_synonym(self, synonym):
#         """Clean and validate a synonym."""
#         # Convert to lowercase and handle basic cleaning
#         cleaned = synonym.replace('_', ' ').lower().strip()
#
#         # Handle compound words with "to" - avoid splitting phrases like "belonging to"
#         if " to " in cleaned and len(cleaned.split()) > 2:
#             return None
#
#         # Check if it contains only valid characters
#         if not all(c.isalpha() or c.isspace() for c in cleaned):
#             return None
#
#         # Validate word length and structure
#         words = cleaned.split()
#         if any((len(word) < 2 or word in self.stop_words) for word in words):
#             return None
#
#         # Prevent partial matches of compound words
#         if len(words) > 1 and any(word.endswith('ing') for word in words[:-1]):
#             return None
#
#         # Additional validation for multi-word synonyms
#         if len(words) > 2:
#             return None  # Limit to single words or two-word compounds
#
#         return cleaned
#
#     def get_spacy_synonyms(self, word, context, n=10):
#         """Get synonyms using spaCy similarity."""
#         doc = self.nlp(context if context else word)
#         word_token = self.nlp(word)[0]
#
#         similar_words = []
#         for token in self.nlp.vocab:
#             # Skip tokens without vectors or invalid tokens
#             if not token.has_vector or not token.is_alpha:
#                 continue
#
#             # Skip the original word and stop words
#             if (token.text.lower() == word.lower() or
#                     token.text.lower() in self.stop_words):
#                 continue
#
#             # Check POS using a temporary doc
#             token_doc = self.nlp(token.text)
#             if len(token_doc) > 0 and word_token.pos_ != token_doc[0].pos_:
#                 continue
#
#             # Skip if token is part of a compound word
#             if ' ' in token.text:
#                 continue
#
#             similarity = token.similarity(word_token)
#             if similarity > 0.6:  # Increased threshold for better quality
#                 cleaned = self.clean_synonym(token.text)
#                 if cleaned:
#                     similar_words.append((cleaned, similarity))
#
#         return [word for word, score in sorted(similar_words, key=lambda x: x[1], reverse=True)[:n]]
#
#     def rank_synonyms(self, synonyms, original_word, context):
#         """Rank synonyms based on multiple criteria."""
#         ranked_synonyms = defaultdict(float)
#         original_sentiment = self.get_sentiment(original_word)
#         original_token = self.nlp(original_word)[0]
#
#         for synonym in synonyms:
#             score = 0.0
#
#             # Sentiment similarity (0-1)
#             sentiment_diff = abs(original_sentiment - self.get_sentiment(synonym))
#             score += (1 - sentiment_diff) * 0.4
#
#             # Context similarity using spaCy (0-1)
#             synonym_token = self.nlp(synonym)[0]
#             if synonym_token.has_vector and original_token.has_vector:
#                 similarity = synonym_token.similarity(original_token)
#                 score += similarity * 0.6
#
#             ranked_synonyms[synonym] = score
#
#         return sorted(ranked_synonyms.items(), key=lambda x: x[1], reverse=True)
#
#     def get_synonyms(self, word, context="", num_synonyms=5, min_word_len=3):
#         """
#         Get more natural synonyms for a word considering its context.
#
#         Args:
#             word: The word to find synonyms for
#             context: The full text context where the word appears
#             num_synonyms: Maximum number of synonyms to return
#             min_word_len: Minimum length of word to process
#         """
#         if len(word) < min_word_len:
#             return []
#
#         if word.lower() in self.stop_words:
#             return []
#
#         # Get POS tag for the word in context
#         context_pos = self.get_word_pos(word, context)
#
#         # Collect synonyms from multiple sources
#         synonyms = set()
#
#         # 1. WordNet synonyms
#         for syn in wn.synsets(word):
#             if context_pos:
#                 if not self._check_pos_match(syn.pos(), context_pos):
#                     continue
#
#             for lemma in syn.lemmas():
#                 synonym = self.clean_synonym(lemma.name())
#                 if synonym and synonym != word.lower():
#                     synonyms.add(synonym)
#
#         # 2. SpaCy synonyms
#         spacy_synonyms = self.get_spacy_synonyms(word, context)
#         synonyms.update(spacy_synonyms)
#
#         # Rank and filter synonyms
#         ranked_synonyms = self.rank_synonyms(synonyms, word, context)
#
#         # Return top N synonyms
#         return [syn for syn, score in ranked_synonyms[:num_synonyms]]
#
#     def _check_pos_match(self, wordnet_pos, nltk_pos):
#         """Check if WordNet POS matches NLTK POS tag."""
#         pos_map = {
#             'n': ['NN', 'NNS', 'NNP', 'NNPS'],
#             'v': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
#             'a': ['JJ', 'JJR', 'JJS'],
#             'r': ['RB', 'RBR', 'RBS']
#         }
#         return nltk_pos in pos_map.get(wordnet_pos, [])
#
#
# # Example usage:
# """
# # Initialize the generator
# synonym_generator = SynonymGenerator()
#
# # Example 1: Word without context
# synonyms = synonym_generator.get_synonyms("beautiful")
# print("Synonyms for 'beautiful':", synonyms)
#
# # Example 2: Word with context
# context = "The beautiful sunset painted the sky with vibrant colors."
# synonyms = synonym_generator.get_synonyms("beautiful", context)
# print("Synonyms for 'beautiful' in context:", synonyms)
# """

import nltk
from .utils import ensure_nltk_resources
# Ensure all required NLTK datasets are available
ensure_nltk_resources()
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from textblob import TextBlob
import spacy
from collections import defaultdict
import numpy as np
from functools import lru_cache


class SynonymGenerator:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_md')
        self.stop_words = set(stopwords.words('english'))
        # Pre-compute most common words for faster similarity
        self.common_words = set()
        for word in self.nlp.vocab:
            if word.is_alpha and word.prob >= -15:  # Filter for common words
                self.common_words.add(word.text.lower())

    @lru_cache(maxsize=1024)
    def get_word_pos(self, word, context_key=None):
        """Get part of speech tag for a word in context. Cached version."""
        if not context_key:
            return None

        tokens = word_tokenize(context_key)
        pos_tags = pos_tag(tokens)

        word_lower = word.lower()
        for token, pos in pos_tags:
            if token.lower() == word_lower:
                return pos
        return None

    @lru_cache(maxsize=1024)
    def get_sentiment(self, word):
        """Cached sentiment analysis."""
        return TextBlob(word).sentiment.polarity

    def clean_synonym(self, synonym):
        """Clean and validate a synonym."""
        cleaned = synonym.replace('_', ' ').lower().strip()

        # Quick rejections first
        if len(cleaned) < 2 or cleaned in self.stop_words:
            return None

        if " to " in cleaned and len(cleaned.split()) > 2:
            return None

        if not all(c.isalpha() or c.isspace() for c in cleaned):
            return None

        words = cleaned.split()
        if len(words) > 2:  # Limit to single words or two-word compounds
            return None

        if any((len(word) < 2 or word in self.stop_words) for word in words):
            return None

        return cleaned

    def get_spacy_synonyms(self, word, context, n=10):
        """Get synonyms using spaCy similarity - optimized version."""
        word_token = self.nlp(word)[0]
        word_vector = word_token.vector

        # Only compare with common words that we pre-computed
        candidates = []
        for candidate in self.common_words:
            if candidate == word.lower() or candidate in self.stop_words:
                continue

            candidate_token = self.nlp(candidate)[0]
            if candidate_token.has_vector:
                # Use numpy for faster vector similarity computation
                similarity = np.dot(word_vector, candidate_token.vector) / (
                        np.linalg.norm(word_vector) * np.linalg.norm(candidate_token.vector))

                if similarity > 0.6:
                    cleaned = self.clean_synonym(candidate)
                    if cleaned:
                        candidates.append((cleaned, similarity))

        return [word for word, _ in sorted(candidates, key=lambda x: x[1], reverse=True)[:n]]

    @lru_cache(maxsize=512)
    def rank_synonyms(self, synonyms_tuple, original_word, context_key=None):
        """Rank synonyms based on multiple criteria. Cached version."""
        synonyms = list(synonyms_tuple)  # Convert tuple to list
        ranked_synonyms = defaultdict(float)
        original_sentiment = self.get_sentiment(original_word)
        original_token = self.nlp(original_word)[0]

        for synonym in synonyms:
            score = 0.0

            # Sentiment similarity (0-1)
            sentiment_diff = abs(original_sentiment - self.get_sentiment(synonym))
            score += (1 - sentiment_diff) * 0.4

            # Context similarity using spaCy (0-1)
            synonym_token = self.nlp(synonym)[0]
            if synonym_token.has_vector and original_token.has_vector:
                similarity = synonym_token.similarity(original_token)
                score += similarity * 0.6

            ranked_synonyms[synonym] = score

        return sorted(ranked_synonyms.items(), key=lambda x: x[1], reverse=True)

    def get_synonyms(self, word, context="", num_synonyms=5, min_word_len=3):
        """Get synonyms with improved performance."""
        if len(word) < min_word_len or word.lower() in self.stop_words:
            return []

        # Convert context to a hashable type for caching
        context_key = context if context else None
        context_pos = self.get_word_pos(word, context_key)

        # Collect synonyms from multiple sources
        synonyms = set()

        # 1. WordNet synonyms (relatively fast)
        for syn in wn.synsets(word):
            if context_pos and not self._check_pos_match(syn.pos(), context_pos):
                continue

            for lemma in syn.lemmas():
                synonym = self.clean_synonym(lemma.name())
                if synonym and synonym != word.lower():
                    synonyms.add(synonym)

        # 2. SpaCy synonyms (optimized)
        spacy_synonyms = self.get_spacy_synonyms(word, context)
        synonyms.update(spacy_synonyms)

        # Convert set to tuple for caching
        synonyms_tuple = tuple(sorted(synonyms))

        # Rank and return top synonyms
        ranked_synonyms = self.rank_synonyms(synonyms_tuple, word, context_key)
        return [syn for syn, score in ranked_synonyms[:num_synonyms]]

    def _check_pos_match(self, wordnet_pos, nltk_pos):
        """Check if WordNet POS matches NLTK POS tag."""
        pos_map = {
            'n': ['NN', 'NNS', 'NNP', 'NNPS'],
            'v': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            'a': ['JJ', 'JJR', 'JJS'],
            'r': ['RB', 'RBR', 'RBS']
        }
        return nltk_pos in pos_map.get(wordnet_pos, [])


# Example usage:
"""
# Initialize once
synonym_generator = SynonymGenerator()

# Get synonyms (will be faster after first call due to caching)
word = "beautiful"
context = "The beautiful sunset painted the sky with vibrant colors."
synonyms = synonym_generator.get_synonyms(word, context)
print(f"Synonyms for '{word}': {synonyms}")
"""