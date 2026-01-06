"""
Phonetic Vocabulary Builder
============================
This module creates a vocabulary from phonetic representations
and converts text to numerical tokens for the neural network.

Key Concepts:
- Token: A single unit (phoneme or phoneme sequence)
- Vocabulary: Mapping from tokens to numbers
- Padding: Making all sequences the same length
- Special Tokens: [PAD], [UNK] for padding and unknown tokens
"""

import pandas as pd
from collections import Counter
import json
import os


class PhoneticVocabulary:
    """
    Builds and manages vocabulary for phonetic text.
    
    Special Tokens:
    - [PAD] = 0: Used to pad shorter sequences
    - [UNK] = 1: Used for unknown phonemes
    """
    
    def __init__(self, max_vocab_size=500):
        """
        Initialize vocabulary.
        
        Args:
            max_vocab_size (int): Maximum vocabulary size
        """
        self.max_vocab_size = max_vocab_size
        
        # Special tokens
        self.PAD_TOKEN = "[PAD]"
        self.UNK_TOKEN = "[UNK]"
        
        # Mappings (will be built later)
        self.token_to_id = {}
        self.id_to_token = {}
        self.token_counts = Counter()
        
        # Initialize with special tokens
        self._add_special_tokens()
        
    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        self.token_to_id[self.PAD_TOKEN] = 0
        self.token_to_id[self.UNK_TOKEN] = 1
        self.id_to_token[0] = self.PAD_TOKEN
        self.id_to_token[1] = self.UNK_TOKEN
    
    def _tokenize(self, phonetic_text):
        """
        Split phonetic text into tokens.
        
        Args:
            phonetic_text (str): Phonetic string like "/jaːr jeː muːviː/"
            
        Returns:
            list: List of phoneme tokens
        """
        # Remove slashes and clean
        clean_text = phonetic_text.strip("/").strip()
        
        # Split by space to get individual phoneme units
        tokens = clean_text.split()
        
        return tokens
    
    def build_vocab(self, texts):
        """
        Build vocabulary from a list of phonetic texts.
        
        Args:
            texts (list): List of phonetic strings
        """
        print("Building vocabulary...")
        
        # Count all tokens
        for text in texts:
            tokens = self._tokenize(text)
            self.token_counts.update(tokens)
        
        print(f"  Total unique tokens found: {len(self.token_counts)}")
        
        # Get most common tokens (leaving room for special tokens)
        most_common = self.token_counts.most_common(self.max_vocab_size - 2)
        
        # Add to vocabulary
        for token, count in most_common:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        print(f"  Vocabulary size: {len(self.token_to_id)}")
        print(f"  Special tokens: [PAD]=0, [UNK]=1")
        
    def encode(self, phonetic_text, max_length=50):
        """
        Convert phonetic text to token IDs.
        
        Args:
            phonetic_text (str): Phonetic string
            max_length (int): Maximum sequence length (will pad/truncate)
            
        Returns:
            list: List of token IDs
        """
        tokens = self._tokenize(phonetic_text)
        
        # Convert to IDs
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.token_to_id[self.UNK_TOKEN])  # Unknown token
        
        # Truncate if too long
        if len(ids) > max_length:
            ids = ids[:max_length]
        
        # Pad if too short
        while len(ids) < max_length:
            ids.append(self.token_to_id[self.PAD_TOKEN])
        
        return ids
    
    def decode(self, ids):
        """
        Convert token IDs back to phonetic text.
        
        Args:
            ids (list): List of token IDs
            
        Returns:
            str: Phonetic string
        """
        tokens = []
        for idx in ids:
            if idx in self.id_to_token:
                token = self.id_to_token[idx]
                if token not in [self.PAD_TOKEN, self.UNK_TOKEN]:
                    tokens.append(token)
        
        return "/" + " ".join(tokens) + "/"
    
    def save(self, filepath):
        """
        Save vocabulary to JSON file.
        
        Args:
            filepath (str): Path to save file
        """
        vocab_data = {
            "token_to_id": self.token_to_id,
            "max_vocab_size": self.max_vocab_size,
            "token_counts": dict(self.token_counts.most_common(100))  # Save top 100 counts
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"Vocabulary saved to: {filepath}")
    
    def load(self, filepath):
        """
        Load vocabulary from JSON file.
        
        Args:
            filepath (str): Path to load file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.token_to_id = vocab_data["token_to_id"]
        self.max_vocab_size = vocab_data["max_vocab_size"]
        
        # Rebuild id_to_token
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        
        print(f"Vocabulary loaded from: {filepath}")
        print(f"  Vocabulary size: {len(self.token_to_id)}")
    
    def __len__(self):
        """Return vocabulary size."""
        return len(self.token_to_id)
    
    def get_stats(self):
        """Return vocabulary statistics."""
        return {
            "vocab_size": len(self.token_to_id),
            "max_vocab_size": self.max_vocab_size,
            "unique_tokens": len(self.token_counts),
            "total_tokens": sum(self.token_counts.values()),
            "top_10_tokens": self.token_counts.most_common(10)
        }


def build_vocabulary_from_dataset():
    """
    Build vocabulary from our processed dataset.
    
    Returns:
        PhoneticVocabulary: Built vocabulary object
    """
    # Get the path to data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data', 'processed')
    
    # Load all phonetic data
    train_df = pd.read_csv(os.path.join(data_dir, 'train_phonetic.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val_phonetic.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_phonetic.csv'))
    
    # Combine all phonetic texts
    all_texts = list(train_df['phonetic']) + list(val_df['phonetic']) + list(test_df['phonetic'])
    
    print(f"Total texts for vocabulary: {len(all_texts)}")
    
    # Build vocabulary
    vocab = PhoneticVocabulary(max_vocab_size=500)
    vocab.build_vocab(all_texts)
    
    return vocab


# =============================================================
# TEST THE VOCABULARY
# =============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Phonetic Vocabulary")
    print("=" * 60)
    
    # Build vocabulary from dataset
    vocab = build_vocabulary_from_dataset()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Vocabulary Statistics")
    print("=" * 60)
    
    stats = vocab.get_stats()
    print(f"\nVocabulary size: {stats['vocab_size']}")
    print(f"Unique tokens found: {stats['unique_tokens']}")
    print(f"Total token occurrences: {stats['total_tokens']}")
    
    print("\nTop 10 most common phonemes:")
    for token, count in stats['top_10_tokens']:
        print(f"  {token:<15} : {count:>4} times")
    
    # Test encoding and decoding
    print("\n" + "=" * 60)
    print("Testing Encode/Decode")
    print("=" * 60)
    
    test_texts = [
        "/jaːr jeː muːviː toː bəhʊt əmeɪzɪŋ tʰiː/",
        "/kjaː bəkvaːs hɛː/",
        "/tʃaːiː piːneː tʃəlẽː/",
    ]
    
    for text in test_texts:
        print(f"\nOriginal:  {text}")
        
        # Encode
        ids = vocab.encode(text, max_length=15)
        print(f"Encoded:   {ids}")
        
        # Decode
        decoded = vocab.decode(ids)
        print(f"Decoded:   {decoded}")
    
    # Save vocabulary
    print("\n" + "=" * 60)
    print("Saving Vocabulary")
    print("=" * 60)
    
    # Get path for saving
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(current_dir, '..', 'models', 'phonetic_vocab.json')
    
    vocab.save(vocab_path)
    
    print("\n✓ Vocabulary ready for model training!")