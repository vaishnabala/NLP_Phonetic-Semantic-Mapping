"""
Triplet Dataset for Contrastive Learning
=========================================
This module creates triplets (anchor, positive, negative) for training.

What is a Triplet?
- Anchor: A sample we're comparing
- Positive: A sample with SAME label as anchor
- Negative: A sample with DIFFERENT label than anchor

Why Triplets?
- Teaches model to group similar items together
- Pushes different items apart in vector space
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phonetic_vocabulary import PhoneticVocabulary


class TripletDataset(Dataset):
    """
    PyTorch Dataset that generates triplets for contrastive learning.
    
    For each sample (anchor), we find:
    - Positive: Another sample with the SAME sentiment
    - Negative: A sample with a DIFFERENT sentiment
    """
    
    def __init__(self, csv_path, vocab, max_length=50):
        """
        Initialize the triplet dataset.
        
        Args:
            csv_path (str): Path to phonetic CSV file
            vocab (PhoneticVocabulary): Vocabulary for encoding
            max_length (int): Maximum sequence length
        """
        self.vocab = vocab
        self.max_length = max_length
        
        # Load data
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        
        # Create label mapping
        self.label_to_idx = {"positive": 0, "negative": 1, "neutral": 2}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
        # Group samples by label for efficient triplet generation
        self.label_groups = {}
        for label in self.label_to_idx.keys():
            self.label_groups[label] = self.df[self.df['label'] == label].index.tolist()
        
        print(f"Samples per label:")
        for label, indices in self.label_groups.items():
            print(f"  {label}: {len(indices)}")
    
    def __len__(self):
        """Return dataset size (number of anchors)."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a triplet (anchor, positive, negative).
        
        Args:
            idx (int): Index of anchor sample
            
        Returns:
            dict: Contains anchor, positive, negative tensors and labels
        """
        # Get anchor
        anchor_row = self.df.iloc[idx]
        anchor_text = anchor_row['phonetic']
        anchor_label = anchor_row['label']
        
        # Get positive (same label, different sample)
        positive_idx = self._get_positive_idx(idx, anchor_label)
        positive_row = self.df.iloc[positive_idx]
        positive_text = positive_row['phonetic']
        
        # Get negative (different label)
        negative_idx = self._get_negative_idx(anchor_label)
        negative_row = self.df.iloc[negative_idx]
        negative_text = negative_row['phonetic']
        negative_label = negative_row['label']
        
        # Encode all three
        anchor_ids = self.vocab.encode(anchor_text, self.max_length)
        positive_ids = self.vocab.encode(positive_text, self.max_length)
        negative_ids = self.vocab.encode(negative_text, self.max_length)
        
        return {
            'anchor': torch.tensor(anchor_ids, dtype=torch.long),
            'positive': torch.tensor(positive_ids, dtype=torch.long),
            'negative': torch.tensor(negative_ids, dtype=torch.long),
            'anchor_label': self.label_to_idx[anchor_label],
            'negative_label': self.label_to_idx[negative_label],
        }
    
    def _get_positive_idx(self, anchor_idx, anchor_label):
        """
        Get index of a positive sample (same label, different sample).
        
        Args:
            anchor_idx (int): Index of anchor
            anchor_label (str): Label of anchor
            
        Returns:
            int: Index of positive sample
        """
        # Get all samples with same label
        same_label_indices = self.label_groups[anchor_label].copy()
        
        # Remove anchor from options
        if anchor_idx in same_label_indices:
            same_label_indices.remove(anchor_idx)
        
        # If no other samples with same label, return anchor itself
        if len(same_label_indices) == 0:
            return anchor_idx
        
        # Random selection
        return random.choice(same_label_indices)
    
    def _get_negative_idx(self, anchor_label):
        """
        Get index of a negative sample (different label).
        
        Args:
            anchor_label (str): Label of anchor
            
        Returns:
            int: Index of negative sample
        """
        # Get labels that are different from anchor
        other_labels = [l for l in self.label_groups.keys() if l != anchor_label]
        
        # Random selection of different label
        negative_label = random.choice(other_labels)
        
        # Random sample from that label
        return random.choice(self.label_groups[negative_label])
    
    def get_sample_triplet(self, idx):
        """
        Get a triplet with human-readable text (for debugging).
        
        Args:
            idx (int): Index of anchor
            
        Returns:
            dict: Triplet with text and labels
        """
        anchor_row = self.df.iloc[idx]
        anchor_label = anchor_row['label']
        
        positive_idx = self._get_positive_idx(idx, anchor_label)
        negative_idx = self._get_negative_idx(anchor_label)
        
        return {
            'anchor': {
                'text': anchor_row['text'],
                'phonetic': anchor_row['phonetic'],
                'label': anchor_label
            },
            'positive': {
                'text': self.df.iloc[positive_idx]['text'],
                'phonetic': self.df.iloc[positive_idx]['phonetic'],
                'label': self.df.iloc[positive_idx]['label']
            },
            'negative': {
                'text': self.df.iloc[negative_idx]['text'],
                'phonetic': self.df.iloc[negative_idx]['phonetic'],
                'label': self.df.iloc[negative_idx]['label']
            }
        }


class SimpleDataset(Dataset):
    """
    Simple dataset for evaluation (no triplets, just samples and labels).
    """
    
    def __init__(self, csv_path, vocab, max_length=50):
        """
        Initialize simple dataset.
        
        Args:
            csv_path (str): Path to phonetic CSV
            vocab (PhoneticVocabulary): Vocabulary
            max_length (int): Maximum sequence length
        """
        self.vocab = vocab
        self.max_length = max_length
        
        # Load data
        self.df = pd.read_csv(csv_path)
        
        # Label mapping
        self.label_to_idx = {"positive": 0, "negative": 1, "neutral": 2}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Encode text
        ids = self.vocab.encode(row['phonetic'], self.max_length)
        
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'label': self.label_to_idx[row['label']]
        }


def create_data_loaders(vocab, batch_size=16, max_length=50):
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        vocab (PhoneticVocabulary): Vocabulary
        batch_size (int): Batch size
        max_length (int): Maximum sequence length
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Get data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data', 'processed')
    
    # Create datasets
    train_dataset = TripletDataset(
        os.path.join(data_dir, 'train_phonetic.csv'),
        vocab, max_length
    )
    
    val_dataset = TripletDataset(
        os.path.join(data_dir, 'val_phonetic.csv'),
        vocab, max_length
    )
    
    test_dataset = TripletDataset(
        os.path.join(data_dir, 'test_phonetic.csv'),
        vocab, max_length
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader


# =============================================================
# TEST THE TRIPLET DATASET
# =============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Triplet Dataset")
    print("=" * 60)
    
    # Load vocabulary
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(current_dir, '..', 'models', 'phonetic_vocab.json')
    
    vocab = PhoneticVocabulary()
    vocab.load(vocab_path)
    
    # Create datasets
    data_dir = os.path.join(current_dir, '..', 'data', 'processed')
    train_path = os.path.join(data_dir, 'train_phonetic.csv')
    
    dataset = TripletDataset(train_path, vocab, max_length=20)
    
    # Show sample triplets
    print("\n" + "=" * 60)
    print("Sample Triplets (Human Readable)")
    print("=" * 60)
    
    for i in range(3):
        print(f"\n--- Triplet {i + 1} ---")
        triplet = dataset.get_sample_triplet(i)
        
        print(f"\n  ANCHOR ({triplet['anchor']['label']}):")
        print(f"    Text:     {triplet['anchor']['text']}")
        print(f"    Phonetic: {triplet['anchor']['phonetic']}")
        
        print(f"\n  POSITIVE ({triplet['positive']['label']}):")
        print(f"    Text:     {triplet['positive']['text']}")
        print(f"    Phonetic: {triplet['positive']['phonetic']}")
        
        print(f"\n  NEGATIVE ({triplet['negative']['label']}):")
        print(f"    Text:     {triplet['negative']['text']}")
        print(f"    Phonetic: {triplet['negative']['phonetic']}")
    
    # Test tensor output
    print("\n" + "=" * 60)
    print("Tensor Output (for Model)")
    print("=" * 60)
    
    sample = dataset[0]
    print(f"\nAnchor tensor shape:   {sample['anchor'].shape}")
    print(f"Positive tensor shape: {sample['positive'].shape}")
    print(f"Negative tensor shape: {sample['negative'].shape}")
    print(f"Anchor label:          {sample['anchor_label']}")
    print(f"Negative label:        {sample['negative_label']}")
    
    # Test DataLoader
    print("\n" + "=" * 60)
    print("Testing DataLoader")
    print("=" * 60)
    
    train_loader, val_loader, test_loader = create_data_loaders(vocab, batch_size=8)
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    
    # Get one batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Anchor:   {batch['anchor'].shape}")
    print(f"  Positive: {batch['positive'].shape}")
    print(f"  Negative: {batch['negative'].shape}")
    
    print("\n✓ Triplet Dataset ready for training!")