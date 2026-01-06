"""
Enhanced Phonetic Encoder with Prosody Features
=================================================
Combines phonetic sequence encoding with prosody features
for richer representation of code-mixed text.

This model leverages:
1. LSTM for sequential phonetic patterns
2. Prosody features for rhythm/stress patterns (music-inspired!)
3. Attention mechanism to focus on important parts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prosody_features import ProsodyExtractor


class EnhancedPhoneticEncoder(nn.Module):
    """
    Enhanced encoder that combines phonetic embeddings with prosody features.
    
    Architecture:
    1. Phonetic Encoder (LSTM) → 64 dim
    2. Prosody Features → 24 dim
    3. Combined → 88 dim → Project → 64 dim
    """
    
    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        output_dim=64,
        prosody_dim=24,
        num_layers=2,
        dropout=0.3,
        use_attention=True
    ):
        """
        Initialize enhanced encoder.
        
        Args:
            vocab_size (int): Size of phoneme vocabulary
            embedding_dim (int): Phoneme embedding dimension
            hidden_dim (int): LSTM hidden dimension
            output_dim (int): Final output dimension
            prosody_dim (int): Prosody feature dimension
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
            use_attention (bool): Whether to use attention
        """
        super(EnhancedPhoneticEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.prosody_dim = prosody_dim
        self.use_attention = use_attention
        
        # ============= Phonetic Encoder =============
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer (optional)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        # ============= Prosody Processing =============
        
        # Prosody feature processor
        self.prosody_fc = nn.Sequential(
            nn.Linear(prosody_dim, prosody_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prosody_dim * 2, prosody_dim),
            nn.ReLU()
        )
        
        # ============= Combination Layer =============
        
        # Phonetic output: hidden_dim * 2 (bidirectional)
        # Prosody output: prosody_dim
        combined_dim = hidden_dim * 2 + prosody_dim
        
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Print model info
        print(f"EnhancedPhoneticEncoder initialized:")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  Prosody dim: {prosody_dim}")
        print(f"  Use attention: {use_attention}")
        print(f"  Total parameters: {self.count_parameters():,}")
    
    def forward(self, x, prosody_features=None):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Token IDs (batch_size, seq_length)
            prosody_features (torch.Tensor): Prosody features (batch_size, prosody_dim)
                                            If None, uses zero features
        
        Returns:
            torch.Tensor: Embeddings (batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # ============= Phonetic Encoding =============
        
        # Embedding
        embedded = self.embedding(x)  # (batch, seq, embed_dim)
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: (batch, seq, hidden*2)
        
        # Get phonetic representation
        if self.use_attention:
            # Attention mechanism
            attn_weights = self.attention(lstm_out)  # (batch, seq, 1)
            attn_weights = F.softmax(attn_weights, dim=1)
            phonetic_repr = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)
        else:
            # Use final hidden states
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            phonetic_repr = torch.cat([hidden_forward, hidden_backward], dim=1)
        
        # ============= Prosody Processing =============
        
        if prosody_features is None:
            # Use zero features if not provided
            prosody_features = torch.zeros(batch_size, self.prosody_dim, device=x.device)
        
        prosody_repr = self.prosody_fc(prosody_features)  # (batch, prosody_dim)
        
        # ============= Combination =============
        
        # Concatenate phonetic and prosody
        combined = torch.cat([phonetic_repr, prosody_repr], dim=1)
        
        # Project to output dimension
        output = self.combiner(combined)
        
        # Normalize
        output = self.layer_norm(output)
        output = F.normalize(output, p=2, dim=1)
        
        return output
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnhancedTripletNet(nn.Module):
    """
    Triplet network using enhanced encoder with prosody.
    """
    
    def __init__(self, encoder, prosody_extractor=None):
        """
        Initialize.
        
        Args:
            encoder (EnhancedPhoneticEncoder): The encoder
            prosody_extractor (ProsodyExtractor): For extracting prosody features
        """
        super(EnhancedTripletNet, self).__init__()
        self.encoder = encoder
        self.prosody_extractor = prosody_extractor or ProsodyExtractor()
    
    def forward(self, anchor, positive, negative, 
                anchor_prosody=None, positive_prosody=None, negative_prosody=None):
        """
        Forward pass for triplet.
        
        Args:
            anchor, positive, negative: Token ID tensors
            anchor_prosody, positive_prosody, negative_prosody: Prosody feature tensors
        
        Returns:
            tuple: (anchor_emb, positive_emb, negative_emb)
        """
        anchor_emb = self.encoder(anchor, anchor_prosody)
        positive_emb = self.encoder(positive, positive_prosody)
        negative_emb = self.encoder(negative, negative_prosody)
        
        return anchor_emb, positive_emb, negative_emb
    
    def get_encoder(self):
        """Return the encoder."""
        return self.encoder


class EnhancedTripletDataset(torch.utils.data.Dataset):
    """
    Dataset that includes prosody features.
    """
    
    def __init__(self, csv_path, vocab, max_length=50):
        """
        Initialize dataset with prosody extraction.
        
        Args:
            csv_path (str): Path to phonetic CSV
            vocab: Vocabulary for encoding
            max_length (int): Max sequence length
        """
        import pandas as pd
        
        self.vocab = vocab
        self.max_length = max_length
        self.prosody_extractor = ProsodyExtractor()
        
        # Load data
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        
        # Label mapping
        self.label_to_idx = {"positive": 0, "negative": 1, "neutral": 2}
        
        # Group by label
        self.label_groups = {}
        for label in self.label_to_idx.keys():
            self.label_groups[label] = self.df[self.df['label'] == label].index.tolist()
        
        # Pre-extract prosody features for all samples
        print("Pre-extracting prosody features...")
        self.prosody_features = []
        for phonetic in self.df['phonetic']:
            features = self.prosody_extractor.extract(phonetic)
            self.prosody_features.append(features)
        print(f"  Extracted {len(self.prosody_features)} prosody vectors")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        import random
        
        # Get anchor
        anchor_row = self.df.iloc[idx]
        anchor_label = anchor_row['label']
        
        # Get positive (same label)
        same_label = self.label_groups[anchor_label].copy()
        if idx in same_label:
            same_label.remove(idx)
        positive_idx = random.choice(same_label) if same_label else idx
        
        # Get negative (different label)
        other_labels = [l for l in self.label_groups.keys() if l != anchor_label]
        neg_label = random.choice(other_labels)
        negative_idx = random.choice(self.label_groups[neg_label])
        
        # Encode texts
        anchor_ids = self.vocab.encode(anchor_row['phonetic'], self.max_length)
        positive_ids = self.vocab.encode(self.df.iloc[positive_idx]['phonetic'], self.max_length)
        negative_ids = self.vocab.encode(self.df.iloc[negative_idx]['phonetic'], self.max_length)
        
        return {
            'anchor': torch.tensor(anchor_ids, dtype=torch.long),
            'positive': torch.tensor(positive_ids, dtype=torch.long),
            'negative': torch.tensor(negative_ids, dtype=torch.long),
            'anchor_prosody': torch.tensor(self.prosody_features[idx], dtype=torch.float),
            'positive_prosody': torch.tensor(self.prosody_features[positive_idx], dtype=torch.float),
            'negative_prosody': torch.tensor(self.prosody_features[negative_idx], dtype=torch.float),
            'anchor_label': self.label_to_idx[anchor_label],
        }


def create_enhanced_model(vocab_size, config=None):
    """
    Create enhanced model with prosody.
    
    Args:
        vocab_size (int): Vocabulary size
        config (dict): Configuration
        
    Returns:
        tuple: (triplet_net, criterion)
    """
    from src.phonetic_encoder import TripletLoss
    
    config = config or {}
    
    encoder = EnhancedPhoneticEncoder(
        vocab_size=vocab_size,
        embedding_dim=config.get('embedding_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        output_dim=config.get('output_dim', 64),
        prosody_dim=config.get('prosody_dim', 24),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3),
        use_attention=config.get('use_attention', True)
    )
    
    triplet_net = EnhancedTripletNet(encoder)
    criterion = TripletLoss(margin=config.get('margin', 0.5))
    
    return triplet_net, criterion


# =============================================================
# TEST THE ENHANCED MODEL
# =============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Enhanced Phonetic Encoder")
    print("=" * 60)
    
    # Test parameters
    vocab_size = 200
    batch_size = 4
    seq_length = 20
    prosody_dim = 24
    
    # Create model
    print("\n--- Creating Enhanced Model ---\n")
    triplet_net, criterion = create_enhanced_model(vocab_size)
    
    # Create dummy input
    print("\n--- Testing Forward Pass ---\n")
    
    anchor = torch.randint(0, vocab_size, (batch_size, seq_length))
    positive = torch.randint(0, vocab_size, (batch_size, seq_length))
    negative = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    anchor_prosody = torch.randn(batch_size, prosody_dim)
    positive_prosody = torch.randn(batch_size, prosody_dim)
    negative_prosody = torch.randn(batch_size, prosody_dim)
    
    print(f"Input shapes:")
    print(f"  Anchor IDs:      {anchor.shape}")
    print(f"  Anchor prosody:  {anchor_prosody.shape}")
    
    # Forward pass
    anchor_emb, positive_emb, negative_emb = triplet_net(
        anchor, positive, negative,
        anchor_prosody, positive_prosody, negative_prosody
    )
    
    print(f"\nOutput shapes:")
    print(f"  Anchor embedding: {anchor_emb.shape}")
    
    # Calculate loss
    loss = criterion(anchor_emb, positive_emb, negative_emb)
    print(f"\nTriplet loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    print("✓ Backward pass successful!")
    
    print("\n" + "=" * 60)
    print("✓ Enhanced model ready!")
    print("=" * 60)