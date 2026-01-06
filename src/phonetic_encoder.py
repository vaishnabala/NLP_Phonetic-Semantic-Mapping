"""
Phonetic Encoder Model
=======================
Neural network that converts phonetic sequences into dense vector embeddings.

Architecture:
1. Embedding Layer: Converts token IDs to dense vectors
2. LSTM Layers: Captures sequential patterns in phonemes
3. Fully Connected: Projects to final embedding space

The output embedding can be used for:
- Similarity comparison (via triplet loss)
- Sentiment classification
- Clustering similar expressions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhoneticEncoder(nn.Module):
    """
    Encodes phonetic sequences into fixed-size embeddings.
    
    This is the core model that learns to represent phonetic text
    in a meaningful vector space where similar sentiments are close.
    """
    
    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        output_dim=64,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    ):
        """
        Initialize the encoder.
        
        Args:
            vocab_size (int): Size of phoneme vocabulary
            embedding_dim (int): Dimension of phoneme embeddings
            hidden_dim (int): LSTM hidden state dimension
            output_dim (int): Final embedding dimension
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
            bidirectional (bool): Use bidirectional LSTM
        """
        super(PhoneticEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Calculate directions multiplier
        self.num_directions = 2 if bidirectional else 1
        
        # Layer 1: Embedding
        # Converts token IDs to dense vectors
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # [PAD] token gets zero vector
        )
        
        # Layer 2: LSTM
        # Processes sequence and captures patterns
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Layer 3: Fully Connected
        # Projects LSTM output to final embedding
        lstm_output_dim = hidden_dim * self.num_directions
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(output_dim)
        
        print(f"PhoneticEncoder initialized:")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  LSTM layers: {num_layers}")
        print(f"  Bidirectional: {bidirectional}")
        print(f"  Total parameters: {self.count_parameters():,}")
    
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input token IDs, shape (batch_size, seq_length)
            
        Returns:
            torch.Tensor: Embeddings, shape (batch_size, output_dim)
        """
        # x shape: (batch_size, seq_length)
        batch_size = x.size(0)
        
        # Step 1: Embedding lookup
        # (batch_size, seq_length) → (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # Step 2: LSTM processing
        # (batch_size, seq_length, embedding_dim) → (batch_size, seq_length, hidden_dim * num_directions)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Step 3: Get final representation
        # Option A: Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
            hidden_forward = hidden[-2, :, :]  # Last forward layer
            hidden_backward = hidden[-1, :, :]  # Last backward layer
            final_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            final_hidden = hidden[-1, :, :]  # Last layer hidden state
        
        # Step 4: Project to output dimension
        # (batch_size, hidden_dim * num_directions) → (batch_size, output_dim)
        output = self.fc(final_hidden)
        output = self.dropout(output)
        
        # Step 5: Layer normalization
        output = self.layer_norm(output)
        
        # Step 6: L2 normalize for cosine similarity
        output = F.normalize(output, p=2, dim=1)
        
        return output
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_embedding(self, x):
        """
        Get embedding for a single sample or batch (alias for forward).
        
        Args:
            x (torch.Tensor): Input token IDs
            
        Returns:
            torch.Tensor: Normalized embeddings
        """
        return self.forward(x)


class TripletNet(nn.Module):
    """
    Wrapper that processes triplets through the encoder.
    
    Takes anchor, positive, and negative samples,
    passes them through the same encoder, and returns their embeddings.
    """
    
    def __init__(self, encoder):
        """
        Initialize TripletNet.
        
        Args:
            encoder (PhoneticEncoder): The encoder model
        """
        super(TripletNet, self).__init__()
        self.encoder = encoder
    
    def forward(self, anchor, positive, negative):
        """
        Get embeddings for a triplet.
        
        Args:
            anchor (torch.Tensor): Anchor samples
            positive (torch.Tensor): Positive samples (same class)
            negative (torch.Tensor): Negative samples (different class)
            
        Returns:
            tuple: (anchor_emb, positive_emb, negative_emb)
        """
        anchor_emb = self.encoder(anchor)
        positive_emb = self.encoder(positive)
        negative_emb = self.encoder(negative)
        
        return anchor_emb, positive_emb, negative_emb
    
    def get_encoder(self):
        """Return the underlying encoder."""
        return self.encoder


class TripletLoss(nn.Module):
    """
    Triplet Loss for contrastive learning.
    
    Formula:
        Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    
    Where:
        d() = distance function (Euclidean)
        margin = minimum desired gap between positive and negative distances
    
    Goal:
        Make anchor closer to positive than to negative by at least 'margin'
    """
    
    def __init__(self, margin=1.0):
        """
        Initialize triplet loss.
        
        Args:
            margin (float): Minimum margin between positive and negative distances
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Calculate triplet loss.
        
        Args:
            anchor (torch.Tensor): Anchor embeddings (batch_size, embedding_dim)
            positive (torch.Tensor): Positive embeddings
            negative (torch.Tensor): Negative embeddings
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Calculate distances
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        
        # Calculate triplet loss
        # We want: pos_distance < neg_distance - margin
        # Loss = max(0, pos_distance - neg_distance + margin)
        losses = F.relu(pos_distance - neg_distance + self.margin)
        
        return losses.mean()


def create_model(vocab_size, config=None):
    """
    Create encoder and triplet network with given configuration.
    
    Args:
        vocab_size (int): Vocabulary size
        config (dict): Model configuration (optional)
        
    Returns:
        tuple: (triplet_net, criterion)
    """
    # Default configuration
    if config is None:
        config = {
            "embedding_dim": 64,
            "hidden_dim": 128,
            "output_dim": 64,
            "num_layers": 2,
            "dropout": 0.3,
        }
    
    # Create encoder
    encoder = PhoneticEncoder(
        vocab_size=vocab_size,
        embedding_dim=config.get("embedding_dim", 64),
        hidden_dim=config.get("hidden_dim", 128),
        output_dim=config.get("output_dim", 64),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.3),
        bidirectional=True
    )
    
    # Create triplet network
    triplet_net = TripletNet(encoder)
    
    # Create loss function
    criterion = TripletLoss(margin=config.get("margin", 1.0))
    
    return triplet_net, criterion


# =============================================================
# TEST THE MODEL
# =============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Phonetic Encoder Model")
    print("=" * 60)
    
    # Test parameters
    vocab_size = 200
    batch_size = 8
    seq_length = 20
    
    # Create model
    print("\n--- Creating Model ---\n")
    triplet_net, criterion = create_model(vocab_size)
    
    # Create dummy input
    print("\n--- Testing Forward Pass ---\n")
    
    anchor = torch.randint(0, vocab_size, (batch_size, seq_length))
    positive = torch.randint(0, vocab_size, (batch_size, seq_length))
    negative = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    print(f"Input shapes:")
    print(f"  Anchor:   {anchor.shape}")
    print(f"  Positive: {positive.shape}")
    print(f"  Negative: {negative.shape}")
    
    # Forward pass
    anchor_emb, positive_emb, negative_emb = triplet_net(anchor, positive, negative)
    
    print(f"\nOutput shapes:")
    print(f"  Anchor embedding:   {anchor_emb.shape}")
    print(f"  Positive embedding: {positive_emb.shape}")
    print(f"  Negative embedding: {negative_emb.shape}")
    
    # Check normalization
    print(f"\nEmbedding norms (should be ~1.0):")
    print(f"  Anchor norm:   {anchor_emb.norm(dim=1).mean().item():.4f}")
    print(f"  Positive norm: {positive_emb.norm(dim=1).mean().item():.4f}")
    print(f"  Negative norm: {negative_emb.norm(dim=1).mean().item():.4f}")
    
    # Calculate loss
    print("\n--- Testing Triplet Loss ---\n")
    
    loss = criterion(anchor_emb, positive_emb, negative_emb)
    print(f"Triplet loss: {loss.item():.4f}")
    
    # Test backward pass
    print("\n--- Testing Backward Pass ---\n")
    
    loss.backward()
    print("✓ Backward pass successful!")
    
    # Check gradients
    total_grad_norm = 0
    for p in triplet_net.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    print(f"Total gradient norm: {total_grad_norm:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ Model ready for training!")
    print("=" * 60)