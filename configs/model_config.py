"""
Model Configuration for Phonetic-Semantic Mapping
==================================================
This file contains all hyperparameters and settings for the model.

Why a config file?
- Easy to change settings without modifying code
- Keep track of experiments
- Makes your research reproducible
"""

# =============================================================
# DATA SETTINGS
# =============================================================

DATA_CONFIG = {
    # Paths
    "train_path": "data/processed/train_phonetic.csv",
    "val_path": "data/processed/val_phonetic.csv",
    "test_path": "data/processed/test_phonetic.csv",
    
    # Text column names
    "text_column": "phonetic",  # We use phonetic representation
    "label_column": "label",
    
    # Labels
    "labels": ["positive", "negative", "neutral"],
    "num_classes": 3,
}

# =============================================================
# MODEL SETTINGS
# =============================================================

MODEL_CONFIG = {
    # Vocabulary
    "max_vocab_size": 500,      # Maximum phoneme vocabulary size
    "max_seq_length": 50,       # Maximum sequence length (phonemes)
    
    # Embedding
    "embedding_dim": 64,        # Dimension of phoneme embeddings
    
    # Encoder architecture
    "hidden_dim": 128,          # Hidden layer dimension
    "output_dim": 64,           # Final embedding dimension
    "num_layers": 2,            # Number of LSTM layers
    "dropout": 0.3,             # Dropout rate
    
    # Model type
    "encoder_type": "lstm",     # Options: "lstm", "gru", "transformer"
}

# =============================================================
# TRAINING SETTINGS
# =============================================================

TRAINING_CONFIG = {
    # Basic training
    "batch_size": 16,           # Number of triplets per batch
    "num_epochs": 50,           # Training epochs
    "learning_rate": 0.001,     # Initial learning rate
    
    # Triplet loss
    "margin": 1.0,              # Margin for triplet loss
    
    # Optimization
    "optimizer": "adam",        # Options: "adam", "sgd"
    "weight_decay": 0.0001,     # L2 regularization
    
    # Learning rate scheduling
    "lr_scheduler": True,       # Use learning rate scheduler
    "lr_patience": 5,           # Patience for LR reduction
    "lr_factor": 0.5,           # Factor to reduce LR
    
    # Early stopping
    "early_stopping": True,     # Enable early stopping
    "es_patience": 10,          # Patience for early stopping
    
    # Checkpointing
    "save_best": True,          # Save best model
    "checkpoint_dir": "models/",
}

# =============================================================
# PROSODY SETTINGS (Your music background!)
# =============================================================

PROSODY_CONFIG = {
    # Enable prosody features
    "use_prosody": True,
    
    # Prosody feature dimensions
    "stress_dim": 8,            # Stress pattern features
    "rhythm_dim": 8,            # Rhythm pattern features
    "intonation_dim": 8,        # Intonation features
    
    # Combined prosody dimension
    "prosody_dim": 24,          # Total prosody features
    
    # Prosody weight in final embedding
    "prosody_weight": 0.3,      # How much prosody contributes
}

# =============================================================
# DEVICE SETTINGS
# =============================================================

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================
# PRINT CONFIGURATION
# =============================================================

def print_config():
    """Print all configuration settings."""
    print("=" * 60)
    print("MODEL CONFIGURATION")
    print("=" * 60)
    
    print("\n📁 DATA SETTINGS:")
    for key, value in DATA_CONFIG.items():
        print(f"   {key}: {value}")
    
    print("\n🧠 MODEL SETTINGS:")
    for key, value in MODEL_CONFIG.items():
        print(f"   {key}: {value}")
    
    print("\n🏋️ TRAINING SETTINGS:")
    for key, value in TRAINING_CONFIG.items():
        print(f"   {key}: {value}")
    
    print("\n🎵 PROSODY SETTINGS:")
    for key, value in PROSODY_CONFIG.items():
        print(f"   {key}: {value}")
    
    print(f"\n💻 DEVICE: {DEVICE}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()