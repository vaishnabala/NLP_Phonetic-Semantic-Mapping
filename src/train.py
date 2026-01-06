"""
Training Script for Phonetic-Semantic Model
=============================================
This script handles the complete training pipeline:
1. Load data and create DataLoaders
2. Initialize model and optimizer
3. Train with triplet loss
4. Validate and save best model
5. Track metrics and visualize progress

Usage:
    python src/train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import json
from datetime import datetime

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phonetic_vocabulary import PhoneticVocabulary
from src.triplet_dataset import TripletDataset, create_data_loaders
from src.phonetic_encoder import PhoneticEncoder, TripletNet, TripletLoss, create_model


class Trainer:
    """
    Handles the complete training process for the phonetic encoder.
    """
    
    def __init__(self, config=None):
        """
        Initialize trainer with configuration.
        
        Args:
            config (dict): Training configuration
        """
        # Default configuration
        self.config = config or {
            # Model
            "embedding_dim": 64,
            "hidden_dim": 128,
            "output_dim": 64,
            "num_layers": 2,
            "dropout": 0.3,
            
            # Training
            "batch_size": 16,
            "num_epochs": 50,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "margin": 1.0,
            
            # Scheduler
            "lr_patience": 5,
            "lr_factor": 0.5,
            
            # Early stopping
            "early_stopping_patience": 10,
            
            # Data
            "max_seq_length": 30,
        }
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize components (will be set up in setup())
        self.vocab = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = os.path.join(self.base_dir, 'models')
        self.results_dir = os.path.join(self.base_dir, 'results')
        
    def setup(self):
        """Set up all components for training."""
        print("\n" + "=" * 60)
        print("Setting Up Training")
        print("=" * 60)
        
        # 1. Load vocabulary
        print("\n1. Loading vocabulary...")
        vocab_path = os.path.join(self.model_dir, 'phonetic_vocab.json')
        self.vocab = PhoneticVocabulary()
        self.vocab.load(vocab_path)
        
        # 2. Create data loaders
        print("\n2. Creating data loaders...")
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            self.vocab,
            batch_size=self.config["batch_size"],
            max_length=self.config["max_seq_length"]
        )
        
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches:   {len(self.val_loader)}")
        print(f"   Test batches:  {len(self.test_loader)}")
        
        # 3. Create model
        print("\n3. Creating model...")
        self.model, self.criterion = create_model(
            vocab_size=len(self.vocab),
            config=self.config
        )
        self.model = self.model.to(self.device)
        
        # 4. Create optimizer
        print("\n4. Setting up optimizer...")
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
        
        # 5. Create learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config["lr_patience"],
            factor=self.config["lr_factor"]
        )
        
        print("\n✓ Setup complete!")
        
    def train_epoch(self):
        """
        Train for one epoch.
        
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch in progress_bar:
            # Move to device
            anchor = batch['anchor'].to(self.device)
            positive = batch['positive'].to(self.device)
            negative = batch['negative'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            anchor_emb, positive_emb, negative_emb = self.model(anchor, positive, negative)
            
            # Calculate loss
            loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / num_batches
    
    def validate(self):
        """
        Validate the model.
        
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                anchor = batch['anchor'].to(self.device)
                positive = batch['positive'].to(self.device)
                negative = batch['negative'].to(self.device)
                
                # Forward pass
                anchor_emb, positive_emb, negative_emb = self.model(anchor, positive, negative)
                
                # Calculate loss
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def calculate_accuracy(self, loader):
        """
        Calculate triplet accuracy.
        
        Accuracy = % of triplets where d(anchor, positive) < d(anchor, negative)
        
        Args:
            loader: DataLoader to evaluate
            
        Returns:
            float: Accuracy (0-1)
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
                anchor = batch['anchor'].to(self.device)
                positive = batch['positive'].to(self.device)
                negative = batch['negative'].to(self.device)
                
                # Get embeddings
                anchor_emb, positive_emb, negative_emb = self.model(anchor, positive, negative)
                
                # Calculate distances
                pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb)
                neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb)
                
                # Count correct (positive closer than negative)
                correct += (pos_dist < neg_dist).sum().item()
                total += anchor.size(0)
        
        return correct / max(total, 1)
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history,
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.model_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.model_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            print(f"   💾 Saved best model (loss: {self.best_val_loss:.4f})")
    
    def train(self):
        """
        Run the complete training loop.
        """
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Epochs: {self.config['num_epochs']}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Learning rate: {self.config['learning_rate']}")
        print(f"  Margin: {self.config['margin']}")
        print(f"  Device: {self.device}")
        
        start_time = datetime.now()
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            print(f"\n{'─' * 60}")
            print(f"Epoch {epoch}/{self.config['num_epochs']}")
            print(f"{'─' * 60}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Calculate accuracy
            train_acc = self.calculate_accuracy(self.train_loader)
            val_acc = self.calculate_accuracy(self.val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(current_lr)
            
            # Print metrics
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2%}")
            print(f"   LR: {current_lr:.6f}")
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Early stopping
            if self.epochs_without_improvement >= self.config['early_stopping_patience']:
                print(f"\n⚠️ Early stopping triggered after {epoch} epochs")
                break
        
        # Training complete
        training_time = datetime.now() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total time: {training_time}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Final evaluation
        self.evaluate()
        
        # Plot training history
        self.plot_history()
        
    def evaluate(self):
        """Evaluate on test set."""
        print("\n" + "=" * 60)
        print("Final Evaluation on Test Set")
        print("=" * 60)
        
        # Load best model
        best_path = os.path.join(self.model_dir, 'checkpoint_best.pt')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded best model checkpoint")
        
        # Calculate metrics
        test_loss = 0
        num_batches = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                anchor = batch['anchor'].to(self.device)
                positive = batch['positive'].to(self.device)
                negative = batch['negative'].to(self.device)
                
                anchor_emb, positive_emb, negative_emb = self.model(anchor, positive, negative)
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                
                test_loss += loss.item()
                num_batches += 1
        
        test_loss = test_loss / max(num_batches, 1)
        test_acc = self.calculate_accuracy(self.test_loader)
        
        print(f"\nTest Loss:     {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2%}")
        
        # Save results
        results = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "history": self.history,
        }
        
        results_path = os.path.join(self.results_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {results_path}")
    
    def plot_history(self):
        """Plot and save training history."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Plot 1: Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Learning Rate
        axes[1].plot(epochs, self.history['learning_rates'], 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Learning Rate', fontsize=12)
        axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Training plot saved to: {plot_path}")


def main():
    """Main training function."""
    print("=" * 60)
    print("Phonetic-Semantic Model Training")
    print("=" * 60)
    
    # Configuration
    config = {
        # Model architecture
        "embedding_dim": 64,
        "hidden_dim": 128,
        "output_dim": 64,
        "num_layers": 2,
        "dropout": 0.3,
        
        # Training hyperparameters
        "batch_size": 8,        # Small batch for our small dataset
        "num_epochs": 30,       # Enough to see learning
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "margin": 0.5,          # Triplet loss margin
        
        # Learning rate scheduler
        "lr_patience": 5,
        "lr_factor": 0.5,
        
        # Early stopping
        "early_stopping_patience": 10,
        
        # Data
        "max_seq_length": 25,
    }
    
    # Create trainer
    trainer = Trainer(config)
    
    # Setup
    trainer.setup()
    
    # Train
    trainer.train()
    
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()