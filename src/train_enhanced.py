"""
Enhanced Training Script with Prosody Features
===============================================
Trains the enhanced model that combines:
- Phonetic sequence encoding (LSTM)
- Prosody features (rhythm, stress, intensity)
- Attention mechanism

This should improve accuracy over the basic model!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import json
from datetime import datetime

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phonetic_vocabulary import PhoneticVocabulary
from src.enhanced_encoder import (
    EnhancedPhoneticEncoder, 
    EnhancedTripletNet, 
    EnhancedTripletDataset,
    create_enhanced_model
)
from src.phonetic_encoder import TripletLoss


class EnhancedTrainer:
    """
    Trainer for enhanced model with prosody features.
    """
    
    def __init__(self, config=None):
        """Initialize trainer."""
        
        self.config = config or {
            # Model
            "embedding_dim": 64,
            "hidden_dim": 128,
            "output_dim": 64,
            "prosody_dim": 24,
            "num_layers": 2,
            "dropout": 0.3,
            "use_attention": True,
            
            # Training
            "batch_size": 8,
            "num_epochs": 40,
            "learning_rate": 0.0005,  # Slightly lower for stability
            "weight_decay": 0.0001,
            "margin": 0.5,
            
            # Scheduler
            "lr_patience": 5,
            "lr_factor": 0.5,
            
            # Early stopping
            "early_stopping_patience": 12,
            
            # Data
            "max_seq_length": 25,
        }
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Components
        self.vocab = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rates": [],
        }
        
        # Best tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
        self.epochs_without_improvement = 0
        
        # Paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = os.path.join(self.base_dir, 'models')
        self.results_dir = os.path.join(self.base_dir, 'results')
    
    def setup(self):
        """Set up all components."""
        print("\n" + "=" * 60)
        print("Setting Up Enhanced Training")
        print("=" * 60)
        
        # 1. Load vocabulary
        print("\n1. Loading vocabulary...")
        vocab_path = os.path.join(self.model_dir, 'phonetic_vocab.json')
        self.vocab = PhoneticVocabulary()
        self.vocab.load(vocab_path)
        
        # 2. Create datasets with prosody
        print("\n2. Creating datasets with prosody features...")
        data_dir = os.path.join(self.base_dir, 'data', 'processed')
        
        train_dataset = EnhancedTripletDataset(
            os.path.join(data_dir, 'train_phonetic.csv'),
            self.vocab,
            self.config['max_seq_length']
        )
        
        val_dataset = EnhancedTripletDataset(
            os.path.join(data_dir, 'val_phonetic.csv'),
            self.vocab,
            self.config['max_seq_length']
        )
        
        test_dataset = EnhancedTripletDataset(
            os.path.join(data_dir, 'test_phonetic.csv'),
            self.vocab,
            self.config['max_seq_length']
        )
        
        # Create loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches:   {len(self.val_loader)}")
        print(f"   Test batches:  {len(self.test_loader)}")
        
        # 3. Create model
        print("\n3. Creating enhanced model with prosody...")
        self.model, self.criterion = create_enhanced_model(
            vocab_size=len(self.vocab),
            config=self.config
        )
        self.model = self.model.to(self.device)
        
        # 4. Optimizer
        print("\n4. Setting up optimizer...")
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 5. Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config['lr_patience'],
            factor=self.config['lr_factor']
        )
        
        print("\n✓ Enhanced setup complete!")
    
    def train_epoch(self):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch in progress:
            # Move to device
            anchor = batch['anchor'].to(self.device)
            positive = batch['positive'].to(self.device)
            negative = batch['negative'].to(self.device)
            anchor_prosody = batch['anchor_prosody'].to(self.device)
            positive_prosody = batch['positive_prosody'].to(self.device)
            negative_prosody = batch['negative_prosody'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward
            anchor_emb, positive_emb, negative_emb = self.model(
                anchor, positive, negative,
                anchor_prosody, positive_prosody, negative_prosody
            )
            
            # Loss
            loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Calculate accuracy
            pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb)
            neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb)
            correct += (pos_dist < neg_dist).sum().item()
            total += anchor.size(0)
            
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                anchor = batch['anchor'].to(self.device)
                positive = batch['positive'].to(self.device)
                negative = batch['negative'].to(self.device)
                anchor_prosody = batch['anchor_prosody'].to(self.device)
                positive_prosody = batch['positive_prosody'].to(self.device)
                negative_prosody = batch['negative_prosody'].to(self.device)
                
                anchor_emb, positive_emb, negative_emb = self.model(
                    anchor, positive, negative,
                    anchor_prosody, positive_prosody, negative_prosody
                )
                
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                total_loss += loss.item()
                
                pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb)
                neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb)
                correct += (pos_dist < neg_dist).sum().item()
                total += anchor.size(0)
        
        avg_loss = total_loss / max(len(self.val_loader), 1)
        accuracy = correct / max(total, 1)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'history': self.history,
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(self.model_dir, 'enhanced_checkpoint_latest.pt'))
        
        # Save best
        if is_best:
            torch.save(checkpoint, os.path.join(self.model_dir, 'enhanced_checkpoint_best.pt'))
            print(f"   💾 New best model! (Acc: {self.best_val_acc:.2%})")
    
    def train(self):
        """Run full training."""
        print("\n" + "=" * 60)
        print("🎵 Starting Enhanced Training with Prosody 🎵")
        print("=" * 60)
        
        print(f"\nConfiguration:")
        print(f"  Epochs: {self.config['num_epochs']}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Learning rate: {self.config['learning_rate']}")
        print(f"  Prosody dim: {self.config['prosody_dim']}")
        print(f"  Use attention: {self.config['use_attention']}")
        
        start_time = datetime.now()
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            print(f"\n{'─' * 60}")
            print(f"Epoch {epoch}/{self.config['num_epochs']}")
            print(f"{'─' * 60}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Get LR
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Print
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2%}")
            print(f"   LR: {current_lr:.6f}")
            
            # Check improvement (using accuracy)
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save
            self.save_checkpoint(epoch, is_best)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Early stopping
            if self.epochs_without_improvement >= self.config['early_stopping_patience']:
                print(f"\n⚠️ Early stopping at epoch {epoch}")
                break
        
        # Done
        duration = datetime.now() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Duration: {duration}")
        print(f"Best Val Accuracy: {self.best_val_acc:.2%}")
        
        # Evaluate
        self.evaluate()
        
        # Plot
        self.plot_history()
    
    def evaluate(self):
        """Evaluate on test set."""
        print("\n" + "=" * 60)
        print("Final Evaluation on Test Set")
        print("=" * 60)
        
        # Load best model
        best_path = os.path.join(self.model_dir, 'enhanced_checkpoint_best.pt')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded best model")
        
        # Evaluate
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                anchor = batch['anchor'].to(self.device)
                positive = batch['positive'].to(self.device)
                negative = batch['negative'].to(self.device)
                anchor_prosody = batch['anchor_prosody'].to(self.device)
                positive_prosody = batch['positive_prosody'].to(self.device)
                negative_prosody = batch['negative_prosody'].to(self.device)
                
                anchor_emb, positive_emb, negative_emb = self.model(
                    anchor, positive, negative,
                    anchor_prosody, positive_prosody, negative_prosody
                )
                
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                total_loss += loss.item()
                
                pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb)
                neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb)
                correct += (pos_dist < neg_dist).sum().item()
                total += anchor.size(0)
        
        test_loss = total_loss / max(len(self.test_loader), 1)
        test_acc = correct / max(total, 1)
        
        print(f"\n📊 Test Results:")
        print(f"   Test Loss:     {test_loss:.4f}")
        print(f"   Test Accuracy: {test_acc:.2%}")
        
        # Compare with basic model
        print(f"\n📈 Comparison:")
        print(f"   Basic Model:    50.00% (previous)")
        print(f"   Enhanced Model: {test_acc:.2%} (with prosody)")
        
        improvement = (test_acc - 0.5) * 100
        if improvement > 0:
            print(f"   Improvement:    +{improvement:.1f}% 🎉")
        
        # Save results
        results = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "best_val_accuracy": self.best_val_acc,
            "basic_model_accuracy": 0.5,
            "improvement": improvement,
            "config": self.config,
        }
        
        results_path = os.path.join(self.results_dir, 'enhanced_training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {results_path}")
    
    def plot_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, [a*100 for a in self.history['train_acc']], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, [a*100 for a in self.history['val_acc']], 'r-', label='Val', linewidth=2)
        axes[0, 1].axhline(y=50, color='gray', linestyle='--', label='Basic Model (50%)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training & Validation Accuracy', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 0].plot(epochs, self.history['learning_rates'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Comparison Bar Chart
        basic_acc = 50
        enhanced_acc = self.best_val_acc * 100
        
        bars = axes[1, 1].bar(['Basic Model', 'Enhanced Model\n(with Prosody)'], 
                              [basic_acc, enhanced_acc],
                              color=['#95a5a6', '#27ae60'])
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Model Comparison', fontweight='bold')
        axes[1, 1].set_ylim(0, 100)
        
        for bar, val in zip(bars, [basic_acc, enhanced_acc]):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                           f'{val:.1f}%', ha='center', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.results_dir, 'enhanced_training_history.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Plot saved to: {plot_path}")


def main():
    """Main function."""
    print("=" * 60)
    print("🎵 Enhanced Phonetic-Semantic Model Training 🎵")
    print("   (with Prosody Features)")
    print("=" * 60)
    
    # Config
    config = {
        # Model
        "embedding_dim": 64,
        "hidden_dim": 128,
        "output_dim": 64,
        "prosody_dim": 24,
        "num_layers": 2,
        "dropout": 0.3,
        "use_attention": True,
        
        # Training
        "batch_size": 8,
        "num_epochs": 40,
        "learning_rate": 0.0005,
        "weight_decay": 0.0001,
        "margin": 0.5,
        
        # Scheduler
        "lr_patience": 5,
        "lr_factor": 0.5,
        
        # Early stopping
        "early_stopping_patience": 12,
        
        # Data
        "max_seq_length": 25,
    }
    
    # Create trainer
    trainer = EnhancedTrainer(config)
    
    # Setup
    trainer.setup()
    
    # Train
    trainer.train()
    
    print("\n" + "=" * 60)
    print("🎉 Enhanced Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()