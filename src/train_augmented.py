"""
Training with Augmented Data
=============================
Trains the enhanced model using the augmented dataset.

Goal: Improve accuracy beyond 60% by using more training data!
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phonetic_vocabulary import PhoneticVocabulary, build_vocabulary_from_dataset
from src.enhanced_encoder import (
    EnhancedPhoneticEncoder,
    EnhancedTripletNet,
    EnhancedTripletDataset,
    create_enhanced_model
)
from src.phonetic_encoder import TripletLoss


class AugmentedTrainer:
    """
    Trainer using augmented dataset.
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
            
            # Training - adjusted for larger dataset
            "batch_size": 16,        # Larger batch now
            "num_epochs": 50,        # More epochs
            "learning_rate": 0.0003, # Slightly lower LR
            "weight_decay": 0.0001,
            "margin": 0.5,
            
            # Scheduler
            "lr_patience": 7,
            "lr_factor": 0.5,
            
            # Early stopping
            "early_stopping_patience": 15,
            
            # Data
            "max_seq_length": 25,
        }
        
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
        self.data_dir = os.path.join(self.base_dir, 'data', 'processed')
    
    def setup(self):
        """Set up all components."""
        print("\n" + "=" * 60)
        print("Setting Up Training with Augmented Data")
        print("=" * 60)
        
        # 1. Rebuild vocabulary including augmented data
        print("\n1. Building vocabulary from augmented data...")
        
        import pandas as pd
        
        # Load all data including augmented
        train_aug_path = os.path.join(self.data_dir, 'train_augmented.csv')
        val_path = os.path.join(self.data_dir, 'val_phonetic.csv')
        test_path = os.path.join(self.data_dir, 'test_phonetic.csv')
        
        train_df = pd.read_csv(train_aug_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        print(f"   Training samples (augmented): {len(train_df)}")
        print(f"   Validation samples: {len(val_df)}")
        print(f"   Test samples: {len(test_df)}")
        
        # Build vocabulary from all phonetic texts
        all_texts = list(train_df['phonetic']) + list(val_df['phonetic']) + list(test_df['phonetic'])
        
        self.vocab = PhoneticVocabulary(max_vocab_size=1000)  # Larger vocab for augmented data
        self.vocab.build_vocab(all_texts)
        
        # Save updated vocabulary
        vocab_path = os.path.join(self.model_dir, 'phonetic_vocab_augmented.json')
        self.vocab.save(vocab_path)
        
        # 2. Create datasets
        print("\n2. Creating datasets with prosody features...")
        
        train_dataset = EnhancedTripletDataset(
            train_aug_path,
            self.vocab,
            self.config['max_seq_length']
        )
        
        val_dataset = EnhancedTripletDataset(
            val_path,
            self.vocab,
            self.config['max_seq_length']
        )
        
        test_dataset = EnhancedTripletDataset(
            test_path,
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
        print(f"   Val batches: {len(self.val_loader)}")
        print(f"   Test batches: {len(self.test_loader)}")
        
        # 3. Create model
        print("\n3. Creating enhanced model...")
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
        
        print("\n✓ Setup complete!")
    
    def train_epoch(self):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch in progress:
            anchor = batch['anchor'].to(self.device)
            positive = batch['positive'].to(self.device)
            negative = batch['negative'].to(self.device)
            anchor_prosody = batch['anchor_prosody'].to(self.device)
            positive_prosody = batch['positive_prosody'].to(self.device)
            negative_prosody = batch['negative_prosody'].to(self.device)
            
            self.optimizer.zero_grad()
            
            anchor_emb, positive_emb, negative_emb = self.model(
                anchor, positive, negative,
                anchor_prosody, positive_prosody, negative_prosody
            )
            
            loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb)
            neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb)
            correct += (pos_dist < neg_dist).sum().item()
            total += anchor.size(0)
            
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / len(self.train_loader), correct / total
    
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
        
        return total_loss / max(len(self.val_loader), 1), correct / max(total, 1)
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'history': self.history,
        }
        
        torch.save(checkpoint, os.path.join(self.model_dir, 'augmented_checkpoint_latest.pt'))
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.model_dir, 'augmented_checkpoint_best.pt'))
            print(f"   💾 New best model! (Acc: {self.best_val_acc:.2%})")
    
    def train(self):
        """Run full training."""
        print("\n" + "=" * 60)
        print("🚀 Training with Augmented Data 🚀")
        print("=" * 60)
        
        print(f"\nConfiguration:")
        print(f"  Epochs: {self.config['num_epochs']}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Learning rate: {self.config['learning_rate']}")
        
        start_time = datetime.now()
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            print(f"\n{'─' * 60}")
            print(f"Epoch {epoch}/{self.config['num_epochs']}")
            print(f"{'─' * 60}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2%}")
            print(f"   LR: {current_lr:.6f}")
            
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            self.save_checkpoint(epoch, is_best)
            self.scheduler.step(val_loss)
            
            if self.epochs_without_improvement >= self.config['early_stopping_patience']:
                print(f"\n⚠️ Early stopping at epoch {epoch}")
                break
        
        duration = datetime.now() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Duration: {duration}")
        print(f"Best Val Accuracy: {self.best_val_acc:.2%}")
        
        self.evaluate()
        self.plot_history()
    
    def evaluate(self):
        """Evaluate on test set."""
        print("\n" + "=" * 60)
        print("Final Evaluation on Test Set")
        print("=" * 60)
        
        best_path = os.path.join(self.model_dir, 'augmented_checkpoint_best.pt')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded best model")
        
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
        
        print(f"\n📈 Model Progression:")
        print(f"   Basic Model (no prosody):      50.00%")
        print(f"   Enhanced Model (+ prosody):    60.00%")
        print(f"   Augmented Model (+ more data): {test_acc:.2%}")
        
        total_improvement = (test_acc - 0.5) * 100
        print(f"\n   Total Improvement: +{total_improvement:.1f}% from baseline! 🎉")
        
        # Save results
        results = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "best_val_accuracy": self.best_val_acc,
            "baseline_accuracy": 0.5,
            "enhanced_accuracy": 0.6,
            "total_improvement": total_improvement,
            "config": self.config,
        }
        
        results_path = os.path.join(self.results_dir, 'augmented_training_results.json')
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
        axes[0, 0].set_title('Loss (Augmented Training)', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, [a*100 for a in self.history['train_acc']], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, [a*100 for a in self.history['val_acc']], 'r-', label='Val', linewidth=2)
        axes[0, 1].axhline(y=50, color='gray', linestyle='--', label='Basic (50%)')
        axes[0, 1].axhline(y=60, color='orange', linestyle='--', label='Enhanced (60%)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy (Augmented Training)', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 0].plot(epochs, self.history['learning_rates'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Model Comparison
        models = ['Basic\n(50%)', 'Enhanced\n(+Prosody)', 'Augmented\n(+Data)']
        accuracies = [50, 60, self.best_val_acc * 100]
        colors = ['#95a5a6', '#3498db', '#27ae60']
        
        bars = axes[1, 1].bar(models, accuracies, color=colors)
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Model Progression', fontweight='bold')
        axes[1, 1].set_ylim(0, 100)
        
        for bar, val in zip(bars, accuracies):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                           f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.results_dir, 'augmented_training_history.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Plot saved to: {plot_path}")


def main():
    """Main function."""
    print("=" * 60)
    print("🚀 Training with Augmented Data 🚀")
    print("=" * 60)
    
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
        "batch_size": 16,
        "num_epochs": 50,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "margin": 0.5,
        
        # Scheduler
        "lr_patience": 7,
        "lr_factor": 0.5,
        
        # Early stopping
        "early_stopping_patience": 15,
        
        # Data
        "max_seq_length": 25,
    }
    
    trainer = AugmentedTrainer(config)
    trainer.setup()
    trainer.train()
    
    print("\n" + "=" * 60)
    print("🎉 Augmented Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()