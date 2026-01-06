"""
Phase 5: Comprehensive Model Evaluation
Evaluates the enhanced phonetic-semantic model on test data
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import our modules
from phonetic_vocabulary import PhoneticVocabulary
from prosody_features import ProsodyExtractor
from enhanced_encoder import EnhancedPhoneticEncoder


class ModelEvaluator:
    """Comprehensive evaluation of the phonetic-semantic model"""
    
    def __init__(self, model_path, vocab_path, device='cpu'):
        """
        Initialize evaluator with trained model
        
        Args:
            model_path: Path to saved model checkpoint
            vocab_path: Path to vocabulary JSON
            device: 'cpu' or 'cuda'
        """
        self.device = device
        
        # Load vocabulary
        print("Loading vocabulary...")
        self.vocab = PhoneticVocabulary()
        self.vocab.load(vocab_path)
        print(f"  Vocabulary size: {len(self.vocab)}")
        
        # Initialize prosody extractor
        self.prosody_extractor = ProsodyExtractor()
        
        # Load model
        print("Loading model...")
        self.model = self._load_model(model_path)
        self.model.eval()
        print("  Model loaded successfully!")
        
        # Storage for analysis
        self.embeddings = []
        self.labels = []
        self.texts = []
        self.predictions = []
    
    def _load_model(self, model_path):
        """Load the trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model config from checkpoint or use defaults
        config = checkpoint.get('config', {
            'vocab_size': len(self.vocab),
            'embedding_dim': 64,
            'hidden_dim': 128,
            'output_dim': 64,
            'prosody_dim': 24,
            'num_layers': 2,
            'dropout': 0.3,
            'use_attention': True
        })
        
        # Initialize model
        model = EnhancedPhoneticEncoder(
            vocab_size=len(self.vocab),
            embedding_dim=config.get('embedding_dim', 64),
            hidden_dim=config.get('hidden_dim', 128),
            output_dim=config.get('output_dim', 64),
            prosody_dim=config.get('prosody_dim', 24),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.3),
            use_attention=config.get('use_attention', True)
        )
        
        # Get state dict from checkpoint
        state_dict = checkpoint['model_state_dict']
        
        # Fix key mismatch: remove 'encoder.' prefix if present
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key[8:]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # Load the fixed state dict
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        
        return model
    
    def encode_text(self, text, phonetic):
        """
        Encode a single text sample
        
        Args:
            text: Original text
            phonetic: Phonetic representation
            
        Returns:
            embedding: 64-dim normalized embedding
        """
        # Tokenize phonetic
        tokens = self.vocab.encode(phonetic, max_length=25)
        tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        # Extract prosody features
        prosody = self.prosody_extractor.extract(phonetic)
        prosody_tensor = torch.tensor([prosody], dtype=torch.float32).to(self.device)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.model(tokens_tensor, prosody_tensor)
        
        return embedding.cpu().numpy()[0]
    
    def evaluate_dataset(self, data_path):
        """
        Evaluate model on a dataset
        
        Args:
            data_path: Path to CSV with text, label, phonetic columns
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        print(f"\nEvaluating on: {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"  Samples: {len(df)}")
        
        # Reset storage
        self.embeddings = []
        self.labels = []
        self.texts = []
        
        # Encode all samples
        print("  Encoding samples...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Encoding"):
            embedding = self.encode_text(row['text'], row['phonetic'])
            self.embeddings.append(embedding)
            self.labels.append(row['label'])
            self.texts.append(row['text'])
        
        self.embeddings = np.array(self.embeddings)
        
        # Perform KNN classification
        print("  Performing KNN classification...")
        metrics = self._knn_evaluate()
        
        return metrics
    
    def _knn_evaluate(self, k=3):
        """
        Evaluate using K-Nearest Neighbors
        """
        label_to_idx = {'positive': 0, 'negative': 1, 'neutral': 2}
        idx_to_label = {0: 'positive', 1: 'negative', 2: 'neutral'}
        
        y_true = [label_to_idx[l] for l in self.labels]
        y_pred = []
        
        # For each sample, find k nearest neighbors
        for i in range(len(self.embeddings)):
            distances = []
            for j in range(len(self.embeddings)):
                if i != j:
                    dist = np.linalg.norm(self.embeddings[i] - self.embeddings[j])
                    distances.append((dist, y_true[j]))
            
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:k]
            
            votes = defaultdict(int)
            for dist, label in k_nearest:
                votes[label] += 1
            
            predicted = max(votes.keys(), key=lambda x: votes[x])
            y_pred.append(predicted)
        
        self.predictions = [idx_to_label[p] for p in y_pred]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        class_report = classification_report(
            y_true, y_pred, 
            target_names=['positive', 'negative', 'neutral'],
            output_dict=True
        )
        
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'class_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': list(zip(self.texts, self.labels, self.predictions))
        }
        
        return metrics
    
    def analyze_errors(self):
        """Analyze misclassified samples"""
        errors = []
        
        for text, true_label, pred_label in zip(self.texts, self.labels, self.predictions):
            if true_label != pred_label:
                errors.append({
                    'text': text,
                    'true_label': true_label,
                    'predicted_label': pred_label
                })
        
        return errors
    
    def get_embedding_stats(self):
        """Compute statistics about embeddings by class"""
        stats = {}
        
        for label in ['positive', 'negative', 'neutral']:
            mask = [l == label for l in self.labels]
            class_embeddings = self.embeddings[mask]
            
            if len(class_embeddings) > 0:
                centroid = np.mean(class_embeddings, axis=0)
                variance = np.mean(np.var(class_embeddings, axis=0))
                distances = [np.linalg.norm(e - centroid) for e in class_embeddings]
                avg_spread = np.mean(distances)
                
                stats[label] = {
                    'count': len(class_embeddings),
                    'centroid': centroid,
                    'variance': float(variance),
                    'avg_spread': float(avg_spread)
                }
        
        labels = list(stats.keys())
        for i, l1 in enumerate(labels):
            for l2 in labels[i+1:]:
                dist = np.linalg.norm(stats[l1]['centroid'] - stats[l2]['centroid'])
                stats[f'distance_{l1}_{l2}'] = float(dist)
        
        return stats


def visualize_embeddings(evaluator, save_path):
    """Create t-SNE visualization of embeddings"""
    print("\nCreating embedding visualization...")
    
    perplexity = min(5, len(evaluator.embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(evaluator.embeddings)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    color_map = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}
    
    # Plot 1: True labels
    ax1 = axes[0]
    for label in ['positive', 'negative', 'neutral']:
        mask = [l == label for l in evaluator.labels]
        mask_indices = [i for i, m in enumerate(mask) if m]
        if mask_indices:
            ax1.scatter(
                embeddings_2d[mask_indices, 0], 
                embeddings_2d[mask_indices, 1],
                c=color_map[label],
                label=label.capitalize(),
                alpha=0.7,
                s=100
            )
    ax1.set_title('t-SNE: True Labels', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predictions with errors
    ax2 = axes[1]
    
    for i in range(len(evaluator.embeddings)):
        true_label = evaluator.labels[i]
        pred_label = evaluator.predictions[i]
        is_correct = true_label == pred_label
        
        marker = 'o' if is_correct else 'x'
        size = 100 if is_correct else 150
        
        ax2.scatter(
            embeddings_2d[i, 0],
            embeddings_2d[i, 1],
            c=color_map[pred_label],
            marker=marker,
            s=size,
            alpha=0.7,
            linewidths=3 if not is_correct else 1
        )
    
    ax2.set_title('t-SNE: Predictions (X = Errors)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    ax2.grid(True, alpha=0.3)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=10, label='Positive'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='Negative'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=10, label='Neutral'),
        Line2D([0], [0], marker='x', color='black', markersize=10, label='Error', linestyle='None')
    ]
    ax2.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def visualize_confusion_matrix(metrics, save_path):
    """Create confusion matrix heatmap"""
    print("Creating confusion matrix visualization...")
    
    conf_matrix = np.array(metrics['confusion_matrix'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Positive', 'Negative', 'Neutral'],
        yticklabels=['Positive', 'Negative', 'Neutral'],
        ax=ax
    )
    
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def visualize_class_separation(evaluator, save_path):
    """Visualize how well classes are separated"""
    print("Creating class separation visualization...")
    
    stats = evaluator.get_embedding_stats()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Intra-class spread
    ax1 = axes[0]
    labels = ['positive', 'negative', 'neutral']
    spreads = [stats[l]['avg_spread'] for l in labels]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    bars = ax1.bar(labels, spreads, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Intra-Class Spread\n(lower = more compact clusters)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Distance from Centroid')
    ax1.set_xlabel('Sentiment Class')
    
    for bar, spread in zip(bars, spreads):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{spread:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Inter-class distances
    ax2 = axes[1]
    pairs = ['distance_positive_negative', 'distance_positive_neutral', 'distance_negative_neutral']
    pair_labels = ['Pos-Neg', 'Pos-Neu', 'Neg-Neu']
    distances = [stats[p] for p in pairs]
    
    bars = ax2.bar(pair_labels, distances, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax2.set_title('Inter-Class Distances\n(higher = better separation)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Distance Between Centroids')
    ax2.set_xlabel('Class Pairs')
    
    for bar, dist in zip(bars, distances):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{dist:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def print_evaluation_report(metrics, errors):
    """Print a formatted evaluation report"""
    print("\n" + "="*60)
    print("           EVALUATION REPORT")
    print("="*60)
    
    print(f"\n📊 OVERALL METRICS:")
    print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"   Precision: {metrics['precision']*100:.2f}%")
    print(f"   Recall:    {metrics['recall']*100:.2f}%")
    print(f"   F1-Score:  {metrics['f1_score']*100:.2f}%")
    
    print(f"\n📈 PER-CLASS METRICS:")
    for label in ['positive', 'negative', 'neutral']:
        report = metrics['class_report'][label]
        print(f"\n   {label.upper()}:")
        print(f"      Precision: {report['precision']*100:.2f}%")
        print(f"      Recall:    {report['recall']*100:.2f}%")
        print(f"      F1-Score:  {report['f1-score']*100:.2f}%")
        print(f"      Support:   {report['support']}")
    
    print(f"\n❌ ERRORS ({len(errors)} misclassified):")
    for i, error in enumerate(errors[:10], 1):
        text = error['text']
        if len(text) > 50:
            text = text[:50] + "..."
        print(f"\n   {i}. \"{text}\"")
        print(f"      True: {error['true_label']} → Predicted: {error['predicted_label']}")
    
    if len(errors) > 10:
        print(f"\n   ... and {len(errors)-10} more errors")
    
    print("\n" + "="*60)


def main():
    """Run comprehensive evaluation"""
    print("="*60)
    print("   PHASE 5: MODEL EVALUATION")
    print("="*60)
    
    # Paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_dir, 'models', 'enhanced_checkpoint_best.pt')
    vocab_path = os.path.join(project_dir, 'models', 'phonetic_vocab.json')
    test_path = os.path.join(project_dir, 'data', 'processed', 'test_phonetic.csv')
    results_dir = os.path.join(project_dir, 'results')
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Check files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    if not os.path.exists(vocab_path):
        print(f"❌ Vocabulary not found: {vocab_path}")
        return
    if not os.path.exists(test_path):
        print(f"❌ Test data not found: {test_path}")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path, vocab_path)
    
    # Evaluate on test set
    metrics = evaluator.evaluate_dataset(test_path)
    
    # Analyze errors
    errors = evaluator.analyze_errors()
    
    # Print report
    print_evaluation_report(metrics, errors)
    
    # Create visualizations
    visualize_embeddings(
        evaluator, 
        os.path.join(results_dir, 'embedding_tsne.png')
    )
    
    visualize_confusion_matrix(
        metrics, 
        os.path.join(results_dir, 'confusion_matrix.png')
    )
    
    visualize_class_separation(
        evaluator, 
        os.path.join(results_dir, 'class_separation.png')
    )
    
    # Save detailed results
    embedding_stats = evaluator.get_embedding_stats()
    
    clean_stats = {}
    for k, v in embedding_stats.items():
        if isinstance(v, dict):
            clean_stats[k] = {
                kk: float(vv) if isinstance(vv, (float, np.floating)) else vv 
                for kk, vv in v.items() 
                if kk != 'centroid'
            }
        else:
            clean_stats[k] = float(v) if isinstance(v, (float, np.floating)) else v
    
    results = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1_score': float(metrics['f1_score']),
        'confusion_matrix': metrics['confusion_matrix'],
        'class_report': {
            k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv for kk, vv in v.items()}
            for k, v in metrics['class_report'].items() 
            if k in ['positive', 'negative', 'neutral']
        },
        'errors': errors,
        'embedding_stats': clean_stats
    }
    
    results_path = os.path.join(results_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Saved detailed results: {results_path}")
    
    print("\n✅ Evaluation complete!")
    print(f"\n📊 Generated visualizations:")
    print(f"   - results/embedding_tsne.png")
    print(f"   - results/confusion_matrix.png")
    print(f"   - results/class_separation.png")


if __name__ == "__main__":
    main()