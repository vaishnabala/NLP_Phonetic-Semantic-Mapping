"""
Phase 5.2: Baseline Comparison
Compare our phonetic model against mBERT and simple baselines
"""

import torch
import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Check if transformers is available
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers not installed. Will skip mBERT baseline.")


class BaselineEvaluator:
    """Evaluate various baseline models"""
    
    def __init__(self, train_path, test_path):
        """Load train and test data"""
        print("Loading data...")
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"  Train samples: {len(self.train_df)}")
        print(f"  Test samples: {len(self.test_df)}")
        
        # Prepare labels
        self.label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        self.idx_to_label = {0: 'positive', 1: 'negative', 2: 'neutral'}
        
        self.y_train = [self.label_map[l] for l in self.train_df['label']]
        self.y_test = [self.label_map[l] for l in self.test_df['label']]
        
        self.results = {}
    
    def evaluate_random_baseline(self):
        """Random guessing baseline"""
        print("\n" + "="*50)
        print("📊 BASELINE 1: Random Guessing")
        print("="*50)
        
        np.random.seed(42)
        y_pred = np.random.randint(0, 3, size=len(self.y_test))
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"   Accuracy: {accuracy*100:.2f}%")
        print(f"   (Expected: ~33.33% for 3 classes)")
        
        self.results['random'] = {
            'accuracy': accuracy,
            'description': 'Random guessing (baseline)'
        }
        
        return accuracy
    
    def evaluate_majority_baseline(self):
        """Always predict most common class"""
        print("\n" + "="*50)
        print("📊 BASELINE 2: Majority Class")
        print("="*50)
        
        # Find most common class in training
        from collections import Counter
        majority_class = Counter(self.y_train).most_common(1)[0][0]
        majority_label = self.idx_to_label[majority_class]
        
        y_pred = [majority_class] * len(self.y_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"   Majority class: {majority_label}")
        print(f"   Accuracy: {accuracy*100:.2f}%")
        
        self.results['majority'] = {
            'accuracy': accuracy,
            'description': f'Always predict {majority_label}'
        }
        
        return accuracy
    
    def evaluate_tfidf_logreg(self):
        """TF-IDF + Logistic Regression"""
        print("\n" + "="*50)
        print("📊 BASELINE 3: TF-IDF + Logistic Regression")
        print("="*50)
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(self.train_df['text'])
        X_test = vectorizer.transform(self.test_df['text'])
        
        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, self.y_train)
        
        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"   Accuracy: {accuracy*100:.2f}%")
        
        # Per-class report
        report = classification_report(
            self.y_test, y_pred,
            target_names=['positive', 'negative', 'neutral'],
            output_dict=True
        )
        
        print(f"\n   Per-class F1:")
        for label in ['positive', 'negative', 'neutral']:
            print(f"      {label}: {report[label]['f1-score']*100:.2f}%")
        
        self.results['tfidf_logreg'] = {
            'accuracy': accuracy,
            'report': {k: v for k, v in report.items() if k in ['positive', 'negative', 'neutral']},
            'description': 'TF-IDF + Logistic Regression'
        }
        
        return accuracy
    
    def evaluate_tfidf_svm(self):
        """TF-IDF + SVM"""
        print("\n" + "="*50)
        print("📊 BASELINE 4: TF-IDF + SVM")
        print("="*50)
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(self.train_df['text'])
        X_test = vectorizer.transform(self.test_df['text'])
        
        # Train SVM
        clf = SVC(kernel='rbf', random_state=42)
        clf.fit(X_train, self.y_train)
        
        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"   Accuracy: {accuracy*100:.2f}%")
        
        report = classification_report(
            self.y_test, y_pred,
            target_names=['positive', 'negative', 'neutral'],
            output_dict=True
        )
        
        print(f"\n   Per-class F1:")
        for label in ['positive', 'negative', 'neutral']:
            print(f"      {label}: {report[label]['f1-score']*100:.2f}%")
        
        self.results['tfidf_svm'] = {
            'accuracy': accuracy,
            'report': {k: v for k, v in report.items() if k in ['positive', 'negative', 'neutral']},
            'description': 'TF-IDF + SVM'
        }
        
        return accuracy
    
    def evaluate_mbert(self):
        """Multilingual BERT baseline"""
        if not TRANSFORMERS_AVAILABLE:
            print("\n⚠️ Skipping mBERT (transformers not installed)")
            return None
        
        print("\n" + "="*50)
        print("📊 BASELINE 5: mBERT Embeddings + LogReg")
        print("="*50)
        print("   Loading mBERT model (this may take a minute)...")
        
        try:
            # Load mBERT
            tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
            model = AutoModel.from_pretrained('bert-base-multilingual-cased')
            model.eval()
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            def get_embedding(text):
                """Get CLS token embedding from mBERT"""
                inputs = tokenizer(
                    text, 
                    return_tensors='pt', 
                    truncation=True, 
                    max_length=128,
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use CLS token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                return embedding[0]
            
            # Get embeddings for train and test
            print("   Encoding train samples...")
            X_train = []
            for text in tqdm(self.train_df['text'], desc="   Train"):
                X_train.append(get_embedding(text))
            X_train = np.array(X_train)
            
            print("   Encoding test samples...")
            X_test = []
            for text in tqdm(self.test_df['text'], desc="   Test"):
                X_test.append(get_embedding(text))
            X_test = np.array(X_test)
            
            # Train classifier
            print("   Training classifier...")
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train, self.y_train)
            
            y_pred = clf.predict(X_test)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"\n   Accuracy: {accuracy*100:.2f}%")
            
            report = classification_report(
                self.y_test, y_pred,
                target_names=['positive', 'negative', 'neutral'],
                output_dict=True
            )
            
            print(f"\n   Per-class F1:")
            for label in ['positive', 'negative', 'neutral']:
                print(f"      {label}: {report[label]['f1-score']*100:.2f}%")
            
            self.results['mbert'] = {
                'accuracy': accuracy,
                'report': {k: v for k, v in report.items() if k in ['positive', 'negative', 'neutral']},
                'description': 'mBERT embeddings + Logistic Regression'
            }
            
            return accuracy
            
        except Exception as e:
            print(f"   ❌ Error loading mBERT: {e}")
            return None
    
    def add_our_model_results(self, accuracy):
        """Add our phonetic model results for comparison"""
        self.results['phonetic_prosody'] = {
            'accuracy': accuracy,
            'description': 'Our model: Phonetic + Prosody + Contrastive Learning'
        }
    
    def create_comparison_chart(self, save_path):
        """Create bar chart comparing all models"""
        print("\nCreating comparison chart...")
        
        # Prepare data
        models = []
        accuracies = []
        colors = []
        
        color_map = {
            'random': '#95a5a6',
            'majority': '#95a5a6',
            'tfidf_logreg': '#3498db',
            'tfidf_svm': '#3498db',
            'mbert': '#9b59b6',
            'phonetic_prosody': '#e74c3c'
        }
        
        label_map = {
            'random': 'Random',
            'majority': 'Majority',
            'tfidf_logreg': 'TF-IDF+LR',
            'tfidf_svm': 'TF-IDF+SVM',
            'mbert': 'mBERT',
            'phonetic_prosody': 'Ours\n(Phonetic)'
        }
        
        for name in ['random', 'majority', 'tfidf_logreg', 'tfidf_svm', 'mbert', 'phonetic_prosody']:
            if name in self.results:
                models.append(label_map[name])
                accuracies.append(self.results[name]['accuracy'] * 100)
                colors.append(color_map[name])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 1,
                f'{acc:.1f}%',
                ha='center', 
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )
        
        # Styling
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_title('Model Comparison: Code-Mixed Sentiment Analysis', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.axhline(y=33.33, color='gray', linestyle='--', alpha=0.5, label='Random chance')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: {save_path}")
    
    def print_summary(self):
        """Print summary of all results"""
        print("\n" + "="*60)
        print("           BASELINE COMPARISON SUMMARY")
        print("="*60)
        
        # Sort by accuracy
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )
        
        print(f"\n{'Rank':<6}{'Model':<30}{'Accuracy':<12}")
        print("-" * 48)
        
        for rank, (name, data) in enumerate(sorted_results, 1):
            marker = "⭐" if name == 'phonetic_prosody' else "  "
            print(f"{rank:<6}{data['description'][:28]:<30}{data['accuracy']*100:>6.2f}% {marker}")
        
        print("\n" + "="*60)


def main():
    """Run baseline comparison"""
    print("="*60)
    print("   PHASE 5.2: BASELINE COMPARISON")
    print("="*60)
    
    # Paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(project_dir, 'data', 'processed', 'train_phonetic.csv')
    test_path = os.path.join(project_dir, 'data', 'processed', 'test_phonetic.csv')
    results_dir = os.path.join(project_dir, 'results')
    
    # Check files
    if not os.path.exists(train_path):
        print(f"❌ Train data not found: {train_path}")
        return
    if not os.path.exists(test_path):
        print(f"❌ Test data not found: {test_path}")
        return
    
    # Initialize evaluator
    evaluator = BaselineEvaluator(train_path, test_path)
    
    # Run baselines
    evaluator.evaluate_random_baseline()
    evaluator.evaluate_majority_baseline()
    evaluator.evaluate_tfidf_logreg()
    evaluator.evaluate_tfidf_svm()
    evaluator.evaluate_mbert()
    
    # Add our model's results (from previous evaluation)
    # Using the KNN evaluation accuracy
    evaluator.add_our_model_results(0.3667)  # 36.67% from KNN eval
    
    # Create comparison chart
    evaluator.create_comparison_chart(
        os.path.join(results_dir, 'baseline_comparison.png')
    )
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    results_path = os.path.join(results_dir, 'baseline_results.json')
    
    # Convert to JSON-serializable format
    save_results = {}
    for name, data in evaluator.results.items():
        save_results[name] = {
            'accuracy': float(data['accuracy']),
            'description': data['description']
        }
        if 'report' in data:
            save_results[name]['report'] = {
                k: {kk: float(vv) for kk, vv in v.items()}
                for k, v in data['report'].items()
            }
    
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\n📁 Saved results: {results_path}")
    
    print("\n✅ Baseline comparison complete!")


if __name__ == "__main__":
    main()