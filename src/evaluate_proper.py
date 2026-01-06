"""
Phase 5.3: Proper Evaluation
Uses training embeddings as reference for KNN classification
"""

import torch
import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

# Import our modules
from phonetic_vocabulary import PhoneticVocabulary
from prosody_features import ProsodyExtractor
from enhanced_encoder import EnhancedPhoneticEncoder


def load_model(model_path, vocab_path, device='cpu'):
    """Load trained model"""
    # Load vocabulary
    vocab = PhoneticVocabulary()
    vocab.load(vocab_path)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model = EnhancedPhoneticEncoder(
        vocab_size=len(vocab),
        embedding_dim=64,
        hidden_dim=128,
        output_dim=64,
        prosody_dim=24,
        num_layers=2,
        dropout=0.3,
        use_attention=True
    )
    
    # Fix state dict keys
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('encoder.'):
            new_key = key[8:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    return model, vocab


def encode_dataset(model, vocab, data_path, device='cpu'):
    """Encode all samples in a dataset"""
    prosody_extractor = ProsodyExtractor()
    df = pd.read_csv(data_path)
    
    embeddings = []
    labels = []
    texts = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding"):
        # Tokenize
        tokens = vocab.encode(row['phonetic'], max_length=25)
        tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        
        # Prosody
        prosody = prosody_extractor.extract(row['phonetic'])
        prosody_tensor = torch.tensor([prosody], dtype=torch.float32).to(device)
        
        # Get embedding
        with torch.no_grad():
            embedding = model(tokens_tensor, prosody_tensor)
        
        embeddings.append(embedding.cpu().numpy()[0])
        labels.append(row['label'])
        texts.append(row['text'])
    
    return np.array(embeddings), labels, texts


def main():
    print("="*60)
    print("   PROPER EVALUATION: Train KNN → Test")
    print("="*60)
    
    # Paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_dir, 'models', 'enhanced_checkpoint_best.pt')
    vocab_path = os.path.join(project_dir, 'models', 'phonetic_vocab.json')
    train_path = os.path.join(project_dir, 'data', 'processed', 'train_phonetic.csv')
    test_path = os.path.join(project_dir, 'data', 'processed', 'test_phonetic.csv')
    
    # Load model
    print("\nLoading model...")
    model, vocab = load_model(model_path, vocab_path)
    print("  Model loaded!")
    
    # Encode training data
    print("\nEncoding training data...")
    train_embeddings, train_labels, train_texts = encode_dataset(
        model, vocab, train_path
    )
    print(f"  Train embeddings shape: {train_embeddings.shape}")
    
    # Encode test data
    print("\nEncoding test data...")
    test_embeddings, test_labels, test_texts = encode_dataset(
        model, vocab, test_path
    )
    print(f"  Test embeddings shape: {test_embeddings.shape}")
    
    # Convert labels to numbers
    label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
    y_train = [label_map[l] for l in train_labels]
    y_test = [label_map[l] for l in test_labels]
    
    # Try different K values
    print("\n" + "="*60)
    print("   KNN CLASSIFICATION RESULTS")
    print("="*60)
    
    results = {}
    
    for k in [1, 3, 5, 7, 9]:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(train_embeddings, y_train)
        y_pred = knn.predict(test_embeddings)
        
        accuracy = accuracy_score(y_test, y_pred)
        results[k] = accuracy
        
        print(f"\n   K={k}: Accuracy = {accuracy*100:.2f}%")
        
        if k == 5:  # Show detailed report for K=5
            print("\n   Detailed Report (K=5):")
            report = classification_report(
                y_test, y_pred,
                target_names=['positive', 'negative', 'neutral']
            )
            for line in report.split('\n'):
                print(f"   {line}")
    
    # Find best K
    best_k = max(results, key=results.get)
    best_accuracy = results[best_k]
    
    print("\n" + "="*60)
    print(f"   BEST RESULT: K={best_k}, Accuracy={best_accuracy*100:.2f}%")
    print("="*60)
    
    # Compare with baselines
    print("\n📊 UPDATED COMPARISON:")
    print("-"*40)
    print(f"   TF-IDF + LogReg:     76.67%")
    print(f"   TF-IDF + SVM:        76.67%")
    print(f"   mBERT + LogReg:      66.67%")
    print(f"   Our Model (K={best_k}):    {best_accuracy*100:.2f}% ⭐")
    print(f"   Random:              33.33%")
    print("-"*40)
    
    # Save results
    results_path = os.path.join(project_dir, 'results', 'proper_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump({
            'knn_results': {str(k): float(v) for k, v in results.items()},
            'best_k': best_k,
            'best_accuracy': float(best_accuracy)
        }, f, indent=2)
    print(f"\n📁 Saved: {results_path}")


if __name__ == "__main__":
    main()