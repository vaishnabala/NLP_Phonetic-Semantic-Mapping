"""
Phase 6: Generate Documentation
Creates abstract, process flow, and research summary
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime


def create_detailed_process_flow():
    """Create detailed process flow diagram"""
    
    fig, ax = plt.subplots(figsize=(16, 20))
    ax.axis('off')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 22)
    
    # Title
    ax.text(8, 21.5, 'PHONETIC-SEMANTIC MAPPING PIPELINE', 
            fontsize=18, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2c3e50', edgecolor='none'),
            color='white')
    
    ax.text(8, 20.8, 'For Hindi-English Code-Mixed Sentiment Analysis', 
            fontsize=12, ha='center', style='italic', color='#7f8c8d')
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: INPUT
    # ═══════════════════════════════════════════════════════════════
    y = 19.5
    
    # Phase header
    ax.text(1, y, 'PHASE 1', fontsize=10, fontweight='bold', color='#e74c3c')
    ax.text(2.5, y, 'INPUT PROCESSING', fontsize=12, fontweight='bold')
    
    y -= 0.8
    
    # Input box
    input_box = mpatches.FancyBboxPatch((1, y-0.6), 6, 1.2, 
                                         boxstyle="round,pad=0.05",
                                         facecolor='#3498db', alpha=0.2,
                                         edgecolor='#3498db', linewidth=2)
    ax.add_patch(input_box)
    ax.text(4, y, 'Raw Input Text', fontsize=11, fontweight='bold', ha='center')
    ax.text(4, y-0.35, '"kiya scene he yar, bohot amazing movie thi"', 
            fontsize=9, ha='center', style='italic', family='monospace')
    
    # Arrow down
    ax.annotate('', xy=(4, y-0.8), xytext=(4, y-0.6),
               arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: SPELLING NORMALIZATION
    # ═══════════════════════════════════════════════════════════════
    y -= 2
    
    ax.text(1, y+0.5, 'PHASE 2', fontsize=10, fontweight='bold', color='#e74c3c')
    ax.text(2.5, y+0.5, 'SPELLING NORMALIZATION', fontsize=12, fontweight='bold')
    
    # Normalizer box
    norm_box = mpatches.FancyBboxPatch((1, y-1.2), 6, 1.8,
                                        boxstyle="round,pad=0.05",
                                        facecolor='#2ecc71', alpha=0.2,
                                        edgecolor='#2ecc71', linewidth=2)
    ax.add_patch(norm_box)
    ax.text(4, y, 'Spelling Normalizer', fontsize=11, fontweight='bold', ha='center')
    ax.text(4, y-0.4, '47+ mapping rules for Romanized Hindi', fontsize=9, ha='center')
    ax.text(4, y-0.8, 'kiya→kya | he→hai | yar→yaar | bohot→bahut', 
            fontsize=8, ha='center', family='monospace', color='#27ae60')
    
    # Side note
    ax.text(8, y-0.3, '📋 Handles variations:\n• kya, kiya, kyaa → kya\n• bohot, boht, bhot → bahut\n• accha, achha → acha',
            fontsize=8, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9f9f9', edgecolor='#ddd'))
    
    # Arrow down
    ax.annotate('', xy=(4, y-1.4), xytext=(4, y-1.2),
               arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))
    
    # Output
    y -= 2.2
    ax.text(4, y, '→ "kya scene hai yaar, bahut amazing movie thi"', 
            fontsize=9, ha='center', family='monospace', color='#27ae60')
    
    # Arrow down
    ax.annotate('', xy=(4, y-0.3), xytext=(4, y-0.1),
               arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: LANGUAGE DETECTION
    # ═══════════════════════════════════════════════════════════════
    y -= 1.2
    
    ax.text(1, y+0.5, 'PHASE 3', fontsize=10, fontweight='bold', color='#e74c3c')
    ax.text(2.5, y+0.5, 'LANGUAGE DETECTION', fontsize=12, fontweight='bold')
    
    # Detector box
    det_box = mpatches.FancyBboxPatch((1, y-1.2), 6, 1.8,
                                       boxstyle="round,pad=0.05",
                                       facecolor='#9b59b6', alpha=0.2,
                                       edgecolor='#9b59b6', linewidth=2)
    ax.add_patch(det_box)
    ax.text(4, y, 'Language Detector', fontsize=11, fontweight='bold', ha='center')
    ax.text(4, y-0.4, '255 Hindi + 289 English vocabulary', fontsize=9, ha='center')
    ax.text(4, y-0.8, 'Tags each word as HI (Hindi) or EN (English)', 
            fontsize=8, ha='center', color='#8e44ad')
    
    # Side note
    ax.text(8, y-0.3, '🏷️ Language Tags:\nkya=HI | scene=EN | hai=HI\nyaar=HI | bahut=HI | amazing=EN\nmovie=EN | thi=HI',
            fontsize=8, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9f9f9', edgecolor='#ddd'))
    
    # Arrow down
    ax.annotate('', xy=(4, y-1.4), xytext=(4, y-1.2),
               arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))
    
    # Output
    y -= 2.2
    ax.text(4, y, '→ [HI, EN, HI, HI, HI, EN, EN, HI]', 
            fontsize=9, ha='center', family='monospace', color='#8e44ad')
    
    # Arrow down
    ax.annotate('', xy=(4, y-0.3), xytext=(4, y-0.1),
               arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: G2P CONVERSION
    # ═══════════════════════════════════════════════════════════════
    y -= 1.2
    
    ax.text(1, y+0.5, 'PHASE 4', fontsize=10, fontweight='bold', color='#e74c3c')
    ax.text(2.5, y+0.5, 'GRAPHEME-TO-PHONEME (G2P)', fontsize=12, fontweight='bold')
    
    # G2P box
    g2p_box = mpatches.FancyBboxPatch((1, y-1.4), 6, 2,
                                       boxstyle="round,pad=0.05",
                                       facecolor='#e67e22', alpha=0.2,
                                       edgecolor='#e67e22', linewidth=2)
    ax.add_patch(g2p_box)
    ax.text(4, y, 'G2P Converter', fontsize=11, fontweight='bold', ha='center')
    ax.text(4, y-0.4, '103 Hindi + 197 English phoneme mappings', fontsize=9, ha='center')
    ax.text(4, y-0.8, 'Converts text to IPA (International Phonetic Alphabet)', 
            fontsize=8, ha='center', color='#d35400')
    ax.text(4, y-1.1, 'Uses language tags to select correct pronunciation', 
            fontsize=8, ha='center', color='#d35400')
    
    # Side note
    ax.text(8, y-0.3, '🔊 Phoneme Examples:\nkya → /kjaː/\nbahut → /bəhʊt/\namazing → /əmeɪzɪŋ/\nmovie → /muːviː/',
            fontsize=8, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9f9f9', edgecolor='#ddd'))
    
    # Arrow down
    ax.annotate('', xy=(4, y-1.6), xytext=(4, y-1.4),
               arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))
    
    # Output
    y -= 2.4
    ax.text(4, y, '→ /kjaː siːn hɛː jaːr bəhʊt əmeɪzɪŋ muːviː tʰiː/', 
            fontsize=9, ha='center', family='monospace', color='#d35400')
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 5: DUAL FEATURE EXTRACTION
    # ═══════════════════════════════════════════════════════════════
    y -= 1
    
    ax.text(1, y, 'PHASE 5', fontsize=10, fontweight='bold', color='#e74c3c')
    ax.text(2.5, y, 'DUAL FEATURE EXTRACTION', fontsize=12, fontweight='bold')
    
    # Split into two paths
    y -= 0.8
    
    # Left path: Phonetic Tokenization
    token_box = mpatches.FancyBboxPatch((0.5, y-1.6), 5, 1.8,
                                         boxstyle="round,pad=0.05",
                                         facecolor='#e74c3c', alpha=0.2,
                                         edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(token_box)
    ax.text(3, y-0.2, 'Phonetic Tokenizer', fontsize=10, fontweight='bold', ha='center')
    ax.text(3, y-0.6, '500-token vocabulary', fontsize=9, ha='center')
    ax.text(3, y-1, 'Converts IPA to token IDs', fontsize=8, ha='center')
    ax.text(3, y-1.3, '→ [12, 45, 8, 23, 67, ...]', 
            fontsize=8, ha='center', family='monospace', color='#c0392b')
    
    # Right path: Prosody Extraction
    pros_box = mpatches.FancyBboxPatch((6.5, y-1.6), 5, 1.8,
                                        boxstyle="round,pad=0.05",
                                        facecolor='#f39c12', alpha=0.2,
                                        edgecolor='#f39c12', linewidth=2)
    ax.add_patch(pros_box)
    ax.text(9, y-0.2, 'Prosody Extractor', fontsize=10, fontweight='bold', ha='center')
    ax.text(9, y-0.6, '24-dimensional features', fontsize=9, ha='center')
    ax.text(9, y-1, '🎵 Inspired by Indian classical music', fontsize=8, ha='center')
    ax.text(9, y-1.3, '→ [0.3, 0.5, 0.2, 0.8, ...]', 
            fontsize=8, ha='center', family='monospace', color='#e67e22')
    
    # Prosody details
    ax.text(12.5, y-0.3, '🎵 Prosody Features:\n• Length (6): vowel duration\n• Stress (6): emphasis patterns\n• Rhythm (6): syllable patterns\n• Intensity (6): aspiration, nasal',
            fontsize=8, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef9e7', edgecolor='#f39c12'))
    
    # Arrows converging
    y -= 2.2
    ax.annotate('', xy=(6, y), xytext=(3, y+0.4),
               arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))
    ax.annotate('', xy=(6, y), xytext=(9, y+0.4),
               arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 6: NEURAL NETWORK ENCODING
    # ═══════════════════════════════════════════════════════════════
    y -= 0.8
    
    ax.text(1, y+0.3, 'PHASE 6', fontsize=10, fontweight='bold', color='#e74c3c')
    ax.text(2.5, y+0.3, 'NEURAL ENCODING', fontsize=12, fontweight='bold')
    
    # Encoder box
    enc_box = mpatches.FancyBboxPatch((2, y-2), 8, 2.2,
                                       boxstyle="round,pad=0.05",
                                       facecolor='#1abc9c', alpha=0.2,
                                       edgecolor='#1abc9c', linewidth=2)
    ax.add_patch(enc_box)
    ax.text(6, y-0.3, 'Enhanced Phonetic Encoder', fontsize=11, fontweight='bold', ha='center')
    
    # Sub-components
    ax.text(3.5, y-0.8, '📦 Embedding Layer\n(64-dim)', fontsize=8, ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#16a085'))
    ax.text(6, y-0.8, '🔄 BiLSTM\n(2 layers, 128 hidden)', fontsize=8, ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#16a085'))
    ax.text(8.5, y-0.8, '🎯 Attention\nMechanism', fontsize=8, ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#16a085'))
    
    ax.text(6, y-1.5, '⊕ Prosody Fusion → LayerNorm → 64-dim Output', 
            fontsize=9, ha='center', color='#16a085')
    
    # Side note
    ax.text(11, y-0.8, '🧠 Model Stats:\n• Parameters: 705,673\n• Output: 64-dim embedding\n• Normalized (unit length)',
            fontsize=8, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9f9f9', edgecolor='#ddd'))
    
    # Arrow down
    ax.annotate('', xy=(6, y-2.2), xytext=(6, y-2),
               arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 7: OUTPUT
    # ═══════════════════════════════════════════════════════════════
    y -= 3
    
    ax.text(1, y+0.3, 'PHASE 7', fontsize=10, fontweight='bold', color='#e74c3c')
    ax.text(2.5, y+0.3, 'OUTPUT EMBEDDING', fontsize=12, fontweight='bold')
    
    # Output box
    out_box = mpatches.FancyBboxPatch((2, y-1), 8, 1.2,
                                       boxstyle="round,pad=0.05",
                                       facecolor='#9b59b6', alpha=0.3,
                                       edgecolor='#9b59b6', linewidth=2)
    ax.add_patch(out_box)
    ax.text(6, y-0.2, '64-Dimensional Semantic Embedding', fontsize=11, fontweight='bold', ha='center')
    ax.text(6, y-0.6, '[0.23, -0.45, 0.12, 0.78, -0.34, ..., 0.56]', 
            fontsize=9, ha='center', family='monospace', color='#8e44ad')
    
    # Usage note
    ax.text(6, y-1.5, '✨ Use for: Similarity search | Clustering | Classification (KNN)', 
            fontsize=9, ha='center', style='italic', color='#7f8c8d')
    
    plt.tight_layout()
    return fig


def create_training_flow():
    """Create training pipeline diagram"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    
    # Title
    ax.text(7, 11.5, 'CONTRASTIVE LEARNING TRAINING PIPELINE', 
            fontsize=16, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#8e44ad', edgecolor='none'),
            color='white')
    
    # ═══════════════════════════════════════════════════════════════
    # TRIPLET FORMATION
    # ═══════════════════════════════════════════════════════════════
    
    ax.text(1, 10.2, '1. TRIPLET FORMATION', fontsize=12, fontweight='bold', color='#2c3e50')
    
    # Anchor
    ax.text(2, 9, '⚓ ANCHOR', fontsize=10, fontweight='bold', ha='center', color='#3498db')
    ax.text(2, 8.5, '"bahut amazing\nmovie thi"', fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#3498db', alpha=0.2))
    ax.text(2, 7.8, 'Label: POSITIVE', fontsize=8, ha='center', color='#3498db')
    
    # Positive
    ax.text(7, 9, '✓ POSITIVE', fontsize=10, fontweight='bold', ha='center', color='#27ae60')
    ax.text(7, 8.5, '"really enjoyed\nthe film"', fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#27ae60', alpha=0.2))
    ax.text(7, 7.8, 'Label: POSITIVE', fontsize=8, ha='center', color='#27ae60')
    ax.text(7, 7.4, '(Same class as anchor)', fontsize=7, ha='center', color='#27ae60')
    
    # Negative
    ax.text(12, 9, '✗ NEGATIVE', fontsize=10, fontweight='bold', ha='center', color='#e74c3c')
    ax.text(12, 8.5, '"bahut boring\ntha"', fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#e74c3c', alpha=0.2))
    ax.text(12, 7.8, 'Label: NEGATIVE', fontsize=8, ha='center', color='#e74c3c')
    ax.text(12, 7.4, '(Different class)', fontsize=7, ha='center', color='#e74c3c')
    
    # Arrows
    ax.annotate('', xy=(5, 8.5), xytext=(3.5, 8.5),
               arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
    ax.annotate('', xy=(10, 8.5), xytext=(8.5, 8.5),
               arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
    
    # ═══════════════════════════════════════════════════════════════
    # ENCODING
    # ═══════════════════════════════════════════════════════════════
    
    ax.text(1, 6.8, '2. ENCODING (Shared Encoder)', fontsize=12, fontweight='bold', color='#2c3e50')
    
    # Encoder box
    enc_box = mpatches.FancyBboxPatch((3, 5.5), 8, 1,
                                       boxstyle="round,pad=0.05",
                                       facecolor='#1abc9c', alpha=0.3,
                                       edgecolor='#1abc9c', linewidth=2)
    ax.add_patch(enc_box)
    ax.text(7, 6, 'Enhanced Phonetic Encoder (shared weights)', fontsize=10, fontweight='bold', ha='center')
    
    # Arrows down
    ax.annotate('', xy=(2, 6.5), xytext=(2, 7.2),
               arrowprops=dict(arrowstyle='->', color='#34495e', lw=1.5))
    ax.annotate('', xy=(7, 6.5), xytext=(7, 7.2),
               arrowprops=dict(arrowstyle='->', color='#34495e', lw=1.5))
    ax.annotate('', xy=(12, 6.5), xytext=(12, 7.2),
               arrowprops=dict(arrowstyle='->', color='#34495e', lw=1.5))
    
    # Embeddings
    ax.text(2, 5, 'Emb_A', fontsize=9, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#3498db', alpha=0.3))
    ax.text(7, 5, 'Emb_P', fontsize=9, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#27ae60', alpha=0.3))
    ax.text(12, 5, 'Emb_N', fontsize=9, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#e74c3c', alpha=0.3))
    
    # ═══════════════════════════════════════════════════════════════
    # TRIPLET LOSS
    # ═══════════════════════════════════════════════════════════════
    
    ax.text(1, 4.3, '3. TRIPLET LOSS COMPUTATION', fontsize=12, fontweight='bold', color='#2c3e50')
    
    # Loss formula box
    loss_box = mpatches.FancyBboxPatch((2, 2.8), 10, 1.2,
                                        boxstyle="round,pad=0.05",
                                        facecolor='#f39c12', alpha=0.2,
                                        edgecolor='#f39c12', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(7, 3.6, 'Triplet Loss Formula:', fontsize=10, fontweight='bold', ha='center')
    ax.text(7, 3.1, 'L = max(0, d(A,P) - d(A,N) + margin)', 
            fontsize=11, ha='center', family='monospace', color='#d35400')
    
    # Explanation
    ax.text(7, 2.3, '📐 Goal: Push d(A,P) < d(A,N) - margin', fontsize=9, ha='center')
    ax.text(7, 1.9, 'd = Euclidean distance | margin = 0.5', fontsize=8, ha='center', color='#7f8c8d')
    
    # Visual representation
    ax.text(2, 1.3, '     A ●───● P   (close)', fontsize=10, ha='left', family='monospace', color='#27ae60')
    ax.text(2, 0.9, '       \\', fontsize=10, ha='left', family='monospace')
    ax.text(2, 0.5, '        \\', fontsize=10, ha='left', family='monospace')
    ax.text(2, 0.1, '         ● N    (far)', fontsize=10, ha='left', family='monospace', color='#e74c3c')
    
    # Training stats
    ax.text(9, 1.2, '📊 Training Config:\n• Epochs: 40\n• Batch: 8 triplets\n• LR: 0.0005\n• Best Val Acc: 60%',
            fontsize=9, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9f9f9', edgecolor='#ddd'))
    
    plt.tight_layout()
    return fig


def save_abstract_to_file(project_dir):
    """Save abstract as text file"""
    
    abstract = """
═══════════════════════════════════════════════════════════════════════════════
                              RESEARCH ABSTRACT
═══════════════════════════════════════════════════════════════════════════════

TITLE: Phonetic-Semantic Mapping for Sentiment Analysis in Hindi-English 
       Code-Mixed Text: A Prosody-Enhanced Contrastive Learning Approach

AUTHORS: [Your Name]
INSTITUTION: [Your Institution]
DATE: {date}

───────────────────────────────────────────────────────────────────────────────

ABSTRACT

Code-mixing, the practice of alternating between languages within a single 
utterance, is prevalent among multilingual speakers, particularly in India 
where Hindi-English mixing is ubiquitous in social media and daily 
communication. Traditional NLP models struggle with code-mixed text due to 
inconsistent Romanized spellings (e.g., "kya", "kiya", "kyaa" for the same 
word) and the absence of standardized orthography.

This paper presents a novel phonetic-semantic mapping approach for sentiment 
analysis in Hindi-English code-mixed text. Our contributions include: 

(1) A grapheme-to-phoneme (G2P) conversion system specifically designed for 
    Romanized Hindi-English text, handling 103 Hindi and 197 English phoneme 
    mappings; 

(2) A spelling normalization module addressing 47+ common spelling variations; 

(3) A prosody feature extractor inspired by Indian classical music theory, 
    capturing rhythmic and tonal patterns in 24-dimensional vectors; and 

(4) A contrastive learning framework using triplet loss to learn semantically 
    meaningful phonetic embeddings.

We evaluate our approach on a dataset of 210 code-mixed samples across three 
sentiment classes (positive, negative, neutral). Our phonetic-prosody model 
achieves 46.67% accuracy, outperforming random baseline (33.33%) by 13.34 
percentage points. While traditional TF-IDF approaches (76.67%) and mBERT 
(66.67%) achieve higher accuracy on this limited dataset, our work establishes 
a foundation for phonetic-aware processing of code-mixed text. The phonetic 
approach shows particular promise for handling spelling variations and could 
benefit from larger training datasets.

───────────────────────────────────────────────────────────────────────────────

KEYWORDS: Code-mixing, Hindi-English, Phonetic embeddings, Sentiment analysis, 
          Prosody features, Contrastive learning, Low-resource NLP

───────────────────────────────────────────────────────────────────────────────

RESEARCH CONTRIBUTIONS:

1. NOVEL G2P SYSTEM
   - First grapheme-to-phoneme converter for Romanized Hindi-English
   - 103 Hindi phoneme mappings + 197 English phoneme mappings
   - Language-aware conversion using automatic language detection

2. SPELLING NORMALIZATION
   - Handles 47+ spelling variations in Romanized Hindi
   - Examples: kya/kiya/kyaa → kya, bohot/boht/bhot → bahut
   - Improves phonetic consistency

3. PROSODY FEATURE EXTRACTION
   - 24-dimensional prosodic features
   - Inspired by Indian classical music concepts (taal, laya)
   - Captures rhythm, stress, and intensity patterns

4. CONTRASTIVE LEARNING APPROACH
   - Triplet loss training for semantic similarity
   - Learns embeddings where similar sentiments cluster together
   - Novel application to code-mixed text

───────────────────────────────────────────────────────────────────────────────

RESULTS SUMMARY:

| Model                          | Accuracy | vs Random |
|--------------------------------|----------|-----------|
| TF-IDF + Logistic Regression   | 76.67%   | +43.34%   |
| TF-IDF + SVM                   | 76.67%   | +43.34%   |
| mBERT + Logistic Regression    | 66.67%   | +33.34%   |
| Our Phonetic-Prosody Model     | 46.67%   | +13.34%   |
| Random Baseline                | 33.33%   | -         |

───────────────────────────────────────────────────────────────────────────────

LIMITATIONS AND FUTURE WORK:

1. Small dataset (210 samples) limits deep learning performance
2. Need larger code-mixed corpora for better training
3. Could incorporate pre-trained phonetic embeddings
4. Extend to other Indian language pairs (Tamil-English, Telugu-English)

═══════════════════════════════════════════════════════════════════════════════
""".format(date=datetime.now().strftime("%B %Y"))
    
    # Save to file
    docs_dir = os.path.join(project_dir, 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    
    abstract_path = os.path.join(docs_dir, 'abstract.txt')
    with open(abstract_path, 'w', encoding='utf-8') as f:
        f.write(abstract)
    
    return abstract_path


def main():
    print("="*60)
    print("   PHASE 6: GENERATING DOCUMENTATION")
    print("="*60)
    
    # Paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create process flow diagram
    print("\n1. Creating detailed process flow diagram...")
    fig1 = create_detailed_process_flow()
    fig1.savefig(os.path.join(results_dir, 'process_flow_detailed.png'), 
                 dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print("   Saved: results/process_flow_detailed.png")
    
    # Create training flow diagram
    print("\n2. Creating training pipeline diagram...")
    fig2 = create_training_flow()
    fig2.savefig(os.path.join(results_dir, 'training_pipeline.png'), 
                 dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print("   Saved: results/training_pipeline.png")
    
    # Save abstract
    print("\n3. Saving abstract to file...")
    abstract_path = save_abstract_to_file(project_dir)
    print(f"   Saved: {abstract_path}")
    
    print("\n" + "="*60)
    print("   ✅ DOCUMENTATION GENERATED!")
    print("="*60)
    
    print("\n📁 Generated Files:")
    print("   - results/process_flow_detailed.png  (Pipeline diagram)")
    print("   - results/training_pipeline.png      (Training flow)")
    print("   - docs/abstract.txt                  (Research abstract)")
    
    print("\n📝 Abstract Preview:")
    print("-"*60)
    print("""
TITLE: Phonetic-Semantic Mapping for Sentiment Analysis in 
       Hindi-English Code-Mixed Text

KEYWORDS: Code-mixing, Hindi-English, Phonetic embeddings, 
          Sentiment analysis, Prosody features, Contrastive learning

BEST RESULT: 46.67% accuracy (vs 33.33% random baseline)
    """)
    print("-"*60)


if __name__ == "__main__":
    main()