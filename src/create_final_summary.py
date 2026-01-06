"""
Phase 5.4: Create Final Summary and Visualizations
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def create_model_comparison():
    """Create final comparison chart"""
    
    # Data
    models = [
        'Random\nBaseline',
        'Our Model\n(Phonetic+Prosody)',
        'mBERT',
        'TF-IDF\n+LogReg'
    ]
    
    accuracies = [33.33, 46.67, 66.67, 76.67]
    
    colors = ['#95a5a6', '#e74c3c', '#9b59b6', '#3498db']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(models, accuracies, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 1.5,
            f'{acc:.1f}%',
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold'
        )
    
    # Add improvement annotations
    ax.annotate(
        '+13.3%',
        xy=(1, 46.67),
        xytext=(1, 38),
        fontsize=11,
        ha='center',
        color='green',
        fontweight='bold'
    )
    
    # Styling
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_xlabel('Model', fontsize=13)
    ax.set_title('Code-Mixed Hindi-English Sentiment Analysis\nModel Comparison', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(0, 95)
    
    # Random chance line
    ax.axhline(y=33.33, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(3.5, 35, 'Random Chance (33.3%)', fontsize=10, color='red', alpha=0.8)
    
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend for our model
    ax.annotate(
        '⭐ Our Novel Approach',
        xy=(1, 46.67),
        xytext=(2.2, 52),
        fontsize=11,
        arrowprops=dict(arrowstyle='->', color='#e74c3c'),
        color='#e74c3c',
        fontweight='bold'
    )
    
    plt.tight_layout()
    return fig


def create_contribution_summary():
    """Create visual summary of research contributions"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Research Contributions Summary', 
            fontsize=18, fontweight='bold', ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.89, 'Phonetic-Semantic Mapping for Indian Code-Mixed Languages', 
            fontsize=14, ha='center', transform=ax.transAxes, style='italic')
    
    # Contributions
    contributions = [
        ('🔤', 'Novel G2P System', 
         'First grapheme-to-phoneme converter for\nRomanized Hindi-English code-mixed text\n(103 Hindi + 197 English mappings)'),
        
        ('🎵', 'Prosody Features', 
         'Music-inspired prosodic features based on\nIndian classical music concepts\n(24-dimensional feature vector)'),
        
        ('📝', 'Spelling Normalizer', 
         'Handles 47+ spelling variations in\nRomanized Hindi (kya, kiya, kyaa → kya)'),
        
        ('🧠', 'Contrastive Learning', 
         'Triplet loss training for learning\nsemantic similarity in code-mixed text'),
    ]
    
    y_positions = [0.72, 0.52, 0.32, 0.12]
    
    for (emoji, title, desc), y in zip(contributions, y_positions):
        # Emoji
        ax.text(0.08, y, emoji, fontsize=30, ha='center', transform=ax.transAxes)
        
        # Title
        ax.text(0.15, y + 0.02, title, fontsize=14, fontweight='bold', 
                ha='left', transform=ax.transAxes)
        
        # Description
        ax.text(0.15, y - 0.08, desc, fontsize=11, ha='left', 
                transform=ax.transAxes, linespacing=1.5)
    
    # Results box
    results_text = """
    RESULTS
    ━━━━━━━━━━━━━━━━━━━━━━━
    Dataset: 210 samples
    (150 train, 30 val, 30 test)
    
    Our Model:     46.67%
    mBERT:         66.67%
    TF-IDF+LR:     76.67%
    Random:        33.33%
    
    ✓ Beats random by +13.3%
    ✓ Novel phonetic approach
    ✓ Limited by small dataset
    """
    
    props = dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', edgecolor='#34495e', linewidth=2)
    ax.text(0.75, 0.45, results_text, fontsize=11, ha='left', va='center',
            transform=ax.transAxes, bbox=props, family='monospace')
    
    plt.tight_layout()
    return fig


def create_pipeline_diagram():
    """Create pipeline architecture diagram"""
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    
    # Title
    ax.text(7, 5.7, 'Model Pipeline Architecture', fontsize=16, fontweight='bold', ha='center')
    
    # Boxes
    boxes = [
        (1, 3, 'INPUT\n"kiya scene he\nyar amazing"', '#3498db'),
        (3.5, 3, 'Spelling\nNormalizer\n(47 rules)', '#2ecc71'),
        (6, 3, 'G2P\nConverter\n(IPA output)', '#2ecc71'),
        (8.5, 3, 'Phonetic\nEncoder\n(LSTM)', '#e74c3c'),
        (11, 4, 'Prosody\nExtractor\n(24 features)', '#f39c12'),
        (11, 2, 'Attention\nMechanism', '#e74c3c'),
        (13, 3, 'OUTPUT\n64-dim\nEmbedding', '#9b59b6'),
    ]
    
    for x, y, text, color in boxes:
        box = plt.Rectangle((x-0.8, y-0.7), 1.6, 1.4, 
                            facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    arrows = [
        (1.8, 3, 2.7, 3),
        (4.3, 3, 5.2, 3),
        (6.8, 3, 7.7, 3),
        (9.3, 3.5, 10.2, 3.8),
        (9.3, 2.5, 10.2, 2.2),
        (11.8, 3.5, 12.2, 3.2),
        (11.8, 2.5, 12.2, 2.8),
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))
    
    # Sample flow
    ax.text(7, 1, 'Sample: "kiya" → "kya" → /kjaː/ → [0.23, -0.45, ...] → Embedding', 
            fontsize=10, ha='center', style='italic', color='#7f8c8d')
    
    plt.tight_layout()
    return fig


def main():
    print("="*60)
    print("   CREATING FINAL SUMMARY VISUALIZATIONS")
    print("="*60)
    
    # Paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create visualizations
    print("\n1. Creating model comparison chart...")
    fig1 = create_model_comparison()
    fig1.savefig(os.path.join(results_dir, 'final_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("   Saved: results/final_comparison.png")
    
    print("\n2. Creating contribution summary...")
    fig2 = create_contribution_summary()
    fig2.savefig(os.path.join(results_dir, 'contribution_summary.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("   Saved: results/contribution_summary.png")
    
    print("\n3. Creating pipeline diagram...")
    fig3 = create_pipeline_diagram()
    fig3.savefig(os.path.join(results_dir, 'pipeline_diagram.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("   Saved: results/pipeline_diagram.png")
    
    print("\n" + "="*60)
    print("   ✅ ALL VISUALIZATIONS CREATED!")
    print("="*60)
    
    print("\n📁 Files in results folder:")
    print("   - final_comparison.png      (Model comparison)")
    print("   - contribution_summary.png  (Research contributions)")
    print("   - pipeline_diagram.png      (Architecture diagram)")
    print("   - embedding_tsne.png        (From earlier)")
    print("   - confusion_matrix.png      (From earlier)")
    print("   - class_separation.png      (From earlier)")
    print("   - baseline_comparison.png   (From earlier)")


if __name__ == "__main__":
    main()