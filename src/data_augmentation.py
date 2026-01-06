"""
Data Augmentation for Hindi-English Code-Mixed Text
=====================================================
Generates additional training samples to improve model performance.

Techniques used:
1. Synonym Replacement (Hindi & English)
2. Spelling Variation (Romanized Hindi)
3. Word Dropout (Random word removal)
4. Word Shuffle (Partial reordering)
5. Code-Mix Ratio Change (Hindi↔English swap)

These augmentations preserve sentiment while creating variety.
"""

import random
import pandas as pd
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CodeMixAugmenter:
    """
    Augments Hindi-English code-mixed text data.
    """
    
    def __init__(self, augment_factor=3):
        """
        Initialize augmenter.
        
        Args:
            augment_factor (int): How many augmented samples per original
        """
        self.augment_factor = augment_factor
        
        # Hindi synonyms (word → list of alternatives)
        self.hindi_synonyms = {
            # Positive words
            "acha": ["badhiya", "shandar", "zabardast", "mast"],
            "badhiya": ["acha", "shandar", "best", "kamaal"],
            "khush": ["happy", "prasann", "mast"],
            "mast": ["acha", "badhiya", "cool", "awesome"],
            "shandar": ["zabardast", "kamaal", "amazing", "great"],
            
            # Negative words
            "bura": ["kharab", "ganda", "bekaar", "worst"],
            "bekaar": ["bekar", "faltu", "waste", "useless"],
            "bakwas": ["bekar", "faltu", "nonsense", "rubbish"],
            "boring": ["sust", "bekaar", "dull"],
            
            # Common words
            "yaar": ["dost", "bhai", "bro", "friend"],
            "dost": ["yaar", "friend", "bhai"],
            "bhai": ["yaar", "dost", "bro"],
            "bahut": ["bohot", "kaafi", "boht", "very"],
            "thoda": ["kuch", "zara", "little", "bit"],
            
            # Verbs
            "hai": ["he", "hain", "is"],
            "tha": ["thaa", "was"],
            "thi": ["thee", "was"],
            "karna": ["karo", "kar", "do"],
            "jana": ["jao", "ja", "go"],
            "aana": ["aao", "aa", "come"],
            "dekhna": ["dekho", "dekh", "see", "watch"],
            
            # Time
            "aaj": ["today", "aj"],
            "kal": ["tomorrow", "yesterday"],
            "abhi": ["now", "abi"],
            
            # Others
            "kya": ["what", "kia"],
            "kyun": ["why", "kyon"],
            "kahan": ["where", "kidhar"],
            "kaun": ["who", "kon"],
        }
        
        # English synonyms
        self.english_synonyms = {
            # Positive
            "good": ["nice", "great", "awesome", "acha"],
            "great": ["amazing", "awesome", "fantastic", "shandar"],
            "amazing": ["awesome", "fantastic", "incredible", "zabardast"],
            "awesome": ["amazing", "great", "cool", "mast"],
            "happy": ["glad", "pleased", "khush", "excited"],
            "nice": ["good", "lovely", "pleasant", "acha"],
            "best": ["greatest", "finest", "sabse acha"],
            "love": ["adore", "like", "pyaar"],
            
            # Negative
            "bad": ["terrible", "awful", "bura", "kharab"],
            "terrible": ["awful", "horrible", "bad", "bekaar"],
            "boring": ["dull", "tedious", "sust", "bekaar"],
            "sad": ["unhappy", "upset", "dukhi"],
            "worst": ["terrible", "horrible", "sabse bura"],
            "disappointed": ["let down", "upset", "dukhi"],
            "frustrating": ["annoying", "irritating"],
            
            # Common
            "very": ["really", "so", "bahut", "kaafi"],
            "really": ["very", "truly", "sach mein"],
            "movie": ["film", "picture"],
            "friend": ["buddy", "pal", "dost", "yaar"],
            
            # Verbs
            "go": ["move", "leave", "jana"],
            "come": ["arrive", "reach", "aana"],
            "see": ["watch", "look", "dekhna"],
            "want": ["need", "wish", "chahna"],
            "like": ["enjoy", "love", "pasand"],
            "think": ["believe", "feel", "sochna"],
        }
        
        # Spelling variations for Hindi words
        self.spelling_variations = {
            "bahut": ["bohot", "boht", "bhot", "bhut"],
            "acha": ["accha", "achha", "achchha"],
            "kya": ["kiya", "kyaa", "kia"],
            "hai": ["he", "h", "hain"],
            "nahi": ["nai", "nahin", "nhi"],
            "theek": ["thik", "theik", "tik"],
            "yaar": ["yar", "yr"],
            "mein": ["me", "mai", "main"],
            "hoon": ["hun", "hu"],
            "toh": ["to", "tho"],
            "bhi": ["bhee", "v"],
            "aur": ["or", "ar"],
            "lekin": ["par", "magar", "but"],
            "abhi": ["abi", "abhe"],
            "gaya": ["gya", "gayaa"],
            "raha": ["rha", "rahaa"],
        }
        
        print(f"CodeMixAugmenter initialized:")
        print(f"  Hindi synonyms: {len(self.hindi_synonyms)} words")
        print(f"  English synonyms: {len(self.english_synonyms)} words")
        print(f"  Spelling variations: {len(self.spelling_variations)} words")
        print(f"  Augment factor: {augment_factor}x")
    
    def synonym_replacement(self, text, n_replacements=2):
        """
        Replace words with synonyms.
        
        Args:
            text (str): Original text
            n_replacements (int): Max words to replace
            
        Returns:
            str: Augmented text
        """
        words = text.lower().split()
        new_words = words.copy()
        
        # Find replaceable words
        replaceable = []
        for i, word in enumerate(words):
            if word in self.hindi_synonyms or word in self.english_synonyms:
                replaceable.append(i)
        
        # Randomly replace some
        random.shuffle(replaceable)
        replaced = 0
        
        for idx in replaceable:
            if replaced >= n_replacements:
                break
            
            word = words[idx]
            
            # Get synonyms
            if word in self.hindi_synonyms:
                synonyms = self.hindi_synonyms[word]
            elif word in self.english_synonyms:
                synonyms = self.english_synonyms[word]
            else:
                continue
            
            # Replace with random synonym
            new_words[idx] = random.choice(synonyms)
            replaced += 1
        
        return " ".join(new_words)
    
    def spelling_variation(self, text):
        """
        Apply spelling variations to Hindi words.
        
        Args:
            text (str): Original text
            
        Returns:
            str: Text with spelling variations
        """
        words = text.lower().split()
        new_words = []
        
        for word in words:
            if word in self.spelling_variations and random.random() < 0.5:
                new_words.append(random.choice(self.spelling_variations[word]))
            else:
                new_words.append(word)
        
        return " ".join(new_words)
    
    def word_dropout(self, text, dropout_prob=0.15):
        """
        Randomly remove words (except important ones).
        
        Args:
            text (str): Original text
            dropout_prob (float): Probability of dropping each word
            
        Returns:
            str: Text with some words removed
        """
        words = text.lower().split()
        
        # Don't drop if too short
        if len(words) <= 3:
            return text
        
        # Important words to keep
        important = {"nahi", "bahut", "very", "not", "no", "best", "worst"}
        
        new_words = []
        for word in words:
            if word in important:
                new_words.append(word)
            elif random.random() > dropout_prob:
                new_words.append(word)
        
        # Ensure at least 2 words remain
        if len(new_words) < 2:
            return text
        
        return " ".join(new_words)
    
    def word_shuffle(self, text, shuffle_ratio=0.2):
        """
        Shuffle some adjacent words.
        
        Args:
            text (str): Original text
            shuffle_ratio (float): Ratio of words to shuffle
            
        Returns:
            str: Text with some words shuffled
        """
        words = text.lower().split()
        
        if len(words) <= 3:
            return text
        
        new_words = words.copy()
        n_shuffles = max(1, int(len(words) * shuffle_ratio))
        
        for _ in range(n_shuffles):
            idx = random.randint(0, len(new_words) - 2)
            new_words[idx], new_words[idx + 1] = new_words[idx + 1], new_words[idx]
        
        return " ".join(new_words)
    
    def code_mix_swap(self, text):
        """
        Swap some Hindi words with English equivalents or vice versa.
        
        Args:
            text (str): Original text
            
        Returns:
            str: Text with language swaps
        """
        # Hindi to English mappings
        hi_to_en = {
            "acha": "good", "bura": "bad", "bahut": "very",
            "khush": "happy", "dukhi": "sad", "yaar": "friend",
            "dost": "friend", "kaam": "work", "ghar": "home",
            "aaj": "today", "kal": "tomorrow", "abhi": "now",
        }
        
        # English to Hindi mappings
        en_to_hi = {v: k for k, v in hi_to_en.items()}
        
        words = text.lower().split()
        new_words = []
        
        for word in words:
            if word in hi_to_en and random.random() < 0.3:
                new_words.append(hi_to_en[word])
            elif word in en_to_hi and random.random() < 0.3:
                new_words.append(en_to_hi[word])
            else:
                new_words.append(word)
        
        return " ".join(new_words)
    
    def augment_text(self, text):
        """
        Apply random augmentation to a single text.
        
        Args:
            text (str): Original text
            
        Returns:
            str: Augmented text
        """
        # Choose random augmentation(s)
        augmentations = [
            self.synonym_replacement,
            self.spelling_variation,
            self.word_dropout,
            self.word_shuffle,
            self.code_mix_swap,
        ]
        
        # Apply 1-2 random augmentations
        n_augs = random.randint(1, 2)
        chosen = random.sample(augmentations, n_augs)
        
        augmented = text
        for aug_func in chosen:
            augmented = aug_func(augmented)
        
        return augmented
    
    def augment_dataset(self, df, text_column='text', label_column='label'):
        """
        Augment entire dataset.
        
        Args:
            df (pd.DataFrame): Original dataset
            text_column (str): Name of text column
            label_column (str): Name of label column
            
        Returns:
            pd.DataFrame: Augmented dataset (original + new samples)
        """
        print(f"\nAugmenting dataset...")
        print(f"  Original samples: {len(df)}")
        
        augmented_texts = []
        augmented_labels = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
            original_text = row[text_column]
            label = row[label_column]
            
            # Generate augmented versions
            for _ in range(self.augment_factor):
                aug_text = self.augment_text(original_text)
                
                # Only add if different from original
                if aug_text != original_text.lower():
                    augmented_texts.append(aug_text)
                    augmented_labels.append(label)
        
        # Create augmented dataframe
        aug_df = pd.DataFrame({
            text_column: augmented_texts,
            label_column: augmented_labels
        })
        
        # Combine original and augmented
        combined_df = pd.concat([df, aug_df], ignore_index=True)
        
        print(f"  Augmented samples: {len(aug_df)}")
        print(f"  Total samples: {len(combined_df)}")
        
        return combined_df


def augment_and_save():
    """
    Augment training data and save.
    """
    print("=" * 60)
    print("Data Augmentation for Code-Mixed Dataset")
    print("=" * 60)
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    
    # Load original training data
    train_path = os.path.join(data_dir, 'train_phonetic.csv')
    train_df = pd.read_csv(train_path)
    
    print(f"\nOriginal training data: {len(train_df)} samples")
    print(f"Label distribution:")
    print(train_df['label'].value_counts())
    
    # Create augmenter
    augmenter = CodeMixAugmenter(augment_factor=3)
    
    # Augment
    augmented_df = augmenter.augment_dataset(train_df)
    
    # Now we need to regenerate phonetic representations for augmented data
    print("\nGenerating phonetic representations for augmented data...")
    
    from src.g2p_converter import G2PConverter
    converter = G2PConverter()
    
    # Process augmented samples
    normalized_texts = []
    language_tags_list = []
    phonetic_texts = []
    
    for idx, row in tqdm(augmented_df.iterrows(), total=len(augmented_df), desc="Converting"):
        result = converter.convert(row['text'])
        normalized_texts.append(result['normalized'])
        language_tags_list.append(str(result['language_tags']))
        phonetic_texts.append(result['phonetic_text'])
    
    # Add columns
    augmented_df['normalized'] = normalized_texts
    augmented_df['language_tags'] = language_tags_list
    augmented_df['phonetic'] = phonetic_texts
    
    # Save augmented data
    aug_path = os.path.join(data_dir, 'train_augmented.csv')
    augmented_df.to_csv(aug_path, index=False, encoding='utf-8')
    
    print(f"\n✓ Augmented data saved to: {aug_path}")
    print(f"  Total samples: {len(augmented_df)}")
    print(f"  Label distribution:")
    print(augmented_df['label'].value_counts())
    
    # Show some examples
    print("\n" + "=" * 60)
    print("Sample Augmented Texts")
    print("=" * 60)
    
    # Show original vs augmented
    for label in ['positive', 'negative', 'neutral']:
        print(f"\n--- {label.upper()} ---")
        
        # Original
        orig = train_df[train_df['label'] == label].iloc[0]['text']
        print(f"Original:  {orig}")
        
        # Find augmented versions (new samples)
        new_samples = augmented_df[len(train_df):][augmented_df['label'] == label].head(2)
        for _, row in new_samples.iterrows():
            print(f"Augmented: {row['text']}")
    
    return augmented_df


# =============================================================
# TEST THE AUGMENTER
# =============================================================

if __name__ == "__main__":
    augment_and_save()