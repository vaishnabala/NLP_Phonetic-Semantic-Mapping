"""
Prosody Feature Extractor
==========================
Extracts rhythm, stress, and intonation patterns from phonetic text.

Inspired by Indian Classical Music concepts:
- Taal (rhythm patterns) → Rhythm features
- Gamak (emphasis) → Stress features  
- Meend (glides) → Vowel length features

These features help capture emotional nuances in speech that
pure phonetic representation might miss.
"""

import numpy as np
import re


class ProsodyExtractor:
    """
    Extracts prosody features from phonetic transcriptions.
    
    Features extracted:
    1. Vowel length patterns (long vs short)
    2. Stress patterns (estimated from vowel length)
    3. Rhythm features (syllable patterns)
    4. Phoneme density (complexity measure)
    """
    
    def __init__(self, feature_dim=24):
        """
        Initialize prosody extractor.
        
        Args:
            feature_dim (int): Total dimension of prosody features
        """
        self.feature_dim = feature_dim
        
        # Long vowels in IPA (indicated by ː)
        self.long_vowel_marker = 'ː'
        
        # Vowels in IPA
        self.vowels = set('aeiouəɛɪɔʊæʌɑɜ')
        
        # Stress-indicating characters
        self.stress_markers = set('ˈˌ')
        
        # Nasal markers (indicate emphasis in Hindi)
        self.nasal_markers = set('̃ŋɲɳ')
        
        # Aspirated consonants (common in Hindi, convey intensity)
        self.aspirated = set('ʰ')
        
        print(f"ProsodyExtractor initialized with {feature_dim} features")
    
    def extract(self, phonetic_text):
        """
        Extract prosody features from phonetic text.
        
        Args:
            phonetic_text (str): IPA phonetic transcription
            
        Returns:
            numpy.ndarray: Feature vector of shape (feature_dim,)
        """
        # Clean the text
        text = phonetic_text.strip('/')
        phonemes = text.split()
        
        # Extract different feature groups
        length_features = self._extract_length_features(phonemes)      # 6 features
        stress_features = self._extract_stress_features(phonemes)      # 6 features
        rhythm_features = self._extract_rhythm_features(phonemes)      # 6 features
        intensity_features = self._extract_intensity_features(phonemes) # 6 features
        
        # Combine all features
        all_features = np.concatenate([
            length_features,
            stress_features,
            rhythm_features,
            intensity_features
        ])
        
        # Ensure correct dimension
        if len(all_features) < self.feature_dim:
            all_features = np.pad(all_features, (0, self.feature_dim - len(all_features)))
        elif len(all_features) > self.feature_dim:
            all_features = all_features[:self.feature_dim]
        
        return all_features.astype(np.float32)
    
    def _extract_length_features(self, phonemes):
        """
        Extract vowel length features.
        
        Long vowels (with ː) often indicate:
        - Emphasis
        - Emotional intensity
        - Important words
        
        Like holding a note longer on Veena for emphasis! 🎵
        """
        features = np.zeros(6)
        
        if not phonemes:
            return features
        
        total_phonemes = len(phonemes)
        long_count = 0
        short_count = 0
        
        for phoneme in phonemes:
            if self.long_vowel_marker in phoneme:
                long_count += 1
            else:
                # Check if has any vowel (short vowel)
                if any(v in phoneme for v in self.vowels):
                    short_count += 1
        
        # Feature 0: Ratio of long vowels
        features[0] = long_count / max(total_phonemes, 1)
        
        # Feature 1: Ratio of short vowels
        features[1] = short_count / max(total_phonemes, 1)
        
        # Feature 2: Long to short ratio
        features[2] = long_count / max(short_count, 1)
        
        # Feature 3: Total vowel density
        features[3] = (long_count + short_count) / max(total_phonemes, 1)
        
        # Feature 4: Position of first long vowel (normalized)
        for i, phoneme in enumerate(phonemes):
            if self.long_vowel_marker in phoneme:
                features[4] = i / max(total_phonemes, 1)
                break
        
        # Feature 5: Position of last long vowel (normalized)
        for i, phoneme in enumerate(reversed(phonemes)):
            if self.long_vowel_marker in phoneme:
                features[5] = (total_phonemes - 1 - i) / max(total_phonemes, 1)
                break
        
        return features
    
    def _extract_stress_features(self, phonemes):
        """
        Extract stress pattern features.
        
        In Hindi-English code-mixing:
        - Emphasized words often have longer vowels
        - Nasalized vowels (̃) indicate emotional weight
        - Word-initial position often carries stress
        
        Like the 'sam' (first beat) in a taal! 🎵
        """
        features = np.zeros(6)
        
        if not phonemes:
            return features
        
        total = len(phonemes)
        
        # Estimate stress based on phoneme characteristics
        stress_scores = []
        
        for phoneme in phonemes:
            score = 0
            
            # Long vowels = higher stress
            if self.long_vowel_marker in phoneme:
                score += 2
            
            # Nasalization = emphasis
            if any(m in phoneme for m in self.nasal_markers):
                score += 1
            
            # Aspiration = intensity
            if any(a in phoneme for a in self.aspirated):
                score += 1
            
            # Length of phoneme representation
            score += len(phoneme) * 0.1
            
            stress_scores.append(score)
        
        if stress_scores:
            # Feature 0: Mean stress
            features[0] = np.mean(stress_scores)
            
            # Feature 1: Max stress
            features[1] = np.max(stress_scores)
            
            # Feature 2: Stress variance (variation in emphasis)
            features[2] = np.var(stress_scores)
            
            # Feature 3: Position of max stress (normalized)
            features[3] = np.argmax(stress_scores) / max(total, 1)
            
            # Feature 4: First half vs second half stress
            mid = len(stress_scores) // 2
            first_half = np.mean(stress_scores[:mid]) if mid > 0 else 0
            second_half = np.mean(stress_scores[mid:]) if mid < len(stress_scores) else 0
            features[4] = first_half - second_half
            
            # Feature 5: Stress contour (rising=1, falling=-1, flat=0)
            if len(stress_scores) >= 2:
                first_third = np.mean(stress_scores[:len(stress_scores)//3+1])
                last_third = np.mean(stress_scores[-(len(stress_scores)//3+1):])
                features[5] = np.sign(last_third - first_third)
        
        return features
    
    def _extract_rhythm_features(self, phonemes):
        """
        Extract rhythm pattern features.
        
        Rhythm in speech is like taal in music:
        - Pattern of long and short syllables
        - Regularity vs irregularity
        - Speed/density of sounds
        
        Think of it like: दा-दिन-दा (da-din-da) patterns! 🎵
        """
        features = np.zeros(6)
        
        if not phonemes:
            return features
        
        # Create binary pattern: 1 for long, 0 for short
        pattern = []
        for phoneme in phonemes:
            if self.long_vowel_marker in phoneme:
                pattern.append(1)
            else:
                pattern.append(0)
        
        if pattern:
            # Feature 0: Pattern density (ratio of 1s)
            features[0] = np.mean(pattern)
            
            # Feature 1: Pattern changes (alternations)
            changes = sum(1 for i in range(1, len(pattern)) if pattern[i] != pattern[i-1])
            features[1] = changes / max(len(pattern) - 1, 1)
            
            # Feature 2: Longest run of same value
            max_run = 1
            current_run = 1
            for i in range(1, len(pattern)):
                if pattern[i] == pattern[i-1]:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 1
            features[2] = max_run / max(len(pattern), 1)
            
            # Feature 3: Regularity (autocorrelation-like)
            if len(pattern) >= 4:
                # Check if pattern repeats
                half = len(pattern) // 2
                first_half = pattern[:half]
                second_half = pattern[half:half*2]
                if len(first_half) == len(second_half):
                    matches = sum(1 for a, b in zip(first_half, second_half) if a == b)
                    features[3] = matches / max(len(first_half), 1)
            
            # Feature 4: Starting pattern (first 3 elements encoded)
            start_pattern = pattern[:3] if len(pattern) >= 3 else pattern + [0] * (3 - len(pattern))
            features[4] = start_pattern[0] * 0.5 + (start_pattern[1] if len(start_pattern) > 1 else 0) * 0.3 + (start_pattern[2] if len(start_pattern) > 2 else 0) * 0.2
            
            # Feature 5: Ending pattern (last 3 elements encoded)
            end_pattern = pattern[-3:] if len(pattern) >= 3 else [0] * (3 - len(pattern)) + pattern
            features[5] = end_pattern[-1] * 0.5 + (end_pattern[-2] if len(end_pattern) > 1 else 0) * 0.3 + (end_pattern[-3] if len(end_pattern) > 2 else 0) * 0.2
        
        return features
    
    def _extract_intensity_features(self, phonemes):
        """
        Extract intensity/energy features.
        
        Intensity markers in phonetic text:
        - Aspirated consonants (ʰ) = forceful
        - Nasalization = emotional
        - Certain consonant clusters = emphasis
        
        Like the intensity of a 'zamzama' in music! 🎵
        """
        features = np.zeros(6)
        
        if not phonemes:
            return features
        
        total = len(phonemes)
        
        # Count intensity markers
        aspirated_count = 0
        nasal_count = 0
        complex_count = 0  # Phonemes with multiple characters
        
        for phoneme in phonemes:
            if any(a in phoneme for a in self.aspirated):
                aspirated_count += 1
            if any(n in phoneme for n in self.nasal_markers):
                nasal_count += 1
            if len(phoneme) > 3:
                complex_count += 1
        
        # Feature 0: Aspiration density
        features[0] = aspirated_count / max(total, 1)
        
        # Feature 1: Nasalization density
        features[1] = nasal_count / max(total, 1)
        
        # Feature 2: Complexity density
        features[2] = complex_count / max(total, 1)
        
        # Feature 3: Average phoneme length (complexity measure)
        avg_length = np.mean([len(p) for p in phonemes])
        features[3] = avg_length / 10  # Normalize
        
        # Feature 4: Total phoneme count (utterance length)
        features[4] = total / 20  # Normalize assuming max ~20 phonemes
        
        # Feature 5: Combined intensity score
        features[5] = (aspirated_count + nasal_count * 0.5 + complex_count * 0.3) / max(total, 1)
        
        return features
    
    def extract_batch(self, phonetic_texts):
        """
        Extract features for a batch of texts.
        
        Args:
            phonetic_texts (list): List of phonetic strings
            
        Returns:
            numpy.ndarray: Feature matrix of shape (batch_size, feature_dim)
        """
        features = [self.extract(text) for text in phonetic_texts]
        return np.stack(features)
    
    def get_feature_names(self):
        """Return names of all features for interpretability."""
        return [
            # Length features
            "long_vowel_ratio", "short_vowel_ratio", "long_short_ratio",
            "vowel_density", "first_long_pos", "last_long_pos",
            # Stress features
            "mean_stress", "max_stress", "stress_variance",
            "max_stress_pos", "stress_balance", "stress_contour",
            # Rhythm features
            "pattern_density", "alternation_rate", "max_run_ratio",
            "regularity", "start_pattern", "end_pattern",
            # Intensity features
            "aspiration_density", "nasal_density", "complexity_density",
            "avg_phoneme_length", "utterance_length", "intensity_score",
        ]


# =============================================================
# TEST THE PROSODY EXTRACTOR
# =============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Prosody Feature Extractor")
    print("=" * 60)
    
    # Create extractor
    extractor = ProsodyExtractor(feature_dim=24)
    
    # Test samples with different emotional content
    test_samples = [
        # Positive - typically more long vowels, musical rhythm
        ("/jaːr jeː muːviː toː bəhʊt əmeɪzɪŋ tʰiː/", "positive"),
        
        # Negative - more aspirated sounds, intense
        ("/kjaː bəkvaːs hɛː toʊtəliː dɪsəpɔɪntɪd/", "negative"),
        
        # Neutral - balanced, less extreme features
        ("/tʃaːiː piːneː tʃəlẽː kjaː aːftər miːtɪŋ/", "neutral"),
    ]
    
    print("\n--- Feature Extraction Results ---\n")
    
    feature_names = extractor.get_feature_names()
    
    for phonetic, sentiment in test_samples:
        print(f"Sentiment: {sentiment.upper()}")
        print(f"Phonetic:  {phonetic}")
        
        # Extract features
        features = extractor.extract(phonetic)
        
        print(f"Features shape: {features.shape}")
        print(f"\nKey features:")
        
        # Show most important features
        important_indices = [0, 1, 6, 7, 12, 18, 23]  # Selected important features
        for idx in important_indices:
            if idx < len(feature_names):
                print(f"  {feature_names[idx]:<20}: {features[idx]:.4f}")
        
        print("-" * 60)
    
    # Compare features across sentiments
    print("\n--- Feature Comparison Across Sentiments ---\n")
    
    all_features = {}
    for phonetic, sentiment in test_samples:
        all_features[sentiment] = extractor.extract(phonetic)
    
    # Compare specific features
    compare_features = [
        (0, "long_vowel_ratio"),
        (6, "mean_stress"),
        (18, "aspiration_density"),
        (23, "intensity_score"),
    ]
    
    print(f"{'Feature':<20} {'Positive':>10} {'Negative':>10} {'Neutral':>10}")
    print("-" * 52)
    
    for idx, name in compare_features:
        pos_val = all_features['positive'][idx]
        neg_val = all_features['negative'][idx]
        neu_val = all_features['neutral'][idx]
        print(f"{name:<20} {pos_val:>10.4f} {neg_val:>10.4f} {neu_val:>10.4f}")
    
    print("\n✓ Prosody extractor ready!")