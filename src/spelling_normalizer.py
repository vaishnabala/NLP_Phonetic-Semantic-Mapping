"""
Spelling Normalizer for Hindi-English Code-Mixed Text
======================================================
This module standardizes common spelling variations in Romanized Hindi.

Example:
    "kiya" → "kya"
    "bohot" → "bahut"
    "accha" → "acha"
"""


class SpellingNormalizer:
    """
    Normalizes spelling variations in Romanized Hindi text.
    
    Why do we need this?
    - Hindi written in English letters has no standard spelling
    - People write the same word differently
    - AI models get confused by these variations
    """
    
    def __init__(self):
        """Initialize the normalizer with a mapping dictionary."""
        
        # Dictionary: variation → standard form
        # Key = how people might write it
        # Value = standard form we'll use
        
        self.spelling_map = {
            # "what" variations
            "kiya": "kya",
            "kyaa": "kya",
            "kia": "kya",
            
            # "very/much" variations
            "bohot": "bahut",
            "boht": "bahut",
            "bhot": "bahut",
            "bhut": "bahut",
            
            # "good" variations
            "accha": "acha",
            "achha": "acha",
            "achchha": "acha",
            "acchha": "acha",
            
            # "okay/fine" variations
            "thik": "theek",
            "theik": "theek",
            "thek": "theek",
            "tik": "theek",
            
            # "is/are" variations
            "he": "hai",
            "h": "hai",
            "hain": "hai",
            "hen": "hai",
            
            # "no/not" variations
            "nai": "nahi",
            "nahin": "nahi",
            "nhi": "nahi",
            "ni": "nahi",
            
            # "in/inside" variations
            "me": "mein",
            "mai": "mein",
            "main": "mein",
            
            # "friend" variations
            "yar": "yaar",
            "yr": "yaar",
            
            # "brother" variations  
            "bhai": "bhai",
            "bhi": "bhai",
            
            # "was" variations
            "tha": "tha",
            "thaa": "tha",
            
            # "and" variations
            "or": "aur",
            "ar": "aur",
            
            # "but" variations
            "par": "lekin",
            "magar": "lekin",
            
            # "this" variations
            "yeh": "ye",
            "y": "ye",
            
            # "that" variations
            "woh": "wo",
            "voh": "wo",
            "vo": "wo",
            
            # "today" variations
            "aaj": "aaj",
            "aj": "aaj",
            
            # "now" variations
            "abhi": "abhi",
            "abi": "abhi",
            
            # "please" variations
            "plz": "please",
            "pls": "please",
        }
        
        print(f"SpellingNormalizer initialized with {len(self.spelling_map)} mappings")
    
    def normalize_word(self, word):
        """
        Normalize a single word.
        
        Args:
            word (str): Input word (possibly misspelled)
            
        Returns:
            str: Normalized word
            
        Example:
            normalize_word("bohot") → "bahut"
        """
        # Convert to lowercase for matching
        word_lower = word.lower().strip()
        
        # Check if word has a known variation
        if word_lower in self.spelling_map:
            return self.spelling_map[word_lower]
        
        # If no variation found, return original (lowercase)
        return word_lower
    
    def normalize_text(self, text):
        """
        Normalize an entire sentence.
        
        Args:
            text (str): Input sentence
            
        Returns:
            str: Normalized sentence
            
        Example:
            normalize_text("kiya scene he yar") → "kya scene hai yaar"
        """
        # Split text into words
        words = text.split()
        
        # Normalize each word
        normalized_words = [self.normalize_word(word) for word in words]
        
        # Join back into sentence
        return " ".join(normalized_words)
    
    def add_mapping(self, variation, standard):
        """
        Add a new spelling variation to the dictionary.
        
        Args:
            variation (str): The misspelled/alternate form
            standard (str): The standard form to map to
            
        Example:
            add_mapping("kyu", "kyon")
        """
        self.spelling_map[variation.lower()] = standard.lower()
        print(f"Added mapping: '{variation}' → '{standard}'")
    
    def get_stats(self):
        """Return statistics about the normalizer."""
        return {
            "total_mappings": len(self.spelling_map),
            "unique_standard_forms": len(set(self.spelling_map.values()))
        }


# =============================================================
# TEST THE NORMALIZER
# =============================================================

if __name__ == "__main__":
    # This code runs when you execute this file directly
    
    print("=" * 50)
    print("Testing Spelling Normalizer")
    print("=" * 50)
    
    # Create normalizer
    normalizer = SpellingNormalizer()
    
    # Test sentences (with intentional spelling variations)
    test_sentences = [
        "kiya scene he yar",
        "bohot accha movie thi",
        "me office me hun abhi",
        "yeh thik nai he",
        "woh bohot boring tha",
    ]
    
    print("\n--- Normalization Results ---\n")
    
    for sentence in test_sentences:
        normalized = normalizer.normalize_text(sentence)
        print(f"Original:   {sentence}")
        print(f"Normalized: {normalized}")
        print("-" * 40)
    
    # Show stats
    stats = normalizer.get_stats()
    print(f"\nTotal mappings: {stats['total_mappings']}")
    print(f"Unique standard forms: {stats['unique_standard_forms']}")