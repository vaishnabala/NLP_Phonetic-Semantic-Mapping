"""
Language Detector for Hindi-English Code-Mixed Text
====================================================
This module identifies whether each word is Hindi or English.

Example:
    "yaar ye movie toh amazing thi"
    → [HI, HI, EN, HI, EN, HI]
"""


class LanguageDetector:
    """
    Detects language (Hindi/English) for each word in code-mixed text.
    
    How it works:
    1. Check if word is in Hindi word list → tag as HI
    2. Check if word is in English word list → tag as EN
    3. If unknown, use simple rules to guess
    """
    
    def __init__(self):
        """Initialize with Hindi and English word lists."""
        
        # Common Hindi words (Romanized)
        # These are words that are definitely Hindi
        self.hindi_words = {
            # Pronouns
            "main", "mein", "me", "tu", "tum", "aap", "wo", "woh", "ye", "yeh",
            "hum", "ham", "kaun", "kya", "kuch", "koi", "sab", "sabhi",
            
            # Verbs (common forms)
            "hai", "hain", "he", "ho", "tha", "thi", "the", "hun", "hoon",
            "kar", "karo", "karna", "karta", "karti", "kiya", "ki", "kiye",
            "ja", "jao", "jana", "jata", "jati", "gaya", "gayi", "gaye",
            "aa", "aao", "aana", "aata", "aati", "aaya", "aayi", "aaye",
            "de", "do", "dena", "deta", "deti", "diya", "di", "diye",
            "le", "lo", "lena", "leta", "leti", "liya", "li", "liye",
            "bol", "bolo", "bolna", "bolta", "bolti", "bola", "boli", "bole",
            "dekh", "dekho", "dekhna", "dekhta", "dekhti", "dekha", "dekhi",
            "sun", "suno", "sunna", "sunta", "sunti", "suna", "suni",
            "reh", "raho", "rehna", "rehta", "rehti", "raha", "rahi", "rahe",
            "mil", "milo", "milna", "milta", "milti", "mila", "mili", "mile",
            "chal", "chalo", "chalna", "chalta", "chalti", "chala", "chali", "chale",
            "peena", "peeta", "peeti", "piya", "piyo", "pee",
            "khana", "khata", "khati", "khaya", "khao", "kha",
            
            # Postpositions
            "ka", "ki", "ke", "ko", "se", "pe", "par", "tak", "mein",
            "ke liye", "ke baad", "ke pehle", "ke saath",
            
            # Conjunctions
            "aur", "or", "ya", "lekin", "par", "magar", "toh", "to", "bhi",
            "kyunki", "isliye", "agar", "jab", "tab", "phir", "fir",
            
            # Adverbs
            "bahut", "bohot", "boht", "bhot", "zyada", "kam", "thoda",
            "abhi", "aaj", "kal", "parso", "kabhi", "hamesha", "aksar",
            "yahan", "wahan", "idhar", "udhar", "kahan", "jahan",
            "jaldi", "dheere", "achanak", "bilkul", "sirf", "bas",
            
            # Adjectives
            "acha", "accha", "achha", "bura", "bada", "chhota", "lamba",
            "naya", "purana", "theek", "thik", "galat", "sahi", "khush",
            "dukhi", "udaas", "boring", "bakwas", "mast", "zabardast",
            "bekar", "shandar", "kamaal", "lajawab",
            
            # Nouns (common)
            "yaar", "yar", "dost", "bhai", "behen", "maa", "papa", "ghar",
            "kaam", "paisa", "waqt", "time", "din", "raat", "subah", "shaam",
            "chai", "khana", "paani", "log", "ladka", "ladki", "baccha",
            
            # Question words
            "kya", "kaise", "kaisa", "kaisi", "kab", "kahan", "kyun", "kyon",
            
            # Negation
            "nahi", "nai", "nahin", "nhi", "mat", "na",
            
            # Numbers
            "ek", "do", "teen", "char", "paanch", "chheh", "saat", "aath", "nau", "das",
            
            # Expressions
            "haan", "han", "ji", "are", "arre", "oye", "hey", "accha",
            "sach", "jhooth", "pakka", "shayad",
        }
        
        # Common English words
        # These are words commonly used in Indian English conversations
        self.english_words = {
            # Common verbs
            "is", "am", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "done",
            "go", "going", "went", "gone", "come", "coming", "came",
            "get", "getting", "got", "make", "making", "made",
            "know", "knowing", "knew", "known", "think", "thinking", "thought",
            "see", "seeing", "saw", "seen", "want", "wanting", "wanted",
            "use", "using", "used", "find", "finding", "found",
            "give", "giving", "gave", "given", "tell", "telling", "told",
            "call", "calling", "called", "try", "trying", "tried",
            "need", "needing", "needed", "feel", "feeling", "felt",
            "become", "becoming", "became", "leave", "leaving", "left",
            "work", "working", "worked", "start", "starting", "started",
            "watch", "watching", "watched", "enjoy", "enjoying", "enjoyed",
            
            # Common nouns
            "movie", "movies", "film", "films", "song", "songs", "music",
            "office", "college", "school", "class", "meeting", "exam", "test",
            "phone", "call", "message", "video", "photo", "picture",
            "friend", "friends", "brother", "sister", "mom", "dad", "family",
            "food", "coffee", "lunch", "dinner", "breakfast",
            "time", "day", "night", "week", "month", "year",
            "work", "job", "project", "report", "presentation",
            "traffic", "car", "bus", "train", "flight",
            "party", "fun", "plan", "idea", "thing", "stuff",
            "place", "city", "country", "world",
            
            # Common adjectives
            "good", "bad", "great", "best", "worst", "nice", "fine",
            "amazing", "awesome", "fantastic", "wonderful", "beautiful",
            "happy", "sad", "angry", "excited", "tired", "bored", "boring",
            "interesting", "funny", "serious", "important", "different",
            "new", "old", "big", "small", "long", "short",
            "easy", "hard", "difficult", "simple", "complicated",
            "true", "false", "real", "fake", "sure", "right", "wrong",
            "late", "early", "fast", "slow", "busy", "free",
            "disappointed", "frustrating", "annoying",
            
            # Common adverbs
            "very", "really", "so", "too", "also", "just", "only",
            "now", "then", "today", "tomorrow", "yesterday",
            "here", "there", "where", "when", "how", "why", "what",
            "always", "never", "sometimes", "usually", "often",
            "already", "still", "yet", "again", "finally",
            "well", "badly", "quickly", "slowly", "totally", "completely",
            
            # Pronouns
            "i", "you", "he", "she", "it", "we", "they",
            "me", "him", "her", "us", "them",
            "my", "your", "his", "its", "our", "their",
            "this", "that", "these", "those",
            
            # Prepositions
            "in", "on", "at", "to", "for", "with", "by", "from",
            "about", "after", "before", "between", "under", "over",
            
            # Conjunctions
            "and", "but", "or", "if", "because", "although", "while",
            
            # Articles
            "a", "an", "the",
            
            # Common expressions
            "ok", "okay", "yes", "no", "please", "thanks", "thank",
            "sorry", "hello", "hi", "bye", "hey",
            "bro", "dude", "man", "guys",
        }
        
        print(f"LanguageDetector initialized:")
        print(f"  - Hindi words: {len(self.hindi_words)}")
        print(f"  - English words: {len(self.english_words)}")
    
    def detect_word(self, word):
        """
        Detect language of a single word.
        
        Args:
            word (str): Input word
            
        Returns:
            str: "HI" for Hindi, "EN" for English, "UNK" for unknown
        """
        word_lower = word.lower().strip()
        
        # Remove common punctuation
        word_clean = word_lower.strip(".,!?;:'\"")
        
        # Check Hindi first (priority for code-mixed)
        if word_clean in self.hindi_words:
            return "HI"
        
        # Check English
        if word_clean in self.english_words:
            return "EN"
        
        # Unknown - use heuristics
        return self._guess_language(word_clean)
    
    def _guess_language(self, word):
        """
        Guess language for unknown words using simple rules.
        
        Rules:
        - Words ending in 'ing', 'ed', 'ly', 'tion' → English
        - Words with 'aa', 'ee', 'oo' patterns → likely Hindi
        - Default to English (since base words are often English)
        """
        # English patterns
        english_endings = ['ing', 'ed', 'ly', 'tion', 'ness', 'ment', 'able', 'ible']
        for ending in english_endings:
            if word.endswith(ending):
                return "EN"
        
        # Hindi patterns (doubled vowels common in Romanized Hindi)
        hindi_patterns = ['aa', 'ee', 'oo', 'ii', 'uu']
        for pattern in hindi_patterns:
            if pattern in word:
                return "HI"
        
        # Default to unknown
        return "UNK"
    
    def detect_text(self, text):
        """
        Detect language for each word in a sentence.
        
        Args:
            text (str): Input sentence
            
        Returns:
            list: List of tuples (word, language_tag)
            
        Example:
            detect_text("yaar ye movie amazing thi")
            → [("yaar", "HI"), ("ye", "HI"), ("movie", "EN"), 
               ("amazing", "EN"), ("thi", "HI")]
        """
        words = text.split()
        results = []
        
        for word in words:
            lang = self.detect_word(word)
            results.append((word, lang))
        
        return results
    
    def get_language_tags(self, text):
        """
        Get just the language tags as a list.
        
        Args:
            text (str): Input sentence
            
        Returns:
            list: List of language tags
            
        Example:
            get_language_tags("yaar ye movie amazing thi")
            → ["HI", "HI", "EN", "EN", "HI"]
        """
        results = self.detect_text(text)
        return [lang for word, lang in results]
    
    def get_language_summary(self, text):
        """
        Get summary statistics for a sentence.
        
        Returns:
            dict: Count of Hindi, English, and Unknown words
        """
        tags = self.get_language_tags(text)
        return {
            "hindi_count": tags.count("HI"),
            "english_count": tags.count("EN"),
            "unknown_count": tags.count("UNK"),
            "total_words": len(tags),
            "hindi_percentage": round(tags.count("HI") / len(tags) * 100, 1) if tags else 0
        }
    
    def add_hindi_word(self, word):
        """Add a word to Hindi dictionary."""
        self.hindi_words.add(word.lower())
        print(f"Added '{word}' to Hindi words")
    
    def add_english_word(self, word):
        """Add a word to English dictionary."""
        self.english_words.add(word.lower())
        print(f"Added '{word}' to English words")


# =============================================================
# TEST THE LANGUAGE DETECTOR
# =============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Language Detector")
    print("=" * 60)
    
    # Create detector
    detector = LanguageDetector()
    
    # Test sentences from our dataset
    test_sentences = [
        "yaar ye movie toh bahut amazing thi",
        "kya bakwas service hai totally disappointed",
        "chai peene chalein kya after meeting",
        "finally exam khatam feeling so happy",
        "traffic mein stuck hun bahut frustrating",
    ]
    
    print("\n--- Language Detection Results ---\n")
    
    for sentence in test_sentences:
        print(f"Sentence: {sentence}")
        
        # Get word-by-word detection
        results = detector.detect_text(sentence)
        
        # Display nicely
        words = [f"{word}" for word, lang in results]
        tags = [f"{lang}" for word, lang in results]
        
        print(f"Words:    {' | '.join(words)}")
        print(f"Tags:     {' | '.join(tags)}")
        
        # Get summary
        summary = detector.get_language_summary(sentence)
        print(f"Summary:  HI={summary['hindi_count']}, EN={summary['english_count']}, "
              f"Hindi%={summary['hindi_percentage']}%")
        print("-" * 60)