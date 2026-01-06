"""
Grapheme-to-Phoneme (G2P) Converter for Hindi-English Code-Mixed Text
======================================================================
This module converts written text to phonetic representation (IPA).

What is IPA?
- International Phonetic Alphabet
- A standard way to write how words SOUND
- Example: "kya" → /kjaː/ (k + y + long-a sound)

Why do we need this?
- "kya" and "kiya" are spelled differently
- But they SOUND the same: /kjaː/
- Phonetic representation captures the actual sound

NOTE: This version uses custom mappings (no external G2P library needed)
"""

# Import our other modules
from spelling_normalizer import SpellingNormalizer
from language_detector import LanguageDetector


class G2PConverter:
    """
    Converts Hindi-English code-mixed text to phonetic representation.
    
    Pipeline:
    1. Normalize spelling variations
    2. Detect language of each word
    3. Apply appropriate G2P rules (Hindi or English)
    4. Return phonetic transcription
    """
    
    def __init__(self):
        """Initialize all components."""
        
        print("Initializing G2P Converter...")
        print("-" * 40)
        
        # Initialize helper modules
        self.normalizer = SpellingNormalizer()
        self.detector = LanguageDetector()
        
        # Custom phoneme mappings for Hindi and English
        self.hindi_phoneme_map = self._create_hindi_phoneme_map()
        self.english_phoneme_map = self._create_english_phoneme_map()
        
        print(f"Hindi phoneme mappings: {len(self.hindi_phoneme_map)}")
        print(f"English phoneme mappings: {len(self.english_phoneme_map)}")
        print("-" * 40)
        print("G2P Converter ready!\n")
    
    def _create_hindi_phoneme_map(self):
        """
        Create phoneme mappings for common Hindi words.
        
        IPA symbols used:
        - ː = long vowel (aa sound)
        - ə = schwa (short 'a' as in "about")
        - ʰ = aspirated (breathy sound after consonant)
        - ŋ = 'ng' sound
        - ʃ = 'sh' sound
        - tʃ = 'ch' sound
        - dʒ = 'j' sound
        - ̃  = nasalized vowel
        """
        
        return {
            # Pronouns
            "main": "mɛ̃ː",
            "mein": "mẽː",
            "tu": "tuː",
            "tum": "tʊm",
            "aap": "aːp",
            "wo": "voː",
            "ye": "jeː",
            "hum": "hʊm",
            "kaun": "kɔːn",
            "kya": "kjaː",
            "kuch": "kʊtʃ",
            "koi": "koːiː",
            "sab": "səb",
            
            # Verbs - "to be"
            "hai": "hɛː",
            "hain": "hɛ̃ː",
            "ho": "hoː",
            "tha": "tʰaː",
            "thi": "tʰiː",
            "the": "tʰeː",
            "hun": "hũː",
            
            # Common verbs
            "kar": "kər",
            "karo": "kəroː",
            "karna": "kərnaː",
            "karta": "kərtaː",
            "karti": "kərtiː",
            "kiya": "kɪjaː",
            "ja": "dʒaː",
            "jao": "dʒaːoː",
            "jana": "dʒaːnaː",
            "gaya": "gəjaː",
            "gayi": "gəjiː",
            "aa": "aː",
            "aao": "aːoː",
            "aana": "aːnaː",
            "aaya": "aːjaː",
            "de": "deː",
            "do": "doː",
            "dena": "deːnaː",
            "diya": "dɪjaː",
            "le": "leː",
            "lo": "loː",
            "lena": "leːnaː",
            "liya": "lɪjaː",
            "bol": "boːl",
            "bolo": "boːloː",
            "bola": "boːlaː",
            "dekh": "deːkʰ",
            "dekho": "deːkʰoː",
            "dekha": "deːkʰaː",
            "sun": "sʊn",
            "suno": "sʊnoː",
            "suna": "sʊnaː",
            "chal": "tʃəl",
            "chalo": "tʃəloː",
            "chala": "tʃəlaː",
            "chalein": "tʃəlẽː",
            "peene": "piːneː",
            "peena": "piːnaː",
            "khana": "kʰaːnaː",
            "khatam": "xətəm",
            "mil": "mɪl",
            "mila": "mɪlaː",
            "mili": "mɪliː",
            "raha": "rəhaː",
            "rahi": "rəhiː",
            "rahe": "rəheː",
            
            # Postpositions
            "ka": "kaː",
            "ki": "kiː",
            "ke": "keː",
            "ko": "koː",
            "se": "seː",
            "pe": "peː",
            "par": "pər",
            "tak": "tək",
            
            # Conjunctions
            "aur": "ɔːr",
            "ya": "jaː",
            "lekin": "leːkɪn",
            "toh": "toː",
            "to": "toː",
            "bhi": "bʰiː",
            "phir": "pʰɪr",
            "fir": "pʰɪr",
            
            # Adverbs
            "bahut": "bəhʊt",
            "zyada": "zjaːdaː",
            "kam": "kəm",
            "thoda": "tʰoːɽaː",
            "abhi": "əbʰiː",
            "aaj": "aːdʒ",
            "kal": "kəl",
            "yahan": "jəhãː",
            "wahan": "vəhãː",
            "kahan": "kəhãː",
            "jaldi": "dʒəldiː",
            "bilkul": "bɪlkʊl",
            "sirf": "sɪrf",
            "bas": "bəs",
            
            # Adjectives
            "acha": "ətʃʰaː",
            "accha": "ətʃʰaː",
            "bura": "bʊraː",
            "bada": "bəɽaː",
            "chhota": "tʃʰoːʈaː",
            "naya": "nəjaː",
            "purana": "pʊraːnaː",
            "theek": "ʈʰiːk",
            "galat": "ɣələt",
            "sahi": "səhiː",
            "khush": "xʊʃ",
            "bakwas": "bəkvaːs",
            "mast": "məst",
            "bekar": "beːkaːr",
            
            # Nouns
            "yaar": "jaːr",
            "dost": "doːst",
            "bhai": "bʰaːiː",
            "behen": "bəhən",
            "ghar": "gʰər",
            "kaam": "kaːm",
            "paisa": "pɛːsaː",
            "waqt": "vəqt",
            "din": "dɪn",
            "raat": "raːt",
            "subah": "sʊbəh",
            "shaam": "ʃaːm",
            "chai": "tʃaːiː",
            "paani": "paːniː",
            "log": "loːg",
            "ladka": "ləɽkaː",
            "ladki": "ləɽkiː",
            
            # Question words
            "kaise": "kɛːseː",
            "kaisa": "kɛːsaː",
            "kab": "kəb",
            "kyun": "kjõː",
            "kyon": "kjõː",
            
            # Negation
            "nahi": "nəhĩː",
            "mat": "mət",
            "na": "naː",
            
            # Expressions
            "haan": "hãː",
            "ji": "dʒiː",
            "are": "əreː",
            "arre": "ərreː",
            "oye": "oːjeː",
            "sach": "sətʃ",
            "shayad": "ʃaːjəd",
            "pakka": "pəkkaː",
            
            # Time expressions
            "monday": "mənɖeː",
            "week": "viːk",
        }
    
    def _create_english_phoneme_map(self):
        """
        Create phoneme mappings for common English words.
        
        These are English words commonly used in Indian code-mixed speech.
        """
        
        return {
            # Common verbs
            "is": "ɪz",
            "am": "æm",
            "are": "ɑːr",
            "was": "wɒz",
            "were": "wɜːr",
            "be": "biː",
            "been": "biːn",
            "being": "biːɪŋ",
            "have": "hæv",
            "has": "hæz",
            "had": "hæd",
            "do": "duː",
            "does": "dʌz",
            "did": "dɪd",
            "done": "dʌn",
            "go": "goʊ",
            "going": "goʊɪŋ",
            "went": "went",
            "gone": "gɒn",
            "come": "kʌm",
            "coming": "kʌmɪŋ",
            "came": "keɪm",
            "get": "get",
            "getting": "getɪŋ",
            "got": "gɒt",
            "make": "meɪk",
            "making": "meɪkɪŋ",
            "made": "meɪd",
            "know": "noʊ",
            "knowing": "noʊɪŋ",
            "knew": "njuː",
            "known": "noʊn",
            "think": "θɪŋk",
            "thinking": "θɪŋkɪŋ",
            "thought": "θɔːt",
            "see": "siː",
            "seeing": "siːɪŋ",
            "saw": "sɔː",
            "seen": "siːn",
            "want": "wɒnt",
            "wanting": "wɒntɪŋ",
            "wanted": "wɒntɪd",
            "call": "kɔːl",
            "calling": "kɔːlɪŋ",
            "called": "kɔːld",
            "try": "traɪ",
            "trying": "traɪɪŋ",
            "tried": "traɪd",
            "need": "niːd",
            "needing": "niːdɪŋ",
            "needed": "niːdɪd",
            "feel": "fiːl",
            "feeling": "fiːlɪŋ",
            "felt": "felt",
            "work": "wɜːrk",
            "working": "wɜːrkɪŋ",
            "worked": "wɜːrkt",
            "start": "stɑːrt",
            "starting": "stɑːrtɪŋ",
            "started": "stɑːrtɪd",
            "watch": "wɒtʃ",
            "watching": "wɒtʃɪŋ",
            "watched": "wɒtʃt",
            "enjoy": "ɪndʒɔɪ",
            "enjoying": "ɪndʒɔɪɪŋ",
            "enjoyed": "ɪndʒɔɪd",
            "happen": "hæpən",
            "happening": "hæpənɪŋ",
            "happened": "hæpənd",
            "stuck": "stʌk",
            "wait": "weɪt",
            "waiting": "weɪtɪŋ",
            
            # Common nouns
            "movie": "muːviː",
            "movies": "muːviːz",
            "film": "fɪlm",
            "films": "fɪlmz",
            "song": "sɒŋ",
            "songs": "sɒŋz",
            "music": "mjuːzɪk",
            "office": "ɒfɪs",
            "college": "kɒlɪdʒ",
            "school": "skuːl",
            "class": "klɑːs",
            "meeting": "miːtɪŋ",
            "exam": "ɪgzæm",
            "exams": "ɪgzæmz",
            "test": "test",
            "phone": "foʊn",
            "message": "mesɪdʒ",
            "video": "vɪdioʊ",
            "photo": "foʊtoʊ",
            "picture": "pɪktʃər",
            "friend": "frend",
            "friends": "frendz",
            "brother": "brʌðər",
            "sister": "sɪstər",
            "mom": "mɒm",
            "dad": "dæd",
            "family": "fæmɪliː",
            "food": "fuːd",
            "coffee": "kɒfiː",
            "lunch": "lʌntʃ",
            "dinner": "dɪnər",
            "breakfast": "brekfəst",
            "time": "taɪm",
            "day": "deɪ",
            "night": "naɪt",
            "week": "wiːk",
            "month": "mʌnθ",
            "year": "jɪər",
            "job": "dʒɒb",
            "project": "prɒdʒekt",
            "report": "rɪpɔːrt",
            "presentation": "prezənteɪʃən",
            "traffic": "træfɪk",
            "car": "kɑːr",
            "bus": "bʌs",
            "train": "treɪn",
            "flight": "flaɪt",
            "party": "pɑːrtiː",
            "fun": "fʌn",
            "plan": "plæn",
            "idea": "aɪdɪə",
            "thing": "θɪŋ",
            "stuff": "stʌf",
            "place": "pleɪs",
            "city": "sɪtiː",
            "scene": "siːn",
            "experience": "ɪkspɪərɪəns",
            "service": "sɜːrvɪs",
            
            # Common adjectives
            "good": "gʊd",
            "bad": "bæd",
            "great": "greɪt",
            "best": "best",
            "worst": "wɜːrst",
            "nice": "naɪs",
            "fine": "faɪn",
            "amazing": "əmeɪzɪŋ",
            "awesome": "ɔːsəm",
            "fantastic": "fæntæstɪk",
            "wonderful": "wʌndərfʊl",
            "beautiful": "bjuːtɪfʊl",
            "happy": "hæpiː",
            "sad": "sæd",
            "angry": "æŋgriː",
            "excited": "ɪksaɪtɪd",
            "tired": "taɪərd",
            "bored": "bɔːrd",
            "boring": "bɔːrɪŋ",
            "interesting": "ɪntrəstɪŋ",
            "funny": "fʌniː",
            "serious": "sɪərɪəs",
            "important": "ɪmpɔːrtənt",
            "different": "dɪfrənt",
            "new": "njuː",
            "old": "oʊld",
            "big": "bɪg",
            "small": "smɔːl",
            "long": "lɒŋ",
            "short": "ʃɔːrt",
            "easy": "iːziː",
            "hard": "hɑːrd",
            "difficult": "dɪfɪkəlt",
            "simple": "sɪmpəl",
            "late": "leɪt",
            "early": "ɜːrliː",
            "fast": "fɑːst",
            "slow": "sloʊ",
            "busy": "bɪziː",
            "free": "friː",
            "disappointed": "dɪsəpɔɪntɪd",
            "frustrating": "frʌstreɪtɪŋ",
            "annoying": "ənɔɪɪŋ",
            
            # Common adverbs
            "very": "veriː",
            "really": "rɪəliː",
            "so": "soʊ",
            "too": "tuː",
            "also": "ɔːlsoʊ",
            "just": "dʒʌst",
            "only": "oʊnliː",
            "now": "naʊ",
            "then": "ðen",
            "today": "tədeɪ",
            "tomorrow": "təmɒroʊ",
            "yesterday": "jestərdeɪ",
            "here": "hɪər",
            "there": "ðeər",
            "always": "ɔːlweɪz",
            "never": "nevər",
            "sometimes": "sʌmtaɪmz",
            "usually": "juːʒuəliː",
            "often": "ɒfən",
            "already": "ɔːlrediː",
            "still": "stɪl",
            "yet": "jet",
            "again": "əgen",
            "finally": "faɪnəliː",
            "totally": "toʊtəliː",
            "completely": "kəmpliːtliː",
            "actually": "æktʃuəliː",
            "probably": "prɒbəbliː",
            "definitely": "defɪnɪtliː",
            
            # Pronouns
            "i": "aɪ",
            "you": "juː",
            "he": "hiː",
            "she": "ʃiː",
            "it": "ɪt",
            "we": "wiː",
            "they": "ðeɪ",
            "me": "miː",
            "him": "hɪm",
            "her": "hɜːr",
            "us": "ʌs",
            "them": "ðem",
            "my": "maɪ",
            "your": "jɔːr",
            "his": "hɪz",
            "its": "ɪts",
            "our": "aʊər",
            "their": "ðeər",
            "this": "ðɪs",
            "that": "ðæt",
            "these": "ðiːz",
            "those": "ðoʊz",
            "what": "wɒt",
            "which": "wɪtʃ",
            "who": "huː",
            "where": "weər",
            "when": "wen",
            "why": "waɪ",
            "how": "haʊ",
            "nothing": "nʌθɪŋ",
            "something": "sʌmθɪŋ",
            "everything": "evriːθɪŋ",
            
            # Prepositions and conjunctions
            "in": "ɪn",
            "on": "ɒn",
            "at": "æt",
            "to": "tuː",
            "for": "fɔːr",
            "with": "wɪð",
            "by": "baɪ",
            "from": "frɒm",
            "about": "əbaʊt",
            "after": "ɑːftər",
            "before": "bɪfɔːr",
            "and": "ænd",
            "but": "bʌt",
            "or": "ɔːr",
            "if": "ɪf",
            "because": "bɪkɒz",
            
            # Articles
            "a": "ə",
            "an": "æn",
            "the": "ðə",
            
            # Common expressions
            "ok": "oʊkeɪ",
            "okay": "oʊkeɪ",
            "yes": "jes",
            "no": "noʊ",
            "please": "pliːz",
            "thanks": "θæŋks",
            "thank": "θæŋk",
            "sorry": "sɒriː",
            "hello": "heloʊ",
            "hi": "haɪ",
            "bye": "baɪ",
            "hey": "heɪ",
            "bro": "broʊ",
            "dude": "duːd",
            "man": "mæn",
            "guys": "gaɪz",
            "back": "bæk",
            "later": "leɪtər",
            "will": "wɪl",
        }
    
    def get_phoneme(self, word, language):
        """
        Get phonetic representation for a single word.
        
        Args:
            word (str): The word to convert
            language (str): "HI" for Hindi, "EN" for English
            
        Returns:
            str: IPA phonetic representation
        """
        word_lower = word.lower().strip()
        
        # Remove common punctuation
        word_clean = word_lower.strip(".,!?;:'\"")
        
        # For Hindi words: use Hindi mapping
        if language == "HI":
            if word_clean in self.hindi_phoneme_map:
                return self.hindi_phoneme_map[word_clean]
            else:
                return self._approximate_phoneme(word_clean)
        
        # For English words: use English mapping
        elif language == "EN":
            if word_clean in self.english_phoneme_map:
                return self.english_phoneme_map[word_clean]
            else:
                return self._approximate_phoneme(word_clean)
        
        # For unknown: check both mappings, then approximate
        else:
            if word_clean in self.hindi_phoneme_map:
                return self.hindi_phoneme_map[word_clean]
            elif word_clean in self.english_phoneme_map:
                return self.english_phoneme_map[word_clean]
            else:
                return self._approximate_phoneme(word_clean)
    
    def _approximate_phoneme(self, word):
        """
        Generate approximate phonemes for unknown words.
        
        This uses simple letter-to-sound rules.
        """
        # Basic mapping for Roman letters to IPA sounds
        letter_map = {
            'a': 'ə', 'aa': 'aː', 'i': 'ɪ', 'ii': 'iː', 'ee': 'iː',
            'u': 'ʊ', 'uu': 'uː', 'oo': 'uː', 'e': 'eː', 'o': 'oː',
            'ai': 'aɪ', 'au': 'aʊ', 'ou': 'aʊ', 'oi': 'ɔɪ',
            'k': 'k', 'kh': 'kʰ', 'g': 'g', 'gh': 'gʰ',
            'ch': 'tʃ', 'chh': 'tʃʰ', 'j': 'dʒ', 'jh': 'dʒʰ',
            't': 't', 'th': 'tʰ', 'd': 'd', 'dh': 'dʰ',
            'n': 'n', 'p': 'p', 'ph': 'f', 'f': 'f',
            'b': 'b', 'bh': 'bʰ', 'm': 'm',
            'y': 'j', 'r': 'r', 'l': 'l', 'v': 'v', 'w': 'w',
            's': 's', 'sh': 'ʃ', 'h': 'h', 'z': 'z',
            'c': 'k', 'q': 'k', 'x': 'ks',
            'ng': 'ŋ',
        }
        
        result = ""
        i = 0
        while i < len(word):
            # Check for two-letter combinations first
            if i < len(word) - 1:
                two_char = word[i:i+2]
                if two_char in letter_map:
                    result += letter_map[two_char]
                    i += 2
                    continue
            
            # Single character
            char = word[i]
            if char in letter_map:
                result += letter_map[char]
            else:
                result += char
            i += 1
        
        return result
    
    def convert(self, text):
        """
        Convert text to phonetic representation.
        
        This is the main method that combines all steps:
        1. Normalize spelling
        2. Detect language
        3. Get phonemes
        
        Args:
            text (str): Input text (Hindi-English code-mixed)
            
        Returns:
            dict: Contains original, normalized, language tags, and phonetic output
        """
        # Step 1: Normalize spelling
        normalized = self.normalizer.normalize_text(text)
        
        # Step 2: Detect language for each word
        word_langs = self.detector.detect_text(normalized)
        
        # Step 3: Get phonemes for each word
        phonemes = []
        for word, lang in word_langs:
            phoneme = self.get_phoneme(word, lang)
            phonemes.append(phoneme)
        
        # Combine phonemes
        phonetic_output = " ".join(phonemes)
        
        return {
            "original": text,
            "normalized": normalized,
            "words": [w for w, l in word_langs],
            "language_tags": [l for w, l in word_langs],
            "phonemes": phonemes,
            "phonetic_text": f"/{phonetic_output}/"
        }
    
    def convert_simple(self, text):
        """
        Simple conversion - just return phonetic text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Phonetic representation
        """
        result = self.convert(text)
        return result["phonetic_text"]


# =============================================================
# TEST THE G2P CONVERTER
# =============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing G2P Converter")
    print("=" * 60)
    
    # Create converter
    converter = G2PConverter()
    
    # Test sentences from our dataset
    test_sentences = [
        "yaar ye movie toh bahut amazing thi",
        "kya bakwas service hai totally disappointed",
        "chai peene chalein kya after meeting",
        "finally exam khatam feeling so happy",
        "traffic mein stuck hun bahut frustrating",
    ]
    
    print("\n" + "=" * 60)
    print("G2P Conversion Results")
    print("=" * 60 + "\n")
    
    for sentence in test_sentences:
        result = converter.convert(sentence)
        
        print(f"Original:    {result['original']}")
        print(f"Normalized:  {result['normalized']}")
        print(f"Languages:   {result['language_tags']}")
        print(f"Phonetic:    {result['phonetic_text']}")
        print("-" * 60)
    
    # Show detailed breakdown for one example
    print("\n" + "=" * 60)
    print("Detailed Breakdown Example")
    print("=" * 60 + "\n")
    
    example = "yaar ye movie toh bahut amazing thi"
    result = converter.convert(example)
    
    print(f"Sentence: {example}\n")
    print(f"{'Word':<12} {'Language':<8} {'Phoneme':<15}")
    print("-" * 35)
    
    for word, lang, phoneme in zip(result['words'], result['language_tags'], result['phonemes']):
        print(f"{word:<12} {lang:<8} {phoneme:<15}")