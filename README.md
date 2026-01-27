# NLP_Phonetic-Semantic Mapping
Research Project

TITLE: Phonetic-Semantic Mapping for Sentiment Analysis in Hindi-English 
       Code-Mixed Text: A Prosody-Enhanced Contrastive Learning Approach

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
mappings; (2) A spelling normalization module addressing 47+ common spelling 
variations; (3) A prosody feature extractor inspired by Indian classical 
music theory, capturing rhythmic and tonal patterns in 24-dimensional vectors; 
and (4) A contrastive learning framework using triplet loss to learn 
semantically meaningful phonetic embeddings.

We evaluate our approach on a dataset of 210 code-mixed samples across three 
sentiment classes (positive, negative, neutral). Our phonetic-prosody model 
achieves 46.67% accuracy, outperforming random baseline (33.33%) by 13.34 
percentage points. While traditional TF-IDF approaches (76.67%) and mBERT 
(66.67%) achieve higher accuracy on this limited dataset, our work establishes 
a foundation for phonetic-aware processing of code-mixed text. The phonetic 
approach shows particular promise for handling spelling variations and could 
benefit from larger training datasets.

KEYWORDS: Code-mixing, Hindi-English, Phonetic embeddings, Sentiment analysis, 
          Prosody features, Contrastive learning, Low-resource NLP
