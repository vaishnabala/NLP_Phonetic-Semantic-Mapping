"""
Dataset Processor for Phonetic Conversion
==========================================
This script processes all our code-mixed data through the G2P pipeline
and saves the results with phonetic representations.

Output columns:
- text: Original text
- label: Sentiment label
- normalized: Spelling-normalized text
- language_tags: Language tag for each word
- phonetic: IPA phonetic representation
"""

import pandas as pd
import os
import sys
from tqdm import tqdm

# Import our G2P converter
from g2p_converter import G2PConverter


def process_dataset(input_path, output_path, converter):
    """
    Process a single CSV file and add phonetic columns.
    
    Args:
        input_path (str): Path to input CSV
        output_path (str): Path to save processed CSV
        converter (G2PConverter): Our G2P converter instance
    
    Returns:
        pd.DataFrame: Processed dataframe
    """
    print(f"\nProcessing: {input_path}")
    print("-" * 40)
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} samples")
    
    # Create new columns
    normalized_texts = []
    language_tags_list = []
    phonetic_texts = []
    
    # Process each row with progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        text = row['text']
        
        # Convert using our G2P system
        result = converter.convert(text)
        
        # Store results
        normalized_texts.append(result['normalized'])
        language_tags_list.append(str(result['language_tags']))
        phonetic_texts.append(result['phonetic_text'])
    
    # Add new columns to dataframe
    df['normalized'] = normalized_texts
    df['language_tags'] = language_tags_list
    df['phonetic'] = phonetic_texts
    
    # Save processed data
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved to: {output_path}")
    
    return df


def main():
    """Main function to process all datasets."""
    
    print("=" * 60)
    print("Processing Code-Mixed Dataset with G2P Converter")
    print("=" * 60)
    
    # Initialize converter
    converter = G2PConverter()
    
    # Define paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    
    # Files to process
    files = [
        ('train.csv', 'train_phonetic.csv'),
        ('val.csv', 'val_phonetic.csv'),
        ('test.csv', 'test_phonetic.csv'),
    ]
    
    # Process each file
    all_results = []
    
    for input_file, output_file in files:
        input_path = os.path.join(data_dir, input_file)
        output_path = os.path.join(data_dir, output_file)
        
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"WARNING: {input_path} not found, skipping...")
            continue
        
        # Process
        df = process_dataset(input_path, output_path, converter)
        all_results.append((input_file, df))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    
    total_samples = 0
    for filename, df in all_results:
        print(f"\n{filename}: {len(df)} samples processed")
        total_samples += len(df)
    
    print(f"\nTotal samples processed: {total_samples}")
    
    # Show sample results
    print("\n" + "=" * 60)
    print("Sample Results (First 3 from training set)")
    print("=" * 60)
    
    if all_results:
        sample_df = all_results[0][1].head(3)
        
        for idx, row in sample_df.iterrows():
            print(f"\n--- Sample {idx + 1} ---")
            print(f"Original:   {row['text']}")
            print(f"Normalized: {row['normalized']}")
            print(f"Languages:  {row['language_tags']}")
            print(f"Phonetic:   {row['phonetic']}")
            print(f"Label:      {row['label']}")
    
    # Print file locations
    print("\n" + "=" * 60)
    print("Output Files Created")
    print("=" * 60)
    
    for input_file, output_file in files:
        output_path = os.path.join(data_dir, output_file)
        if os.path.exists(output_path):
            print(f"✓ {output_path}")


if __name__ == "__main__":
    main()