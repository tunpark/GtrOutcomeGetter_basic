import json
from collections import Counter

def analyze_dataset_balance(input_file):
    """
    Analyzes the distribution of positive and negative samples in the dataset.
    Supports both JSON and JSONL formats.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data = json.load(f)
                file_format = 'JSON'
            else:
                data = [json.loads(line) for line in f if line.strip()]
                file_format = 'JSONL'
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise ValueError(f"Could not parse the file. Please check its format. Error: {e}")

    stats = {
        'total_samples': len(data),
        'positive_samples': 0,  # Samples with entities
        'negative_samples': 0,  # Samples without entities
        'entity_counts': Counter(),
        'file_format': file_format
    }
    
    for item in data:
        entity_count = len(item.get('entities', []))
        stats['entity_counts'][entity_count] += 1
        if entity_count == 0:
            stats['negative_samples'] += 1
        else:
            stats['positive_samples'] += 1
    
    total = stats['total_samples']
    if total > 0:
        stats['positive_ratio'] = stats['positive_samples'] / total
        stats['negative_ratio'] = stats['negative_samples'] / total
    
    return stats

def print_dataset_analysis(stats):
    """Prints the analysis results and provides a recommendation."""
    print("=" * 50)
    print("Dataset Distribution Analysis")
    print("=" * 50)
    
    print(f"File Format: {stats.get('file_format', 'Unknown')}")
    print(f"Total Samples: {stats['total_samples']}")
    print(f"Positive Samples (with entities): {stats['positive_samples']} ({stats.get('positive_ratio', 0):.1%})")
    print(f"Negative Samples (no entities): {stats['negative_samples']} ({stats.get('negative_ratio', 0):.1%})")
    
    print("\nRecommendation on Handling Empty Samples:")
    neg_ratio = stats.get('negative_ratio', 0)
    if neg_ratio > 0.8:
        print("  Advice: Negative samples are highly dominant (>80%). Consider removing them to balance the data.")
        return True  # Recommends removal
    elif neg_ratio > 0.6:
        print("  Advice: Negative samples are quite common (>60%). Optionally remove them.")
        return True  # Recommends optional removal
    else:
        print("  Advice: Data distribution is relatively balanced. It is recommended to keep empty samples as valid negative examples.")
        return False # Recommends keeping

def clean_single_item(item, remove_empty_entities=False):
    """Cleans a single data item by removing specified empty fields."""
    # Fields to remove if they are empty
    fields_to_remove = ['relations', 'comments']
    if remove_empty_entities and not item.get('entities'):
        # Only remove the 'entities' key if it's empty AND removal is requested
        fields_to_remove.append('entities')
    
    # Create a new dictionary without the empty fields
    return {k: v for k, v in item.items() if not (k in fields_to_remove and not v)}

def clean_doccano_data(input_file, output_file, remove_empty_entities=False):
    """Cleans a standard JSON file from doccano."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cleaned_data = [clean_single_item(item, remove_empty_entities) for item in data]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nJSON data cleaning complete! Saved to {output_file}")

def clean_jsonl_data(input_file, output_file, remove_empty_entities=False):
    """Cleans a JSONL file, processing one JSON object per line."""
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if line.strip():
                item = json.loads(line.strip())
                # If we need to remove empty entities, we check and potentially skip the whole line
                if remove_empty_entities and not item.get('entities'):
                    continue
                cleaned_item = clean_single_item(item, remove_empty_entities=False) # We already filtered, so don't re-check
                outfile.write(json.dumps(cleaned_item, ensure_ascii=False) + '\n')
    
    print(f"\nJSONL data cleaning complete! Saved to {output_file}")

def smart_clean_doccano_data(input_file, output_file):
    """
    Intelligently cleans doccano data by first analyzing its distribution
    and then deciding whether to remove samples with no entities.
    """
    print("Analyzing dataset...")
    stats = analyze_dataset_balance(input_file)
    should_remove = print_dataset_analysis(stats)
    
    print("\nStarting data cleaning process...")
    if stats['file_format'] == 'JSONL':
        clean_jsonl_data(input_file, output_file, remove_empty_entities=should_remove)
    else:
        clean_doccano_data(input_file, output_file, remove_empty_entities=should_remove)
    
    if should_remove:
        print("Action: Samples with empty 'entities' fields were removed.")
    else:
        print("Action: Samples with empty 'entities' fields were kept as negative examples.")

# --- Example Usage ---
if __name__ == "__main__":
    # Recommended usage: Smartly analyze and clean the data
    smart_clean_doccano_data('all.jsonl', 'cleaned_all.jsonl')