import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import json
import random
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings("ignore")

class SpacyNERTrainer:
    """A trainer for spaCy Named Entity Recognition (NER) models."""
    
    def __init__(self, model_name="en_core_web_sm", output_dir="spacy_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.nlp = None
        self.entity_labels = set()
        self.best_model_path = None

    def load_data(self, file_path):
        """Loads NER data from a JSONL file into spaCy's format."""
        print(f"Loading data from: {file_path}")
        training_data = []
        skipped_samples = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    text = data['text']
                    entities = data.get('entities', [])
                    
                    if not text.strip():
                        skipped_samples += 1
                        continue
                        
                    spacy_entities = []
                    for start, end, label in entities:
                        if start < end and end <= len(text):
                            spacy_entities.append((start, end, label))
                            self.entity_labels.add(label)
                        else:
                            print(f"Warning: Invalid entity boundary in line {line_num}. Skipping entity.")
                    
                    training_data.append((text, {"entities": spacy_entities}))
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping malformed line {line_num}: {e}")
                    skipped_samples += 1
        
        print(f"Successfully loaded {len(training_data)} samples.")
        if skipped_samples > 0:
            print(f"Skipped {skipped_samples} problematic samples.")
        
        self.print_data_stats(training_data)
        return training_data

    def print_data_stats(self, data):
        """Prints statistics about the loaded dataset."""
        entity_counter = Counter(label for _, ann in data for _, _, label in ann.get('entities', []))
        print("\nDataset Statistics:")
        print(f"  Total samples: {len(data)}")
        print(f"  Total entities: {sum(entity_counter.values())}")
        print(f"  Entity types: {len(self.entity_labels)}")
        print("  Entity distribution:")
        for label, count in entity_counter.most_common():
            print(f"    {label}: {count}")
        print()

    def balance_data(self, training_data, method="oversample", target_ratio=0.7):
        """Balances the entity distribution in the training data."""
        print(f"Balancing data using '{method}' method with target ratio {target_ratio}...")
        entity_samples = defaultdict(list)
        for text, annotations in training_data:
            for _, _, label in annotations.get('entities', []):
                entity_samples[label].append((text, annotations))

        if not entity_samples:
            print("No entities found to balance. Returning original data.")
            return training_data
        
        if method == "oversample":
            max_samples = max(len(s) for s in entity_samples.values())
            target_count = int(max_samples * target_ratio)
            balanced_data = list(training_data) # Start with all original data
            
            for label, samples in entity_samples.items():
                if len(samples) < target_count:
                    needed = target_count - len(samples)
                    balanced_data.extend(random.choices(samples, k=needed))
            
            print(f"Data balanced via oversampling. New total samples: {len(balanced_data)}")
            return balanced_data
        else:
            print(f"Warning: Unknown balancing method '{method}'. Returning original data.")
            return training_data

    def setup_model(self, use_pretrained=True):
        """Sets up the spaCy model pipeline."""
        print("Setting up spaCy model...")
        if use_pretrained:
            try:
                self.nlp = spacy.load(self.model_name)
                print(f"Loaded pre-trained model: {self.model_name}")
            except OSError:
                print(f"Warning: Pre-trained model '{self.model_name}' not found. Creating a blank model.")
                self.nlp = spacy.blank("en")
        else:
            self.nlp = spacy.blank("en")
            print("Created a blank English model.")
            
        if "ner" not in self.nlp.pipe_names:
            self.nlp.add_pipe("ner", last=True)
        ner = self.nlp.get_pipe("ner")
        
        for label in self.entity_labels:
            ner.add_label(label)
        print("NER labels added to the pipeline.")

    def train_model(self, train_data, n_iter=20, dropout=0.2):
        """Trains the spaCy NER model."""
        print("\nStarting model training...")
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            for itn in range(n_iter):
                random.shuffle(train_data)
                losses = {}
                batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    examples = [Example.from_dict(self.nlp.make_doc(text), anns) for text, anns in batch]
                    self.nlp.update(examples, drop=dropout, sgd=optimizer, losses=losses)
                print(f"Iteration {itn + 1}/{n_iter}, Loss: {losses.get('ner', 0.0):.4f}")
        
        print("Training complete.")

    def evaluate_model(self, test_data):
        """Evaluates the model's performance on test data."""
        print("\nEvaluating model performance...")
        examples = [Example.from_dict(self.nlp.make_doc(text), anns) for text, anns in test_data]
        scores = self.nlp.evaluate(examples)
        
        print("Evaluation Results:")
        print(f"  Precision: {scores.get('ents_p', 0.0):.4f}")
        print(f"  Recall: {scores.get('ents_r', 0.0):.4f}")
        print(f"  F1-Score: {scores.get('ents_f', 0.0):.4f}")
        return scores

    def save_model(self):
        """Saves the trained model to the output directory."""
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(output_path)
        print(f"Model saved to: {output_path}")

    def predict_examples(self, test_data, num_examples=3):
        """Shows prediction examples on test data."""
        print("\nPrediction Examples:")
        for text, _ in random.sample(test_data, min(num_examples, len(test_data))):
            doc = self.nlp(text)
            print(f"\nText: {text}")
            print("Entities:")
            if doc.ents:
                for ent in doc.ents:
                    print(f"  -> '{ent.text}' ({ent.label_})")
            else:
                print("  No entities found.")

def main():
    """Main training pipeline."""
    DATA_FILE = 'cleaned_all.jsonl'
    OUTPUT_DIR = 'spacy_ner_model'
    MODEL_NAME = "en_core_web_lg"
    N_ITER = 30

    print("--- spaCy NER Training Pipeline ---")
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at '{DATA_FILE}'")
        return
        
    trainer = SpacyNERTrainer(model_name=MODEL_NAME, output_dir=OUTPUT_DIR)
    
    # 1. Load and prepare data
    all_data = trainer.load_data(DATA_FILE)
    if not all_data:
        print("Error: No valid data loaded. Exiting.")
        return
        
    # 2. Balance data
    balanced_data = trainer.balance_data(all_data, method="oversample", target_ratio=0.7)
    
    # 3. Split data
    train_data, test_data = train_test_split(balanced_data, test_size=0.2, random_state=42)
    print(f"Data split: {len(train_data)} training samples, {len(test_data)} test samples.")
    
    # 4. Setup and train model
    trainer.setup_model(use_pretrained=True)
    trainer.train_model(train_data, n_iter=N_ITER)
    
    # 5. Evaluate, predict, and save
    trainer.evaluate_model(test_data)
    trainer.predict_examples(test_data)
    trainer.save_model()
    
    print("\nPipeline finished successfully!")

if __name__ == "__main__":
    main()