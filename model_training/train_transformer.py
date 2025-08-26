import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, AutoConfig,
    get_linear_schedule_with_warmup, BertTokenizerFast, RobertaTokenizerFast
)
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import os
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NERDataset(Dataset):
    """A PyTorch Dataset for NER data in JSONL format."""
    
    def __init__(self, texts, tags, tokenizer, max_length=128, label2id=None):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id if label2id else {}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        entities = self.tags[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        labels = self._align_labels(text, entities, encoding['offset_mapping'].squeeze().tolist())
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

    def _align_labels(self, text, entities, offset_mapping):
        """Aligns entity labels with tokenizer tokens using BIO scheme."""
        char_labels = ['O'] * len(text)
        for start, end, label in entities:
            if start < len(text) and end <= len(text):
                char_labels[start] = f'B-{label}'
                for i in range(start + 1, end):
                    char_labels[i] = f'I-{label}'
        
        token_labels = []
        for start, end in offset_mapping:
            if start == end: # Special tokens
                token_labels.append('O')
                continue
            
            # Use the label of the first character of the token
            relevant_char_label = char_labels[start]
            token_labels.append(relevant_char_label)
        
        return [self.label2id.get(lbl, self.label2id['O']) for lbl in token_labels]

class TransformerNERTrainer:
    """A trainer for Transformer-based NER models."""
    
    def __init__(self, model_name="bert-base-uncased", output_dir="transformer_ner_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        Tokenizer = RobertaTokenizerFast if 'roberta' in model_name.lower() else BertTokenizerFast
        self.tokenizer = Tokenizer.from_pretrained(model_name)
        
        self.label2id = {}
        self.id2label = {}
        self.model = None
        
        logger.info(f"Using device: {self.device}")

    def load_data(self, file_path):
        """Loads NER data from a JSONL file."""
        logger.info(f"Loading data from: {file_path}")
        texts, all_entities, entity_labels = [], [], set()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                texts.append(data['text'])
                ents = [(s, e, lbl) for s, e, lbl in data.get('entities', [])]
                all_entities.append(ents)
                for _, _, label in ents:
                    entity_labels.add(label)
        
        self._build_label_vocab(entity_labels)
        return texts, all_entities

    def _build_label_vocab(self, entity_labels):
        """Builds BIO label vocabulary."""
        labels = ['O']
        for label in sorted(entity_labels):
            labels.extend([f'B-{label}', f'I-{label}'])
        
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        logger.info(f"Built label vocabulary with {len(labels)} labels.")

    def train(self, train_texts, train_tags, val_texts, val_tags, epochs=5, batch_size=16, lr=2e-5):
        """Trains the Transformer NER model."""
        train_dataset = NERDataset(train_texts, train_tags, self.tokenizer, label2id=self.label2id)
        val_dataset = NERDataset(val_texts, val_tags, self.tokenizer, label2id=self.label2id)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        config = AutoConfig.from_pretrained(
            self.model_name, num_labels=len(self.label2id), 
            id2label=self.id2label, label2id=self.label2id
        )
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, config=config).to(self.device)
        
        optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        
        logger.info("Starting training...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            val_metrics = self.evaluate(val_loader)
            logger.info(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val F1={val_metrics['weighted avg']['f1-score']:.4f}")

    def evaluate(self, data_loader):
        """Evaluates the model on a given dataset."""
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=2)
                
                for i in range(labels.shape[0]):
                    active_indices = attention_mask[i] == 1
                    all_labels.extend([self.id2label[l.item()] for l in labels[i][active_indices]])
                    all_preds.extend([self.id2label[p.item()] for p in preds[i][active_indices]])
        
        return classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    def save_model(self):
        """Saves the trained model and tokenizer."""
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        logger.info(f"Model saved to {output_path}")

def main():
    DATA_FILE = 'cleaned_all.jsonl'
    OUTPUT_DIR = 'transformer_ner_model'
    MODEL_NAME = "distilbert-base-uncased"
    EPOCHS = 10
    
    print("--- Transformer NER Training Pipeline ---")
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at '{DATA_FILE}'")
        return

    trainer = TransformerNERTrainer(model_name=MODEL_NAME, output_dir=OUTPUT_DIR)
    texts, entities = trainer.load_data(DATA_FILE)
    
    train_texts, test_texts, train_tags, test_tags = train_test_split(
        texts, entities, test_size=0.2, random_state=42
    )
    
    trainer.train(train_texts, train_tags, test_texts, test_tags, epochs=EPOCHS)
    
    # Final evaluation on test set
    test_dataset = NERDataset(test_texts, test_tags, trainer.tokenizer, label2id=trainer.label2id)
    test_loader = DataLoader(test_dataset, batch_size=16)
    final_metrics = trainer.evaluate(test_loader)
    print("\nFinal Test Set Evaluation:")
    print(f"  F1 Score (weighted): {final_metrics['weighted avg']['f1-score']:.4f}")
    
    trainer.save_model()
    print("Pipeline finished successfully!")

if __name__ == "__main__":
    main()