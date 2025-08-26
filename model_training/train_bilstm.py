import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report as seq_report
from seqeval.metrics import f1_score as seq_f1
import pickle
import os
import re

class NERDataset(Dataset):
    """PyTorch Dataset for NER."""
    def __init__(self, sentences, tags, word2idx, tag2idx):
        self.sentences = sentences
        self.tags = tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.unk_idx = word2idx.get('<UNK>', 1)
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sent_indices = [self.word2idx.get(w, self.unk_idx) for w in self.sentences[idx]]
        tag_indices = [self.tag2idx.get(t, 0) for t in self.tags[idx]]
        return torch.tensor(sent_indices, dtype=torch.long), torch.tensor(tag_indices, dtype=torch.long)

def collate_fn(batch):
    """Pads sequences to the max length in a batch."""
    sentences, tags = zip(*batch)
    lengths = [len(s) for s in sentences]
    max_len = max(lengths)
    
    padded_sents = torch.zeros(len(sentences), max_len, dtype=torch.long)
    padded_tags = torch.zeros(len(sentences), max_len, dtype=torch.long)
    masks = torch.zeros(len(sentences), max_len, dtype=torch.bool)
    
    for i, s in enumerate(sentences):
        end = lengths[i]
        padded_sents[i, :end] = s[:end]
        padded_tags[i, :end] = tags[i][:end]
        masks[i, :end] = 1
        
    return padded_sents, padded_tags, masks

class BiLSTMCRF(nn.Module):
    # This is a simplified BiLSTM-CRF implementation.
    # For brevity, this example omits the CRF layer logic.
    # A full implementation would include transition scores and Viterbi decoding.
    def __init__(self, vocab_size, tag_size, embedding_dim=100, hidden_dim=128):
        super(BiLSTMCRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0) # Ignore PAD token

    def forward(self, sentences, tags=None, mask=None):
        embeds = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeds)
        logits = self.hidden2tag(lstm_out)
        
        if tags is not None:
            # Calculate loss
            active_loss = mask.view(-1) == 1
            active_logits = logits.view(-1, logits.shape[-1])[active_loss]
            active_labels = tags.view(-1)[active_loss]
            loss = self.loss_fn(active_logits, active_labels)
            return loss
        else:
            # Return predictions
            return torch.argmax(logits, dim=2)

class NERTrainer:
    """A trainer for the BiLSTM-CRF NER model."""
    
    def __init__(self, output_dir="bilstm_crf_model"):
        self.output_dir = output_dir
        self.word2idx, self.tag2idx = {}, {}
        self.idx2word, self.idx2tag = {}, {}
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(output_dir, exist_ok=True)

    def load_bio_data(self, file_path):
        """Loads data from a file in BIO format."""
        print(f"Loading BIO format data from: {file_path}")
        sentences, tags = [], []
        words, current_tags = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        words.append(parts[0])
                        current_tags.append(parts[-1])
                elif words:
                    sentences.append(words)
                    tags.append(current_tags)
                    words, current_tags = [], []
        if words:
            sentences.append(words)
            tags.append(current_tags)
        print(f"Loaded {len(sentences)} samples.")
        return sentences, tags

    def build_vocab(self, sentences, tags):
        """Builds word and tag vocabularies."""
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        for sent in sentences:
            for word in sent:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    
        self.tag2idx = {'<PAD>': 0}
        for tag_seq in tags:
            for tag in tag_seq:
                if tag not in self.tag2idx:
                    self.tag2idx[tag] = len(self.tag2idx)
                    
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.idx2tag = {i: t for t, i in self.tag2idx.items()}
        print(f"Vocabulary built: {len(self.word2idx)} words, {len(self.tag2idx)} tags.")

    def train(self, train_data, val_data, epochs=10, lr=0.001):
        """Trains the BiLSTM-CRF model."""
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_data, batch_size=32, collate_fn=collate_fn)
        
        self.model = BiLSTMCRF(
            len(self.word2idx), len(self.tag2idx)
        ).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        print("\nStarting model training...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for sents, tags, mask in train_loader:
                sents, tags, mask = sents.to(self.device), tags.to(self.device), mask.to(self.device)
                optimizer.zero_grad()
                loss = self.model(sents, tags, mask)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            val_f1 = self.evaluate(val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val F1: {val_f1:.4f}")

    def evaluate(self, data_loader):
        """Evaluates the model on a dataset."""
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for sents, tags, mask in data_loader:
                sents, tags, mask = sents.to(self.device), tags.to(self.device), mask.to(self.device)
                preds = self.model(sents, mask=mask)
                
                for i in range(len(tags)):
                    active_indices = mask[i] == 1
                    all_labels.append([self.idx2tag[t.item()] for t in tags[i][active_indices]])
                    all_preds.append([self.idx2tag[p.item()] for p in preds[i][active_indices]])
        
        return seq_f1(all_labels, all_preds, average="weighted", zero_division=0)

    def save(self):
        """Saves the model and vocabularies."""
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "model.pth"))
        vocabs = {'word2idx': self.word2idx, 'tag2idx': self.tag2idx}
        with open(os.path.join(self.output_dir, "vocabs.pkl"), "wb") as f:
            pickle.dump(vocabs, f)
        print(f"Model and vocabs saved to {self.output_dir}")

def main():
    DATA_FILE = 'bio_format.txt'
    OUTPUT_DIR = 'bilstm_crf_model'

    print("--- BiLSTM-CRF NER Training Pipeline ---")
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at '{DATA_FILE}'")
        return

    trainer = NERTrainer(output_dir=OUTPUT_DIR)
    sentences, tags = trainer.load_bio_data(DATA_FILE)
    trainer.build_vocab(sentences, tags)
    
    train_sents, test_sents, train_tags, test_tags = train_test_split(
        sentences, tags, test_size=0.2, random_state=42
    )
    
    train_dataset = NERDataset(train_sents, train_tags, trainer.word2idx, trainer.tag2idx)
    test_dataset = NERDataset(test_sents, test_tags, trainer.word2idx, trainer.tag2idx)
    
    trainer.train(train_dataset, test_dataset, epochs=20)
    
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
    final_f1 = trainer.evaluate(test_loader)
    print(f"\nFinal Test F1-Score: {final_f1:.4f}")
    
    trainer.save()
    print("Pipeline finished successfully!")

if __name__ == "__main__":
    main()