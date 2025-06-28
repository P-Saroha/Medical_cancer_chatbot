import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import numpy as np
import json
import pickle
import random
from collections import Counter
from tqdm import tqdm

# ------------------- Config -------------------
class Config:
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    kb_path = "cancer_knowledge_base.json"
    model_save_path = "fold3_best_model.pth"
    max_len = 128
    batch_size = 8
    learning_rate = 2e-5
    epochs = 10
    dropout_rate = 0.3
    patience = 3
    k_folds = 3
    seed = 42
    use_focal_loss = True
    alpha = 0.25
    gamma = 2.0
    warmup_ratio = 0.1
    accumulation_steps = 2
    weight_decay = 0.01

config = Config()

# ------------------- Dataset -------------------
class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=config.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx])
        }

# ------------------- Focal Loss -------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()

# ------------------- Model -------------------
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.model_name)
        self.drop = nn.Dropout(config.dropout_rate)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask)
        return self.fc(self.drop(out.pooler_output))

# ------------------- Data Load -------------------
def load_data():
    with open(config.kb_path, 'r') as f:
        data = json.load(f)

    texts, labels = [], []
    tag2idx, idx2tag = {}, {}

    for i, intent in enumerate(data['intents']):
        tag2idx[intent['tag']] = i
        idx2tag[i] = intent['tag']
        for pattern in intent['patterns']:
            texts.append(pattern)
            labels.append(i)

    with open("tag2idx.pickle", 'wb') as f:
        pickle.dump(tag2idx, f)
    with open("idx2tag.pickle", 'wb') as f:
        pickle.dump(idx2tag, f)

    return texts, labels, idx2tag

# ------------------- Training -------------------
def train():
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    texts, labels, idx2tag = load_data()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(idx2tag)

    skf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\nFold {fold+1}/{config.k_folds}")

        train_ds = MedicalDataset([texts[i] for i in train_idx], [labels[i] for i in train_idx], tokenizer)
        val_ds = MedicalDataset([texts[i] for i in val_idx], [labels[i] for i in val_idx], tokenizer)

        label_counts = Counter(train_ds.labels)
        weights = [1.0 / label_counts[label] for label in train_ds.labels]
        sampler = WeightedRandomSampler(weights, len(weights))

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size)

        model = Classifier(num_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        total_steps = len(train_loader) * config.epochs // config.accumulation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config.warmup_ratio), total_steps)
        criterion = FocalLoss() if config.use_focal_loss else nn.CrossEntropyLoss()

        best_f1 = 0
        patience = 0

        for epoch in range(config.epochs):
            model.train()
            total_loss = 0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                targets = batch['label'].to(device)

                outputs = model(ids, mask)
                loss = criterion(outputs, targets)
                loss = loss / config.accumulation_steps
                loss.backward()

                if (step + 1) % config.accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * config.accumulation_steps

            # Validation
            model.eval()
            preds, truths = [], []
            with torch.no_grad():
                for batch in val_loader:
                    ids = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    targets = batch['label'].to(device)

                    outputs = model(ids, mask)
                    preds.extend(outputs.argmax(dim=1).cpu().numpy())
                    truths.extend(targets.cpu().numpy())

            f1 = f1_score(truths, preds, average='weighted')
            print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | F1 Score: {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                patience = 0
                if fold == 2:  # Save only fold 3 for compatibility with your app
                    torch.save(model.state_dict(), config.model_save_path)
                    print("Saved best fold3 model")
            else:
                patience += 1
                if patience >= config.patience:
                    print("Early stopping")
                    break

if __name__ == '__main__':
    train()
