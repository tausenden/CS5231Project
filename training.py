import random
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

CSV_PATH = "./cleaned_sample_parsed_cadets_tagged_chunked.csv"
MAX_LEN = 256
BATCH_SIZE = 256
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 4
FFN_DIM = 256
DROPOUT = 0.2

LR = 5e-4
EPOCHS = 15
SEED = 5231

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_df(csv_path):
    df = pd.read_csv(csv_path)
    df = df[["subject_uuid", "sequence", "label"]].dropna()

    label2id = {"normal": 0, "attack": 1}
    df = df[df["label"].isin(label2id.keys())].copy()
    df["y"] = df["label"].map(label2id)
    return df

def build_vocab(df):
    counter = Counter()
    for seq in df["sequence"]:
        counter.update(str(seq).split())

    token2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for i, tok in enumerate(counter.keys(), start=2):
        token2id[tok] = i

    id2token = {v: k for k, v in token2id.items()}
    pad_id = token2id[PAD_TOKEN]
    unk_id = token2id[UNK_TOKEN]
    vocab_size = len(token2id)
    return token2id, id2token, pad_id, unk_id, vocab_size

class SeqDataset(Dataset):
    def __init__(self, df, token2id, max_len):
        self.max_len = max_len
        self.token2id = token2id
        self.pad_id = token2id[PAD_TOKEN]

        self.seqs = []
        self.labels = []

        for _, row in df.iterrows():
            tokens = str(row["sequence"]).split()
            ids = [self.token2id.get(t, self.token2id[UNK_TOKEN]) for t in tokens]
            self.seqs.append(ids)
            self.labels.append(int(row["y"]))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [len(s) for s in sequences]
    max_len = min(max(lengths), MAX_LEN)

    batch_size = len(sequences)
    input_ids = torch.full((batch_size, max_len), 0, dtype=torch.long) # 0 is pad_id
    # pad_id variable is not global here if imported, but we know PAD_TOKEN=0 from build_vocab
    # To be safe, we can use token2id[PAD_TOKEN] if passed, but hardcoding 0 is safe with current build_vocab logic
    
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, seq in enumerate(sequences):
        seq = seq[:max_len]
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :len(seq)] = 1

    labels = torch.tensor(labels, dtype=torch.float32)
    return input_ids, attention_mask, labels

def evaluate(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            total_correct += (preds == labels.long()).sum().item()
            total_samples += labels.size(0)
    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc

def evaluate_metrics(model, data_loader, loss_fn, threshold=0.5):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_prob = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * labels.size(0)
            y_true.extend(labels.long().tolist())
            y_prob.extend(probs.detach().cpu().tolist())
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob > threshold).astype(int)

    avg_loss = total_loss / max(1, len(y_true))
    acc = (y_pred == y_true).mean() if len(y_true) > 0 else 0.0
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    
    # Calculate confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        # Handle edge cases where not all classes are present
        tn, fp, fn, tp = 0, 0, 0, 0

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "missed_attacks": int(fn),
        "false_alarms": int(fp)
    }

def collect_probs(model, data_loader):
    """Return y_true and y_prob for a given loader."""
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)

            y_true.extend(labels.long().tolist())
            y_prob.extend(probs.detach().cpu().tolist())

    return np.array(y_true), np.array(y_prob)

def find_best_threshold(model, valid_loader, target="recall"):
    y_true, y_prob = collect_probs(model, valid_loader)
    best_thr, best_score = 0.5, -1.0


    for thr in np.linspace(0.0, 1.0, 100):
        y_pred = (y_prob > thr).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        score = {"precision": precision, "recall": recall, "f1": f1}[target]
        if score > best_score:
            best_score, best_thr = score, thr

    print(f"Best {target}: {best_score:.4f} at threshold={best_thr:.3f}")
    return best_thr

def predict_sequence(model, seq_str, token2id, pad_id, max_len, device):
    tokens = str(seq_str).split()
    ids = [token2id.get(t, token2id[UNK_TOKEN]) for t in tokens]
    ids = ids[:max_len]

    input_ids = torch.full((1, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((1, max_len), dtype=torch.long)
    input_ids[0, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    attention_mask[0, :len(ids)] = 1

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        prob_attack = torch.sigmoid(logits)[0].item()
    return prob_attack

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers,
                 dim_feedforward, dropout, max_len, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len

        # Token + positional embeddings to encode discrete events with order
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Transformer encoder stack for contextual sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, input_ids, attention_mask):
        # input_ids: (B, L), attention_mask: (B, L)
        B, L = input_ids.size()
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)

        # Embed tokens + positions, switch to (L, B, D) for torch Transformer
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = x.transpose(0, 1)

        # True where token is padding so encoder can ignore it
        src_key_padding_mask = (attention_mask == 0)

        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        encoded = encoded.transpose(0, 1)  # back to (B, L, D)

        # Masked mean pooling keeps only real tokens when averaging
        mask = attention_mask.unsqueeze(-1)
        masked_encoded = encoded * mask
        summed = masked_encoded.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = summed / lengths

        pooled = self.dropout(pooled)
        logits = self.fc(pooled).squeeze(-1)
        return logits

class LSTMClassifier(nn.Module):
    """Embedding + LSTM encoder with masked mean pooling for classification."""

    def __init__(self, vocab_size, d_model, hidden_size, num_layers,
                 dropout, pad_id):
        super().__init__()
        self.pad_id = pad_id

        # Token embedding shared with Transformer for fair comparison
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # Unidirectional LSTM to capture sequential dependencies
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, input_ids, attention_mask):
        """Return logits of shape (B,) for the provided batch inputs."""
        x = self.token_emb(input_ids)  # (B, L, D)

        outputs, (h_n, c_n) = self.lstm(x)      # outputs: (B, L, H)

        # Masked mean pooling to ignore padded tokens
        mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
        masked_outputs = outputs * mask
        summed = masked_outputs.sum(dim=1)  # (B, H)
        lengths = mask.sum(dim=1).clamp(min=1)  # (B, 1)
        pooled = summed / lengths  # (B, H)

        pooled = self.dropout(pooled)
        logits = self.fc(pooled).squeeze(-1)    # (B,)
        # logits = self.fc(h_n[-1]).squeeze(-1)
        return logits

def main():
    print(device)

    # Data loading and split
    df = load_df(CSV_PATH)
    token2id, id2token, pad_id, unk_id, vocab_size = build_vocab(df)
    # Sequence length statistics

    seq_lengths = df["sequence"].astype(str).apply(lambda x: len(x.split()))
    print(f"\nSequence Length Statistics:")
    print(seq_lengths.describe())
    print(f"90th percentile: {seq_lengths.quantile(0.9)}")
    print(f"95th percentile: {seq_lengths.quantile(0.95)}")
    print(f"99th percentile: {seq_lengths.quantile(0.99)}")
    print(f"Max length setting: {MAX_LEN}")
    print(f"Sequences longer than MAX_LEN: {(seq_lengths > MAX_LEN).sum()} ({(seq_lengths > MAX_LEN).mean()*100:.2f}%)")

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["y"], random_state=SEED
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["y"], random_state=SEED
    )

    train_counts = train_df["y"].value_counts().to_dict()
    valid_counts = valid_df["y"].value_counts().to_dict()
    test_counts = test_df["y"].value_counts().to_dict()
    num_pos = float(train_counts.get(1, 0))
    num_neg = float(train_counts.get(0, 0))
    POS_WEIGHT = num_neg / max(1.0, num_pos)

    train_dataset = SeqDataset(train_df, token2id, MAX_LEN)
    valid_dataset = SeqDataset(valid_df, token2id, MAX_LEN)
    test_dataset = SeqDataset(test_df, token2id, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(len(df), df["y"].value_counts())
    print(f"Vocabulary size: {vocab_size}")

    # Training configuration display
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Sequence Length: {MAX_LEN}")
    print(f"Dataset Size (total): {len(df)}")
    print(f" - Train: {len(train_dataset)}  (batches: {len(train_loader)})")
    print(f" - Valid: {len(valid_dataset)}  (batches: {len(valid_loader)})")
    print(f" - Test : {len(test_dataset)}   (batches: {len(test_loader)})")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Class distribution (train): {train_counts}")
    print(f"Class distribution (valid): {valid_counts}")
    print(f"Class distribution (test) : {test_counts}")
    print(f"Loss pos_weight: {POS_WEIGHT:.2f}")
    print(f"{'='*60}\n")

    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=FFN_DIM,
        dropout=DROPOUT,
        max_len=MAX_LEN,
        pad_id=pad_id,
    ).to(device)

    pos_weight_tensor = torch.tensor([POS_WEIGHT]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_epochs = EPOCHS
    num_training_steps = len(train_loader) * total_epochs
    num_warmup_steps = int(0.05 * num_training_steps)

    print(f"num_warmup_steps: {num_warmup_steps}")
    print(f"num_training_steps: {num_training_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


    best_val_f1 = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*60}")
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        epoch_start = time.time()

        # Add progress bar for batches
        pbar = tqdm(train_loader, desc=f"Training", unit="batch")
        for input_ids, attention_mask, labels in pbar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).long()
                total_correct += (preds == labels.long()).sum().item()
                total_samples += labels.size(0)
            
            # Update progress bar with current metrics
            current_loss = total_loss / max(1, total_samples)
            current_acc = total_correct / max(1, total_samples)
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.4f}'
            })

        train_loss = total_loss / max(1, total_samples)
        train_acc = total_correct / max(1, total_samples)

        # Validation at end of epoch
        # Calculate metrics for validation set
        val_metrics = evaluate_metrics(model, valid_loader, loss_fn)
        val_loss = val_metrics["loss"]
        val_f1 = val_metrics["f1"]
        val_missed = val_metrics["missed_attacks"]
        
        epoch_time = time.time() - epoch_start
        print(f"\n Epoch {epoch} Summary: TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, ValF1={val_f1:.4f}")
        print(f" -> Missed Attacks: {val_missed}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), "best_transformer_model.pt")
            print(f" -> Saved best model (F1: {best_val_f1:.4f})")
        elif val_f1 < best_val_f1:
            patience_counter += 1
            print(f" -> Patience {patience_counter}/{patience} (Val F1 < Best Val F1)")
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        else:
            patience_counter = 0

    print(f"\n{'='*60}")
    print("Training LSTM classifier")
    print(f"{'='*60}")

    LSTM_NUM_LAYERS = 3
    LSTM_HIDDEN_SIZE = D_MODEL

    lstm_model = LSTMClassifier(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=DROPOUT,
        pad_id=pad_id,
    ).to(device)

    lstm_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LR)

    best_lstm_val_f1 = 0.0
    lstm_patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'-'*60}")
        print(f"LSTM Epoch {epoch}/{EPOCHS}")
        print(f"{'-'*60}")
        lstm_model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc="LSTM Training", unit="batch")
        for input_ids, attention_mask, labels in pbar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            lstm_optimizer.zero_grad()
            logits = lstm_model(input_ids, attention_mask)
            loss = lstm_loss_fn(logits, labels)
            loss.backward()
            lstm_optimizer.step()

            total_loss += loss.item() * labels.size(0)
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).long()
                total_correct += (preds == labels.long()).sum().item()
                total_samples += labels.size(0)

            current_loss = total_loss / max(1, total_samples)
            current_acc = total_correct / max(1, total_samples)
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.4f}'})

        train_loss = total_loss / max(1, total_samples)
        train_acc = total_correct / max(1, total_samples)
        val_metrics = evaluate_metrics(lstm_model, valid_loader, lstm_loss_fn)
        val_loss = val_metrics["loss"]
        val_f1 = val_metrics["f1"]
        val_missed = val_metrics["missed_attacks"]

        print(f" LSTM Epoch {epoch} Summary: TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, ValF1={val_f1:.4f}")
        print(f" -> Missed Attacks: {val_missed}")

        if val_f1 > best_lstm_val_f1:
            best_lstm_val_f1 = val_f1
            lstm_patience_counter = 0
            torch.save(lstm_model.state_dict(), "best_lstm_model.pt")
            print(f" -> Saved best LSTM model (F1: {best_lstm_val_f1:.4f})")
        elif val_f1 < best_lstm_val_f1:
            lstm_patience_counter += 1
            print(f" -> Patience {lstm_patience_counter}/{patience} (Val F1 < Best Val F1)")
            if lstm_patience_counter >= patience:
                print("LSTM Early stopping triggered")
                break
        else:
            lstm_patience_counter = 0

    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}")

    # Load best transformer model
    try:
        model.load_state_dict(torch.load("best_transformer_model.pt", weights_only=True))
        print("Loaded best transformer model from best_transformer_model.pt")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Could not load best model ({e}), using last checkpoint.")

    # Load best LSTM model
    try:
        lstm_model.load_state_dict(torch.load("best_lstm_model.pt", weights_only=True))
        print("Loaded best LSTM model from best_lstm_model.pt")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Could not load best LSTM model ({e}), using last checkpoint.")
    print("transformer threshold:")
    best_thr = find_best_threshold(model, valid_loader, target="f1")
    transformer_test_metrics = evaluate_metrics(model, test_loader, loss_fn, threshold=best_thr)

    print("lstm threshold:")
    best_lstm_thr = find_best_threshold(lstm_model, valid_loader, target="f1")
    lstm_test_metrics = evaluate_metrics(lstm_model, test_loader, lstm_loss_fn, threshold=best_lstm_thr)

    print("Transformer Test Metrics:")
    for k, v in transformer_test_metrics.items():
        print(f" - {k}: {v:.4f}")

    print("\nLSTM Test Metrics:")
    for k, v in lstm_test_metrics.items():
        print(f" - {k}: {v:.4f}")

    example_seq_normal = df[df["label"] == "normal"].iloc[0]["sequence"]
    example_seq_attack = df[df["label"] == "attack"].iloc[0]["sequence"]

    def get_label(prob, thr):
        return 'attack' if prob > thr else 'normal'

    print(f"\nTransformer inference (threshold={best_thr:.3f}):")
    p_normal = predict_sequence(model, example_seq_normal, token2id, pad_id, MAX_LEN, device)
    p_attack = predict_sequence(model, example_seq_attack, token2id, pad_id, MAX_LEN, device)
    print(f"Normal sample -> Prob: {p_normal:.4f}, Prediction: {get_label(p_normal, best_thr)}")
    print(f"Attack sample -> Prob: {p_attack:.4f}, Prediction: {get_label(p_attack, best_thr)}")

    print(f"\nLSTM inference (threshold={best_lstm_thr:.3f}):")
    p_normal_lstm = predict_sequence(lstm_model, example_seq_normal, token2id, pad_id, MAX_LEN, device)
    p_attack_lstm = predict_sequence(lstm_model, example_seq_attack, token2id, pad_id, MAX_LEN, device)
    print(f"Normal sample -> Prob: {p_normal_lstm:.4f}, Prediction: {get_label(p_normal_lstm, best_lstm_thr)}")
    print(f"Attack sample -> Prob: {p_attack_lstm:.4f}, Prediction: {get_label(p_attack_lstm, best_lstm_thr)}")

if __name__ == "__main__":
    main()
