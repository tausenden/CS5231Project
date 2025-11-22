import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from training import (
    load_df, build_vocab, SeqDataset, collate_fn,
    TransformerClassifier, LSTMClassifier, evaluate_metrics,
    CSV_PATH, MAX_LEN, BATCH_SIZE, SEED,
    D_MODEL, NHEAD, NUM_LAYERS, FFN_DIM, DROPOUT,
    device
)

def main():
    # Prepare Data
    print("Loading data...")
    df = load_df(CSV_PATH)
    token2id, id2token, pad_id, unk_id, vocab_size = build_vocab(df)
    
    # Recreate splits to get the exact same test set
    _, temp_df = train_test_split(df, test_size=0.2, stratify=df["y"], random_state=SEED)
    _, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["y"], random_state=SEED)
    
    test_dataset = SeqDataset(test_df, token2id, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    print(f"Test Set Size: {len(test_dataset)}")
    
    from torch import nn
    loss_fn = nn.BCEWithLogitsLoss() 
    output_dir = "output\\ver1\\"
    T_model_name = output_dir + "best_transformer_model.pt"
    L_model_name = output_dir + "best_lstm_model.pt"

    # Evaluate Transformer
    print(f"\n{'='*60}")
    print("Evaluating Transformer Model")
    print(f"{'='*60}")
    model = TransformerClassifier(vocab_size, D_MODEL, NHEAD, NUM_LAYERS, FFN_DIM, DROPOUT, MAX_LEN, pad_id).to(device)
    try:
        model.load_state_dict(torch.load(T_model_name, map_location=device, weights_only=True))
        metrics = evaluate_metrics(model, test_loader, loss_fn)
        for k, v in metrics.items():
            print(f" - {k}: {v}")
    except Exception as e:
        print(f"Error loading Transformer model from {T_model_name}: {e}")

    # Evaluate LSTM
    print(f"\n{'='*60}")
    print("Evaluating LSTM Model")
    print(f"{'='*60}")
    lstm_model = LSTMClassifier(vocab_size, D_MODEL, D_MODEL, 3, DROPOUT, pad_id).to(device)
    try:
        lstm_model.load_state_dict(torch.load(L_model_name, map_location=device, weights_only=True))
        metrics = evaluate_metrics(lstm_model, test_loader, loss_fn)
        for k, v in metrics.items():
            print(f" - {k}: {v}")
    except Exception as e:
        print(f"Error loading LSTM model from {L_model_name}: {e}")

if __name__ == "__main__":
    main()
