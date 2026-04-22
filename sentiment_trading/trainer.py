"""
Fine-tune a TransformerSentimentModel on FinancialPhraseBank.
AdamW + linear warmup, early stopping on val F1 (patience 1). (paper III-C)
"""
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from .config import (
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, WARMUP_RATIO,
    MAX_EPOCHS, EARLY_STOP_PATIENCE, CHECKPOINTS_DIR,
)
from .data import SentimentDataset, get_splits
from .evaluator import compute_metrics
from .models import TransformerSentimentModel


def train_model(model_key: str, hf_name: str, force: bool = False) -> dict:
    """Fine-tune and evaluate a single model. Returns best val metrics."""
    ckpt_dir = CHECKPOINTS_DIR / model_key
    metrics_file = ckpt_dir / "metrics.json"
    if metrics_file.exists() and not force:
        return json.loads(metrics_file.read_text())

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Training {model_key} ({hf_name}) ===")
    wrapper = TransformerSentimentModel(model_key, hf_name)

    train_rows, val_rows, _ = get_splits()
    train_ds = SentimentDataset(train_rows, hf_name)
    val_ds = SentimentDataset(val_rows, hf_name)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = AdamW(
        wrapper.model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * MAX_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1 = -1.0
    patience_counter = 0
    best_metrics: dict = {}

    for epoch in range(1, MAX_EPOCHS + 1):
        wrapper.model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            # Remap canonical labels to native head indices
            raw_labels = batch["labels"].tolist()
            native_labels = torch.tensor(
                [wrapper.native_label_for(lbl) for lbl in raw_labels],
                dtype=torch.long, device=wrapper.device
            )
            input_ids = batch["input_ids"].to(wrapper.device)
            attention_mask = batch["attention_mask"].to(wrapper.device)
            kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, labels=native_labels)
            if batch.get("token_type_ids") is not None:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(wrapper.device)

            optimizer.zero_grad()
            out = wrapper.model(**kwargs)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        wrapper.model.eval()
        y_true, y_pred = [], []
        for batch in val_loader:
            sentences = [val_rows[i]["sentence"] for i in range(len(batch["labels"]))]
            preds = wrapper.predict_class(sentences)
            y_true.extend(batch["labels"].tolist())
            y_pred.extend(preds)

        metrics = compute_metrics(y_true, y_pred)
        print(
            f"  Epoch {epoch} | loss={avg_loss:.4f} | "
            f"acc={metrics['accuracy']:.4f} | f1={metrics['f1']:.4f} | mae={metrics['mae']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = metrics
            patience_counter = 0
            wrapper.model.save_pretrained(ckpt_dir)
            wrapper.tokenizer.save_pretrained(ckpt_dir)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    metrics_file.write_text(json.dumps(best_metrics))
    return best_metrics
