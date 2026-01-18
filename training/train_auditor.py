import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from auditor.auditor_model import AuditorMLP
from models.primary import PrimaryCNN
from training.data import get_corrupted_loader
from visualization.plotting import (
    plot_auditor_scores,
    plot_confidence_vs_failure,
    plot_latent_pca,
)


def collect_latents(model, loader, device, max_batches: int | None = None):
    model.eval()
    latents = []
    failures = []
    confidences = []
    preds = []
    truths = []
    corruptions = []
    with torch.no_grad():
        for i, (images, labels, corruption, severity) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            logits, latent = model(images, capture=True)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            failure = (pred != labels).float()
            latents.append(latent.cpu())
            failures.append(failure.cpu())
            confidences.append(conf.cpu())
            preds.append(pred.cpu())
            truths.append(labels.cpu())
            corruptions.extend(list(corruption))
            if max_batches is not None and i + 1 >= max_batches:
                break
    latents = torch.cat(latents, dim=0)
    failures = torch.cat(failures, dim=0)
    confidences = torch.cat(confidences, dim=0)
    preds = torch.cat(preds, dim=0)
    truths = torch.cat(truths, dim=0)
    return latents, failures, confidences, preds, truths, corruptions


def train_auditor_model(train_loader, val_loader, input_dim, device, epochs: int = 6):
    auditor = AuditorMLP(input_dim=input_dim).to(device)
    optimizer = optim.Adam(auditor.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        auditor.train()
        total_loss = 0.0
        total = 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = auditor(feats)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feats.size(0)
            total += feats.size(0)
        mean_loss = total_loss / total
        val_auc = evaluate_auditor(auditor, val_loader, device)
        print(f"epoch {epoch}: loss {mean_loss:.4f} | val AUC {val_auc:.3f}")
    return auditor


def evaluate_auditor(auditor, loader, device):
    auditor.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for feats, l in loader:
            feats = feats.to(device)
            logits = auditor(feats)
            probs = torch.sigmoid(logits).cpu()
            scores.append(probs)
            labels.append(l)
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    return roc_auc_score(labels, scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    primary = PrimaryCNN().to(device)
    primary_path = Path("models/primary_cnn.pt")
    if not primary_path.exists():
        raise FileNotFoundError("train the primary model first (models/primary_cnn.pt missing)")
    primary.load_state_dict(torch.load(primary_path, map_location=device))

    loader = get_corrupted_loader(
        split="train", batch_size=args.batch_size, corruption_chance=0.8
    )
    latents, failures, confidences, preds, truths, corruptions = collect_latents(
        primary, loader, device, max_batches=args.batches
    )

    permutation = torch.randperm(latents.size(0))
    latents = latents[permutation]
    failures = failures[permutation]
    confidences = confidences[permutation]

    split = int(0.8 * latents.size(0))
    train_feats, val_feats = latents[:split], latents[split:]
    train_labels, val_labels = failures[:split], failures[split:]

    train_data = TensorDataset(train_feats, train_labels)
    val_data = TensorDataset(val_feats, val_labels)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

    #train auditor
    auditor = train_auditor_model(
        train_loader, val_loader, input_dim=primary.latent_dim, device=device, epochs=args.epochs
    )

    auditor.eval()
    with torch.no_grad():
        auditor_scores = torch.sigmoid(auditor(latents.to(device))).cpu().numpy()
    softmax_failure_score = (1.0 - confidences.numpy())
    failure_labels = failures.numpy()

    auditor_auc = roc_auc_score(failure_labels, auditor_scores)
    softmax_auc = roc_auc_score(failure_labels, softmax_failure_score)

    results = {
        "auditor_roc_auc": float(auditor_auc),
        "softmax_roc_auc": float(softmax_auc),
        "samples": int(latents.size(0)),
    }
    print(json.dumps(results, indent=2))

    Path("auditor").mkdir(exist_ok=True, parents=True)
    torch.save(auditor.state_dict(), Path("auditor/auditor_mlp.pt"))
    with open(Path("auditor/metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    cache_path = Path("visualization/latent_cache.npz")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        latents=latents.numpy(),
        failures=failure_labels,
        confidences=confidences.numpy(),
        auditor_scores=auditor_scores,
        preds=preds.numpy(),
        truths=truths.numpy(),
        corruptions=np.array(corruptions),
    )

    plot_confidence_vs_failure(1.0 - softmax_failure_score, failure_labels, Path("visualization/confidence_vs_failure.png"))
    plot_auditor_scores(auditor_scores, failure_labels, Path("visualization/auditor_vs_failure.png"))
    plot_latent_pca(latents.numpy(), failure_labels, auditor_scores, Path("visualization/latent_pca.png"))

    print(f"auditor ROC-AUC {auditor_auc:.3f} vs softmax {softmax_auc:.3f}")

if __name__ == "__main__":
    main()
