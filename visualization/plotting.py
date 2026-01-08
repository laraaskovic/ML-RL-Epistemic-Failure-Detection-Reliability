from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_confidence_vs_failure(confidences, failures, out_path: Path):
    plt.figure(figsize=(6, 4))
    jitter = (np.random.rand(len(failures)) - 0.5) * 0.06
    plt.scatter(confidences, failures + jitter, s=12, alpha=0.4, c=failures, cmap="coolwarm")
    plt.xlabel("Softmax confidence")
    plt.yticks([0, 1], ["correct", "failure"])
    plt.title("Confidence vs. outcome")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_auditor_scores(scores, failures, out_path: Path):
    plt.figure(figsize=(6, 4))
    jitter = (np.random.rand(len(failures)) - 0.5) * 0.06
    plt.scatter(scores, failures + jitter, s=12, alpha=0.4, c=failures, cmap="coolwarm")
    plt.xlabel("Auditor score (prob failure)")
    plt.yticks([0, 1], ["correct", "failure"])
    plt.title("Auditor vs. outcome")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_latent_pca(latents, failures, scores, out_path: Path):
    reducer = PCA(n_components=2)
    coords = reducer.fit_transform(latents)
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=scores,
        cmap="magma",
        alpha=0.8,
        s=18,
        edgecolors="none",
    )
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Latent trust landscape")
    cb = plt.colorbar(sc)
    cb.set_label("Auditor failure prob")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
