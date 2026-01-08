import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from auditor.auditor_model import AuditorMLP
from models.primary import PrimaryCNN
from training.data import apply_corruption, normalize_tensor


def load_models(device: torch.device) -> Tuple[PrimaryCNN, AuditorMLP]:
    primary = PrimaryCNN().to(device)
    auditor = AuditorMLP(input_dim=primary.latent_dim).to(device)
    primary.load_state_dict(torch.load("models/primary_cnn.pt", map_location=device))
    auditor.load_state_dict(torch.load("auditor/auditor_mlp.pt", map_location=device))
    primary.eval()
    auditor.eval()
    return primary, auditor


def load_cache():
    cache_path = Path("visualization/latent_cache.npz")
    if not cache_path.exists():
        return None
    data = np.load(cache_path, allow_pickle=True)
    return data


def fetch_dataset():
    return datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())


def run_inference(primary, auditor, image, corruption: str, severity: float, device: torch.device):
    corrupted = apply_corruption(image.clone(), corruption, severity)
    normalized = normalize_tensor(corrupted)
    with torch.no_grad():
        logits, latent = primary(normalized.unsqueeze(0).to(device), capture=True)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        score = torch.sigmoid(auditor(latent.to(device))).item()
    return corrupted, pred.item(), conf.item(), score


def find_confident_failure(primary, auditor, device, attempts: int = 300):
    data = fetch_dataset()
    rng = np.random.default_rng(7)
    for _ in range(attempts):
        idx = rng.integers(0, len(data))
        image, label = data[idx]
        corruption = rng.choice(["rotate", "noise", "occlude", "shift"])
        severity = rng.uniform(0.5, 1.0)
        corrupted, pred, conf, score = run_inference(
            primary, auditor, image, corruption=corruption, severity=severity, device=device
        )
        if pred != label and conf > 0.6 and score > 0.5:
            return {
                "image": corrupted,
                "pred": pred,
                "label": label,
                "confidence": conf,
                "auditor": score,
                "corruption": corruption,
                "severity": severity,
            }
    return None


def save_snapshot(device: torch.device):
    primary, auditor = load_models(device)
    sample = find_confident_failure(primary, auditor, device)
    if sample is None:
        print("no high-confidence failure found for snapshot")
        return
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(sample["image"].squeeze(0), cmap="gray")
    title = (
        f"pred {sample['pred']} ({sample['confidence']:.2f}) vs true {sample['label']}\n"
        f"auditor={sample['auditor']:.2f} corruption={sample['corruption']} s={sample['severity']:.2f}"
    )
    ax.set_title(title)
    ax.axis("off")
    out_path = Path("visualization/demo_snapshot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved snapshot to {out_path}")


def launch_streamlit():
    import streamlit as st
    from sklearn.decomposition import PCA

    device = torch.device("cpu")
    primary, auditor = load_models(device)
    cache = load_cache()
    dataset = fetch_dataset()

    st.title("Epistemic Failure Detection")
    st.write("Primary CNN + auditor that flags latent failures beyond softmax confidence.")

    if cache is not None and "pca_coords" not in st.session_state:
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(cache["latents"])
        st.session_state["pca_reducer"] = reducer
        st.session_state["pca_coords"] = coords

    col1, col2 = st.columns([2, 1])
    with col2:
        idx = st.number_input("sample index", min_value=0, max_value=len(dataset) - 1, value=0, step=1)
        corruption = st.selectbox(
            "corruption",
            ["clean", "rotate", "noise", "occlude", "shift"],
            index=1,
        )
        severity = st.slider("severity", 0.0, 1.0, 0.5, 0.05)
    image, label = dataset[int(idx)]
    corrupted, pred, conf, auditor_score = run_inference(
        primary, auditor, image, corruption=corruption, severity=severity, device=device
    )
    trust = auditor_score < 0.5

    with col1:
        st.image(corrupted.squeeze(0).numpy(), caption=f"corrupted ({corruption}, s={severity:.2f})", width=196)
    st.markdown(
        f"**prediction**: {pred} | **confidence**: {conf:.2f} | **label**: {label} | "
        f"**auditor**: {'TRUST' if trust else 'DO NOT TRUST'} ({auditor_score:.2f})"
    )

    if cache is not None and "pca_coords" in st.session_state:
        reducer = st.session_state["pca_reducer"]
        coords = st.session_state["pca_coords"]
        sample_latent = None
        with torch.no_grad():
            _, latent = primary(normalize_tensor(corrupted).unsqueeze(0).to(device), capture=True)
        sample_latent = reducer.transform(latent.numpy())
        st.subheader("latent view")
        fig, ax = plt.subplots(figsize=(4, 4))
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=cache["auditor_scores"], cmap="magma", alpha=0.8, s=12)
        ax.scatter(sample_latent[0, 0], sample_latent[0, 1], c="cyan", s=80, edgecolors="black")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, label="auditor failure prob")
        st.pyplot(fig)

    plots = ["visualization/confidence_vs_failure.png", "visualization/auditor_vs_failure.png"]
    for p in plots:
        if Path(p).exists():
            st.image(p, caption=Path(p).name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", action="store_true", help="generate a static snapshot of a flagged failure")
    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.snapshot:
        save_snapshot(device)
    else:
        launch_streamlit()


if __name__ == "__main__":
    main()
