Epistemic Failure Detection: Models That Know When They Are Wrong
=================================================================

Problem
-------
Image classifiers on real deployments fail silently when inputs drift or are corrupted. Softmax confidence stays high even when the model is wrong because it reflects relative logit scale, not epistemic coverage. This repository pairs a standard MNIST classifier with a secondary auditor that reasons over internal activations to decide when the classifier should not be trusted.

Approach
--------
- Primary model: shallow CNN trained on MNIST.
- Failure injection: rotations, translation jitter, occlusion, and additive noise applied at inference to induce distribution shift.
- Auditor: a small MLP that ingests concatenated latent activations from early and late layers. It is supervised directly by correctness labels (failure = 1), never by confidence.
- Objective: auditor ROC-AUC must exceed the softmax-confidence baseline as a failure detector.

Data and Corruptions
--------------------
- Dataset: torchvision MNIST (60k train / 10k test) with per-channel standardization (mean 0.1307, std 0.3081).
- Corruptions sampled at load time with 0.7–0.8 probability: `rotate` (20–80 deg bilinear), `noise` (Gaussian, sigma 0.3–0.8), `occlude` (zero mask 6–18 px radius), `shift` (integer grid shift up to 3 px). Remaining samples are clean.
- Corruptions are applied on the fly; severity is sampled uniformly in [0,1] and logged alongside each example for later analysis.
- Latents use normalized tensors; demo corruption controls reuse the same primitives for exact parity with the training pipeline.

Training Pipeline
-----------------
- Primary classifier (`models/primary.py`): 2 conv + BN + ReLU blocks with max-pooling, 128-dense hidden, and 10-way logits. Trained for 3 epochs with Adam (lr=1e-3, batch=128) on normalized MNIST; seeds set for torch/cuda.
- Latent capture: pooled conv1 (32) + pooled conv2 (64) + hidden (128) concatenated into a 224-D representation for every input when `capture=True`.
- Auditor dataset (`training/train_auditor.py`): run the frozen primary over ~38k corrupted digits (150 batches x 256) with corruption chance 0.8. Each sample carries `(latent, failure_label)` where `failure_label = 1` iff predicted class != ground truth.
- Auditor model (`auditor/auditor_model.py`): 224 -> 128 -> 64 -> 1 MLP with dropout 0.1, trained 6 epochs with Adam (lr=1e-3) and `BCEWithLogitsLoss`. Data split 80/20 for validation AUC monitoring.
- Metrics: primary accuracy logged per epoch to `models/primary_metrics.json`. Auditor vs. softmax ROC-AUC computed on the full latent set and saved to `auditor/metrics.json`.

Architecture
------------
```
          input image
               |
         [ Primary CNN ]
           conv -> conv -> fc -> logits
             |         |
             +---- latent stack ---------+
                                         v
                                [ Auditor MLP ]
                                 failure score
```

Results
-------
- Primary accuracy (clean MNIST): 0.987.
- Failure detection (corrupted MNIST mix, ~38k samples, corruption p=0.8):
  - Auditor ROC-AUC: 0.958
  - Softmax (1-confidence) ROC-AUC: 0.835
- Visuals in `visualization/`: confidence vs. outcome, auditor score vs. outcome, latent PCA map with failure probabilities.
- `visualization/demo_snapshot.png` shows a high-confidence wrong prediction flagged by the auditor.

Limitations
-----------
- Auditor is trained on MNIST-specific corruptions; generalization to other datasets or shifts is untested.
- Only a lightweight CNN and MLP are used; richer feature hierarchies may expose stronger epistemic signals.
- No explicit adversarial training; adversarial robustness remains open.

Future Work
-----------
- Expand corruption space (blur, lighting, spatial frequency shifts) and evaluate transfer to unseen perturbations.
- Calibrate auditor thresholds for operational risk budgets and integrate with abstention policies.
- Replace the MLP with contrastive or energy-based detectors over latents.
- Study causal attribution: which layers and channels carry the strongest failure evidence.

Live Demo
---------
- Hosted Streamlit endpoint (replace with your URL once deployed): `<your-streamlit-link-here>`.
- To deploy on Streamlit Community Cloud: push this repo to GitHub, create a new Streamlit app pointing at `demo.py`, and keep the working directory as repo root. The default command is `streamlit run demo.py --server.headless true`.

Usage
-----
1) Install dependencies: `python -m pip install -r requirements.txt`
2) Train primary classifier (saves weights + metrics under `models/`): `python training/train_classifier.py`
3) Train auditor, compute ROC-AUC, and generate plots/cache: `python training/train_auditor.py`
   - Outputs: `auditor/auditor_mlp.pt`, `auditor/metrics.json`, `visualization/latent_cache.npz`, and PNG plots.
4) Launch demo: `streamlit run demo.py --server.headless true`
   - Controls: choose sample index, corruption type, and severity; view prediction, confidence, auditor decision, and latent PCA view (cached PCA from training).
5) Generate static snapshot without UI (finds a confident failure): `python demo.py --snapshot`

Repository Layout
-----------------
- `models/primary.py`: CNN with latent capture hooks.
- `training/train_classifier.py`: primary training loop on MNIST.
- `training/train_auditor.py`: corruption sampling, latent extraction, auditor training, metrics, and plots.
- `auditor/auditor_model.py`: MLP failure detector.
- `visualization/`: generated plots and latent cache used by the demo.
- `demo.py`: Streamlit UI and snapshot generator.
