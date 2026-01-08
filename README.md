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
- Failure detection (corrupted MNIST mix, n=40,960):
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

Usage
-----
1) Install dependencies: `python -m pip install -r requirements.txt`
2) Train primary classifier: `python training/train_classifier.py`
3) Train auditor + plots: `python training/train_auditor.py`
4) Launch demo: `streamlit run demo.py --server.headless true`
   - Controls: choose sample index, corruption type, and severity; view prediction, confidence, auditor decision, and latent PCA view.
5) Generate static snapshot without UI: `python demo.py --snapshot`

Repository Layout
-----------------
- `models/primary.py`: CNN with latent capture hooks.
- `training/train_classifier.py`: primary training loop on MNIST.
- `training/train_auditor.py`: corruption sampling, latent extraction, auditor training, metrics, and plots.
- `auditor/auditor_model.py`: MLP failure detector.
- `visualization/`: generated plots and latent cache used by the demo.
- `demo.py`: Streamlit UI and snapshot generator.
