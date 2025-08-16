
# AIDI 1002 Final Project — Lightweight TextCNN (Reproduction + Contribution)

**Paper:** *Light-Weighted CNN for Text Classification* (Yadav, 2020)  
**ArXiv:** https://arxiv.org/abs/2004.07922  
**Project Date:** 2025-08-15

This repository contains a reproducible implementation of a **Lightweight TextCNN** for text classification and a **significant contribution** beyond the paper:
- A baseline **MLP** model for comparison
- A **dual-optimizer** training schedule (Adam → SGD on validation plateau)
- Optional **FAST mode** for quick development runs on CPU

The project uses the **AG_NEWS** dataset via **HuggingFace `datasets`** (no `torchtext` required).

---

## Repository Contents

- `Final_Project_AGNEWS_HF.ipynb` – End-to-end notebook (HuggingFace-based).  
  - Loads and preprocesses AG_NEWS  
  - Implements Lightweight TextCNN (depthwise separable 1D conv)  
  - Trains with Adam→SGD schedule and evaluates on test set  
  - Trains and evaluates an MLP baseline  
  - Saves model weights and a results JSON to `./artifacts`  
  - Includes optional learning-curve plots

- `artifacts/` – Created after running the notebook; contains model weights and metrics:
  - `lightweight_textcnn.pt`
  - `mlp_text.pt`
  - `results.json`

---

## How to Run (Local or Colab)

### 1) Install dependencies (first time only)
```bash
pip install datasets scikit-learn torch torchvision torchaudio
# (Optional) for plots/widgets
pip install matplotlib ipywidgets
```

> Windows note: you may see a HuggingFace cache symlink warning. It’s harmless; caching still works.  
> To silence it: set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` in your environment.

### 2) Open and run the notebook
- Launch Jupyter Lab/Notebook or open in Google Colab.
- Open **`Final_Project_AGNEWS_HF.ipynb`**.
- Run cells **top to bottom**. The notebook automatically detects CPU/GPU.

### 3) (Optional) FAST Mode for quick iterations
Inside the notebook you can enable a **fast smoke test** on a small subset with a smaller model and just a couple of epochs.  
This is helpful on CPU to verify everything works before a full training run.

---

## Output & Artifacts

After a successful run, you’ll get:
- **Console logs** per epoch for train/val metrics and optimizer state (Adam → SGD when plateau).  
- **Test metrics** (accuracy, macro-F1) for both TextCNN and MLP.  
- **Saved artifacts** under `./artifacts`:
  - `lightweight_textcnn.pt` (TextCNN weights)  
  - `mlp_text.pt` (MLP weights)  
  - `results.json` (metrics + parameter counts)

Example `results.json` structure:
```json
{
  "TextCNN_test": {"loss": 0.65, "acc": 0.89, "f1m": 0.88},
  "MLP_test":     {"loss": 0.90, "acc": 0.83, "f1m": 0.82},
  "TextCNN_params": 1234567,
  "MLP_params":  9876543
}
```

---

## Notes & Tips

- **CPU is fine** but slower. Use the notebook’s FAST mode for quick checks.
- For **better final accuracy**, use the full model (`embed_dim=128`, `channels=64`) and more epochs (e.g., 12–15).
- You can easily add **pretrained embeddings** (e.g., GloVe) or try different kernel sizes (`(2,3,4,5)`) as ablations.

---

## Academic Submission Checklist

- [ ] Public GitHub repository with this README and the notebook
- [ ] Notebook runs end-to-end and saves artifacts
- [ ] Report/summary (per course template) includes:
  - Dataset & preprocessing details
  - Model architecture & hyperparameters
  - Training strategy (Adam→SGD) and rationale
  - Test metrics for both models and comparison
  - Insights & any ablations

---

## References

- Ritu Yadav (2020). *Light-Weighted CNN for Text Classification.* arXiv:2004.07922.
- AG_NEWS dataset (news topic classification), available via HuggingFace `datasets`.

---
