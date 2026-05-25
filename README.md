# DermaStratif Inference

Web app for **8-class skin lesion stratification** with structured clinical-style reports. Combines multimodal vision analysis with an **EfficientNet-B0 + LoRA** classifier (ISIC 2019 taxonomy) and a local CNN fallback.

**Repository:** [github.com/abd84/Derma-Stratif-Inference](https://github.com/abd84/Derma-Stratif-Inference)

---

## Features

- 8 lesion classes (melanoma, BCC, SCC, AK, benign keratosis, dermatofibroma, vascular lesion, nevus)
- Dual pipeline: cloud vision analysis + EfficientNet-LoRA fallback
- Upload validation, risk tiers, 12+ report sections per scan
- Flask UI — scan, report, and about pages

---

## Quick start

```bash
git clone https://github.com/abd84/Derma-Stratif-Inference.git
cd Derma-Stratif-Inference

make setup
make dev
```

Open **http://127.0.0.1:5001**

Optional: copy `.env.example` to `.env` and set `VISION_API_KEY` for cloud vision analysis (falls back to the local LoRA model if unset).

See [QUICKSTART.md](QUICKSTART.md) and [DEPLOYMENT.md](DEPLOYMENT.md) for Docker and production.

---

## Model performance (validation)

| Metric | Score |
|--------|--------|
| Accuracy | 95.0% |
| F1 (macro) | 93.3% |
| Precision (macro) | 94.8% |
| Recall (macro) | 92.2% |

CNN-only LoRA checkpoint (fallback): **80%** validation accuracy on the same 8-class task.

Weights: `Saved Models/best_model1_lora.pth` (required), `best_model_base.pth` (optional).

---

## Project layout

```
├── Application/          # Flask app (app.py, templates, static)
├── Saved Models/         # Trained .pth weights
├── Model Training Scripts/
├── Notebooks/
├── requirements.txt
├── Makefile
├── Dockerfile
└── docker-compose.yml
```

---

## Training methods (research)

1. **Base** EfficientNet-B0 fine-tune — 74% val accuracy  
2. **Adapter** fine-tune — 76%  
3. **LoRA** fine-tune (deployed) — 80% CNN checkpoint; full pipeline tuned to 95% / 93%+ F1  

Dataset: [ISIC 2019](https://challenge.isic-archive.com/data/#2019)

---

## Disclaimer

For **education and early awareness only**. Not a substitute for professional diagnosis. Always confirm with a board-certified dermatologist.

---

## License

MIT — see [LICENSE](LICENSE).
