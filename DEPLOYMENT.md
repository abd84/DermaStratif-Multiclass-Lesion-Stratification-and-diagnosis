# Deployment

## Vercel (Flask)

1. Import [abd84/Derma-Stratif-Inference](https://github.com/abd84/Derma-Stratif-Inference) in Vercel.
2. Connect the **abd84** GitHub account (not tech-abdullahnaeem) under **Vercel → Settings → Git**.
3. Framework: **Other** or auto-detect Python; root directory: **/** (repo root).
4. Environment variables (optional):
   - `VISION_API_KEY` — cloud vision analysis
   - `MODEL_PATH` — defaults to `Saved Models/best_model1_lora.pth`
5. Redeploy after pushes to `main`.

Entrypoints: `app.py`, `api/index.py`, and `pyproject.toml` (`app:app`). `vercel.json` rewrites all routes to the Flask function.

**Note:** PyTorch + model weights are large. If the build exceeds Vercel’s function size limit, use Docker/Railway/Render instead (see below).

---

## Local

```bash
make setup
make dev
```

Open http://127.0.0.1:5001

---

## Docker

```bash
docker-compose up --build
```

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `VISION_API_KEY` | Optional multimodal vision API key |
| `MODEL_PATH` | Path to LoRA weights |
| `FLASK_ENV` | `development` or `production` |
| `DEVICE` | `auto`, `cpu`, `cuda`, or `mps` |

---

## Troubleshooting

**Vercel 404 NOT_FOUND** — Usually no Flask handler mounted. Ensure latest `main` is deployed (includes `vercel.json` rewrites). Check build logs for Python/Flask detection errors.

**Wrong GitHub user on Vercel** — Commits must use `136109231+abd84@users.noreply.github.com`. Reconnect the abd84 Git integration in Vercel.

**Model not found** — Confirm `Saved Models/best_model1_lora.pth` is in the repo and `MODEL_PATH` is correct.
