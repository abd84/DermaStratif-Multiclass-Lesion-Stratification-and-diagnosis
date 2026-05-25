# DermaStratif - Deployment Guide

This guide covers deploying the DermaStratif skin lesion classification application in different environments.

---

## Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- Git
- 2GB+ available disk space
- GPU (optional, but recommended for faster inference)

---

## Quick Start - Local Deployment

### 1. Clone and Setup

```bash
git clone https://github.com/abd84/Derma-Stratif-Inference.git
cd Derma-Stratif-Inference
```

### 2. Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Optional: edit .env and set VISION_API_KEY for cloud vision analysis
```

### 5. Run Application

```bash
cd Application
python app.py
```

The app will be available at `http://localhost:5000`

---

## Docker Deployment

### Build and Run with Docker Compose

```bash
docker-compose up --build
```

Access at `http://localhost:5000`

### Stop the Service

```bash
docker-compose down
```

### Using Docker Directly

```bash
# Build
docker build -t dermastratif:latest .

# Run
docker run -p 5000:5000 \
  -e VISION_API_KEY="your-api-key" \
  -v uploads:/app/uploads \
  dermastratif:latest
```

---

## Production Deployment

### Using Gunicorn (Recommended)

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
cd Application
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Nginx as Reverse Proxy

Create `/etc/nginx/sites-available/dermastratif`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 50M;
    }

    location /static {
        alias /path/to/app/static;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/dermastratif /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Environment type (production/development) | production |
| `FLASK_DEBUG` | Enable debug mode | False |
| `SECRET_KEY` | Flask secret key for sessions | (required) |
| `VISION_API_KEY` | Optional cloud vision analysis API key | (optional) |
| `HOST` | Bind address | 0.0.0.0 |
| `PORT` | Port number | 5000 |
| `MODEL_PATH` | Path to saved model | ./Saved Models/best_model1_lora.pth |
| `DEVICE` | Compute device (auto/cpu/cuda/mps) | auto |
| `MAX_CONTENT_LENGTH` | Max upload size in bytes | 52428800 (50MB) |

---

## Model Files

Ensure the following model file exists in `Saved Models/`:

- **best_model1_lora.pth** - LoRA fine-tuned model (recommended - 80% accuracy)

Optional models:
- **best_model_base.pth** - Base model (74% accuracy)

---

## Performance Tuning

### For Better Performance

1. **Use GPU** - Set `DEVICE=cuda` (NVIDIA) or `mps` (Apple Silicon)
2. **Increase Workers** - Use Gunicorn with more workers: `gunicorn -w 8`
3. **Enable Caching** - Cache model in memory to avoid reloading
4. **Optimize Image Size** - Process images at reduced resolution for speed

### Monitoring

Monitor application logs:
```bash
docker-compose logs -f dermastratif
```

---

## Database Backup

The application doesn't use a database by default, but uploads are stored in `uploads/`. Back them up regularly:

```bash
tar -czf uploads_backup_$(date +%Y%m%d).tar.gz uploads/
```

---

## SSL/TLS Configuration

For HTTPS, use Let's Encrypt with Certbot:

```bash
sudo certbot certonly --standalone -d your-domain.com
```

Update Nginx config to use certificates:

```nginx
listen 443 ssl;
ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
```

---

## Troubleshooting

### Issue: Model not found
- Ensure `Saved Models/best_model1_lora.pth` exists
- Check `MODEL_PATH` environment variable

### Issue: Vision API key not working
- Verify `VISION_API_KEY` is set correctly in `.env`
- Check API quota and permissions with your provider
- App still works via local EfficientNet-LoRA fallback if the key is missing

### Issue: Out of memory
- Reduce number of Gunicorn workers
- Use a smaller batch size in the application

### Issue: Slow predictions
- Enable GPU acceleration with `DEVICE=cuda` or `mps`
- Check system resources: `top` or `docker stats`

---

## Security Checklist

- [ ] Change `SECRET_KEY` in production
- [ ] Use HTTPS/SSL certificates
- [ ] Restrict file upload types and sizes
- [ ] Validate user input
- [ ] Use strong passwords for any databases
- [ ] Keep dependencies updated
- [ ] Enable CORS only for trusted origins
- [ ] Use environment variables for secrets (never commit .env)
- [ ] Set appropriate file permissions
- [ ] Monitor and log access

---

## Support

For issues and questions:
1. Check logs: `docker-compose logs`
2. Review SETUP_GUIDE.md for initial setup
3. Check README.md for project overview
