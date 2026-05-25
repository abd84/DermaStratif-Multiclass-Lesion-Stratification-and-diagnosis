.PHONY: help install dev prod test lint format clean docker-build docker-up docker-down deploy

help:
	@echo "DermaStratif - Available Commands"
	@echo "=================================="
	@echo "make install       - Install dependencies"
	@echo "make dev           - Run in development mode"
	@echo "make prod          - Run in production mode (gunicorn)"
	@echo "make test          - Run tests"
	@echo "make lint          - Lint code"
	@echo "make format        - Format code with black"
	@echo "make clean         - Clean up temporary files"
	@echo "make docker-build  - Build Docker image"
	@echo "make docker-up     - Start Docker container"
	@echo "make docker-down   - Stop Docker container"
	@echo "make deploy        - Deploy to production"

install:
	pip install -r requirements.txt

dev:
	@cd Application && FLASK_PORT=5001 FLASK_ENV=development python app.py

prod:
	@cd Application && gunicorn -w 4 -b 0.0.0.0:5000 app:app

test:
	pytest tests/ -v --cov=Application

lint:
	pylint Application/ --disable=all --enable=E,F

format:
	black Application/ --line-length=120

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

docker-build:
	docker build -t dermastratif:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

deploy: clean lint test docker-build docker-up
	@echo "✅ Deployment complete!"

setup:
	bash setup.sh

.env:
	cp .env.example .env
	@echo "⚠️  Please edit .env with your settings"
