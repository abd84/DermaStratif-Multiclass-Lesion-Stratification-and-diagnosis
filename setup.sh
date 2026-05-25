#!/bin/bash

# DermaStratif Deployment Helper Script

set -e

echo "🚀 DermaStratif Deployment Setup"
echo "=================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📥 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create .env if not exists
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Optional: edit .env and add VISION_API_KEY for cloud vision analysis"
fi

# Create uploads directory
mkdir -p uploads
mkdir -p logs

# Check if models exist
if [ ! -f "Saved Models/best_model1_lora.pth" ]; then
    echo "⚠️  Warning: best_model1_lora.pth not found!"
    echo "   Please ensure the model file is in Saved Models/"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run: make dev"
echo "2. Open: http://127.0.0.1:5001"
echo ""
echo "Or use Docker:"
echo "  docker-compose up --build"
echo ""
