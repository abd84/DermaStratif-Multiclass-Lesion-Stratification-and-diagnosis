# Production Configuration for DermaStratif

import os
from datetime import timedelta

# Flask Configuration
class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    
    # File upload
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 52428800))  # 50MB
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'}
    
    # Model
    MODEL_PATH = os.environ.get('MODEL_PATH', './Saved Models/best_model1_lora.pth')
    DEVICE = os.environ.get('DEVICE', 'auto')
    
    # Optional cloud vision analysis
    VISION_API_KEY = os.environ.get('VISION_API_KEY') or os.environ.get('GEMINI_API_KEY')


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    SESSION_COOKIE_SECURE = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    SESSION_COOKIE_SECURE = False


def get_config():
    """Get appropriate config based on environment"""
    env = os.environ.get('FLASK_ENV', 'production')
    
    if env == 'development':
        return DevelopmentConfig()
    elif env == 'testing':
        return TestingConfig()
    else:
        return ProductionConfig()
