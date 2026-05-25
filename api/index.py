"""Vercel serverless entry — routes all traffic to the Flask app."""
from Application.app import app

__all__ = ["app"]
