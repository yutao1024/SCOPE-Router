"""
CosineCLS Router

Dual-contrastive learning framework based on RouterDC, using LXMERT as the vision-language encoder.

Key features:
- Use LXMERT to encode (image, text) -> embedding
- Learn trainable LLM embeddings
- Dual contrastive losses: Sample-LLM + Sample-Sample
- Use raw text and images as inputs
"""

from .router import CosineCLSRRouter

__all__ = ['CosineCLSRRouter']

