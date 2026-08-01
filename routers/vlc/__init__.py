"""
VLC (Vision-Language Compositional) Router

Routing with pretrained vision-language models:
- VisualBERT: single-stream architecture, visual embeddings + text tokens
- LXMERT: dual-stream architecture, separate vision/language encoders + cross-modality encoder

Reference: BERT router (text handling) + MobileNet router (image handling)
"""

from .router import VLCRouter

__all__ = ['VLCRouter']

