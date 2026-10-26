"""CLIP prompt<->image matching pipeline for the SAR missing-person profiles.

Trains/fine-tunes CLIP ViT-B/16 (LoRA via IRRA's model/petl.py) to align the
auto-generated CLIP prompts with their reference crops, and renders Trail-style
similarity heatmaps (before vs after) + a numeric table into Dataset/Test.
"""
