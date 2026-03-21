"""
OCR submodule for extracting player names from tennis videos.

This module provides text extraction from video frames using OCR
and matches detected text against known player names.
"""

from .video_text_extractor import VideoTextExtractor

__all__ = ['VideoTextExtractor']
