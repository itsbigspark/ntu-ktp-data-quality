#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# duplicate_detection_tool/__init__.py

from .preprocessing import preprocess_text
from .processing import process_data
from .similarity import detect_duplicates_and_similars
from .clustering import run_clustering, cluster_analysis

__all__ = [
    "preprocess_text",
    "process_data",
    "detect_duplicates_and_similars",
    "run_clustering",
    "cluster_analysis"
]


