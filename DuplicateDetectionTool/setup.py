#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='duplicate_detection_tool',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'fuzzywuzzy',
        'python-Levenshtein',
        'metaphone',
        'imblearn',
        'xgboost'
    ],
    entry_points={
        'console_scripts': [
            'duplicate_detection_tool=duplicate_detection_tool.main:main',
        ],
    },
    author="Your Name",
    description="A package to detect duplicate, overly similar, and unique records in datasets.",
    url="https://github.com/yourusername/duplicate_detection_tool"
)


