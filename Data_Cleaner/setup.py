from setuptools import setup, find_packages

setup(
    name='datacleaner',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'fuzzywuzzy'
    ],
    entry_points={
        'console_scripts': [
            'datacleaner = main:main',
        ]
    },
    description='A tool for detecting and correcting errors in datasets.',
    author='Your Name',
    license='MIT',
    python_requires='>=3.7',
)
