from setuptools import setup, find_packages

setup(
    name="data_validator",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["pandas", "numpy", "matplotlib", "scipy"],
    entry_points={
        "console_scripts": [
            "data-validator=main:main",
        ],
    },
)
