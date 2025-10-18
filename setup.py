"""
Setup script for ECG Arrhythmia Classification package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ecg-arrhythmia-classification",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="ECG 4-Class Arrhythmia Classification with Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ecg-arrhythmia-classification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ecg-preprocess=scripts.preprocess_data:main",
            "ecg-train=scripts.train_model:main",
            "ecg-evaluate=scripts.evaluate_model:main",
        ],
    },
)