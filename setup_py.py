"""
Setup script for Trustworthy Distributed Deep Learning
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="trustworthy-distributed-dl",
    version="0.1.0",
    author="Haiying Shen, Tanmoy Sen, Suraiya Tairin",
    author_email="your.email@virginia.edu",
    description="Trustworthy Distributed Deep Learning with Adversarial Attack Mitigation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/trustworthy-distributed-dl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "mypy>=1.4.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "experiments": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "trustworthy-dl-train=trustworthy_dl.cli:main",
            "trustworthy-dl-experiment=trustworthy_dl.experiments.runner:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)