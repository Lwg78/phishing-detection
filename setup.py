"""
Setup configuration for pip install -e .
Allows the 'src' folder to be imported anywhere in the project.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for the long description (good practice)
# We wrap this in try-except in case README is missing during CI/CD
try:
    readme_file = Path(__file__).parent / "README.md"
    long_description = readme_file.read_text() if readme_file.exists() else ""
except Exception:
    long_description = "Phishing Detection System"

setup(
    name="phishing-detection",
    version="1.0.0",
    author="Lim Wen Gio",
    author_email="lim_wengio@hotmail.com",
    description='Hybrid Phishing Detection System using ML and Heuristics',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lwg78/phishing-detection",
    license='MIT',

    # Automatically finds the 'src' package
    packages=find_packages(exclude=["tests", "notebooks"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    
    # --- CRITICAL DEPENDENCIES ---   
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.7.0",
        'requests>=2.28.0',
        "lightgbm>=3.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "joblib>=1.1.0",
        "tqdm>=4.60.0",
    ],
    
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=21.0",
            "flake8>=4.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "phishing-train=scripts.train_individual:main",
            "phishing-predict=scripts.predict:main",
        ],
    },
)
