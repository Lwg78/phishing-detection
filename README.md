# Phishing Website Detection System 🛡️

![CI Status](https://github.com/lwg78/phishing-detection/actions/workflows/ci.yml/badge.svg)

A production-ready, hybrid phishing detection engine achieving **99.32% F1-score**.  
Combines **Machine Learning (XGBoost)** with a **Real-time Heuristic Rule Engine** to detect zero-day phishing attacks, typosquatting, and obfuscated URLs.

---

## 🎯 Key Features

### 1. 🧠 Hybrid "3-Layer" Architecture
Unlike standard ML models that just guess, this system uses a strict 3-layer defense protocol:
* **Layer 1 (The Green Lane):** Instant pass for Official Whitelisted Domains (e.g., `.gov.sg`, `dbs.com`, `google.com`).
* **Layer 2 (The Red Lane):** Hard-coded blocking of high-risk indicators:
    * **IP Addresses:** Blocks raw IPs (e.g., `192.168.0.1`).
    * **Subdomain Traps:** Catches `amazon.com.verify.xyz`.
    * **TLD Penalties:** Flags dangerous TLDs (`.ml`, `.ga`, `.xyz`) paired with sensitive keywords.
    * **Messy URLs:** Heuristic detection of machine-generated, excessive-length URLs.
* **Layer 3 (The AI Brain):** XGBoost model analyzes feature patterns (Entropy, UTS, Structure) for unknown URLs.

### 2. 🔗 Intelligent Link Expansion
* **Shortener Unmasking:** Automatically resolves `bit.ly`, `tinyurl.com`, `t.co`, etc.
* **Abuse Page Detection:** Inspects the destination. If the shortener redirects to a "Google/Bitly Warning Page," it is flagged as **PHISHING** immediately.

### 3. 🛡️ Advanced Feature Engineering
* **URL Typical Score (UTS):** A weighted scoring system for suspicion.
* **Domain Entropy:** Calculates character randomness to detect DGA (Domain Generation Algorithms).
* **Typosquatting Detection:** Levenshtein distance analysis to catch `cltibank.com` vs `citibank.com`.

---

## 🏗️ Project Structure

```text
phishing_detection/
├── config/               
│   └── whitelist.json         # ⚡ JSON Rules: Brands, Keywords, & Safe Extensions
├── data/
│   ├── raw/                   # Original SQLite database or CSVs
│   ├── processed/             # Cleaned training data
│   └── outputs/               # Prediction results (CSVs)
├── models/               
│   ├── xgboost.pkl            # 🧠 The Main AI Model (Production)
│   └── xgboost_url_only.pkl   # Lightweight Model (CLI fallback)
├── notebooks/                 # Jupyter notebooks for EDA/Experiments
├── src/                  
│   ├── __init__.py
│   ├── config.py              # Path definitions & Constants
│   ├── feature_engineering.py # Entropy, UTS, Typosquatting Logic
│   ├── utils.py               # Helper functions
│   ├── data/                  
│   │   ├── __init__.py
│   │   ├── dataloader.py      # 🔨 Load SQL/CSV -> Pandas
│   │   └── preprocessing.py   # 🔨 Train/Test Split & Cleaning
│   └── models/           
│       ├── __init__.py
│       ├── predict_model.py   # 🚀 The Logic Engine (Class)
│       ├── base_model.py      # Model definitions
│       └── evaluation.py      # 🔨 Calculate F1, Confusion Matrix
├── scripts/
│   ├── train_individual.py    # Script to train core models
│   ├── train_url_only.py      # Script to train the fast URL-text model
│   ├── predict.py             # CLI Tool for prediction
│   └── test_pipeline.py       # System Health Check
├── tests/                     # Unit tests (if any)
├── .gitignore                 # Files to ignore (e.g., venv, __pycache__)
├── my_test_urls.txt           # Your custom test list
├── validation_urls.txt        # Ground truth list for sanity checks
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── setup.py                   # Package installer
```

---

## 📊 Performance

| Model | Accuracy | F1-Score | ROC-AUC | Use Case |
|-------|----------|----------|---------|----------|
| **XGBoost** | **99.07%** | **0.9932** | **0.9982** | **Production (Fast & Accurate)** |
| Stacking Ensemble | 99.09% | 0.9934 | 0.9984 | Research (Max Accuracy) |
| Random Forest | 98.82% | 0.9914 | 0.9971 | Benchmark |

**Security Metrics:**
* **False Positive Rate:** ~0.0% on known legitimate brands (Google, Microsoft, SG Govt).
* **Detection Rate:** 100% on Brand Impersonation & IP-based attacks.

---

## 🚀 Installation & Usage

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# ⚠️ CRITICAL: Install project in editable mode (links 'src' folder)
pip install -e .
```

### 2. Training (Optional)
If you want to retrain the models from scratch:
```bash
# Train the specialized URL-Only model (Used for CLI)
python scripts/train_url_only.py --model xgboost
```

### 3. Prediction (CLI)
Use the smart prediction tool to scan URLs or files.

**Scan a Single URL:**
```bash
python scripts/predict.py --url "[http://secure-login.dbs.com.verify.ml](http://secure-login.dbs.com.verify.ml)"
```

**Scan a Text File:**
```bash
python scripts/predict.py --file my_test_urls.txt --output results.csv
```

**System Health Check:**
```bash
python scripts/test_pipeline.py
```

---

## 🔬 Technical Logic

### Character Probability (Domain Entropy)
We calculate the **distribution homogeneity** of characters to detect random strings (DGA algorithms):
```python
# Measures probability of character distribution (Simpsons Index)
Char_Prob = Σ (count(char_i) / len(domain))²
```
- **High Score (>0.06):** Legitimate domains (e.g., `google.com`)
- **Low Score (<0.04):** Random phishing domains (e.g., `x7z-9q.com`)

### Subdomain Trap Detection
Detects when a safe domain is used as a subdomain to trick users.
* **Legit:** `gemini.google.com` (Ends with `google.com` ✅)
* **Phishing:** `google.com.verify-login.xyz` (Contains `google.com` but ends with `.xyz` ❌)

---

## 📝 Citation

If you use this work, please cite:
```bibtex
@software{phishing_detection_2026,
  author = {Lim Wen Gio},
  title = {Phishing Website Detection using Hybrid ML & Heuristics},
  year = {2026},
  url = {[https://github.com/lwg78/phishing-detection](https://github.com/lwg78/phishing-detection)}
}
```
