"""
Feature engineering pipeline.
Implements Levenshtein (Typos), Entropy (Char Prob), and UTS (Scoring).
"""
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from src.utils import load_whitelist
from src.config import UTS_WEIGHTS

# --- LOAD CONFIG ONCE ---
WHITELIST_DATA = load_whitelist()

# 1. Exact Matches (Fast Set Lookup)
EXACT_MATCHES_DICT = WHITELIST_DATA.get('exact_matches', {})
SAFE_DOMAINS = set()
for category, domains in EXACT_MATCHES_DICT.items():
    for domain in domains:
        SAFE_DOMAINS.add(domain)

# 2. Typosquatting Targets
SENSITIVE_BRANDS_LIST = WHITELIST_DATA.get('sensitive_brands_for_typo_check', [])
# 3. Safe Extensions
SAFE_EXTENSIONS = WHITELIST_DATA.get('safe_extensions', [])
#for safe_ext in SAFE_EXTENSIONS:  # Contains ['.gov.sg', '.edu.sg']
#    if domain.endswith(safe_ext):
#        return 1, 0.0  # ✅ SAFE (Official Government Domain)

# ==========================================
# 🧠 LOGIC 1: LEVENSHTEIN DISTANCE (Typosquatting)
# ==========================================
def levenshtein_distance(s1, s2):
    """
    Calculates the 'Edit Distance' between two strings.
    Example: 'google' vs 'g00gle' -> Distance 2
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

# ==========================================
# 🧠 LOGIC 2: ENTROPY / CHAR PROBABILITY
# ==========================================
def calculate_char_probability(domain):
    """
    Calculates the 'Randomness' of a domain.
    Technically: Sum of Squared Probabilities (Homogeneity Index).
    
    - High Score (~0.1+) = Structured (e.g. 'amazon') -> SAFE
    - Low Score (<0.04)  = Random (e.g. 'wqz-9a')   -> PHISHING
    """
    if not domain: return 0.0
    
    # Remove dots/hyphens to analyze just the letters
    clean = re.sub(r'[.-]', '', str(domain).lower())
    if not clean: return 0.0
    
    # Count frequency of each character
    char_counts = {}
    for c in clean:
        char_counts[c] = char_counts.get(c, 0) + 1
        
    # Calculate sum of squared probabilities
    prob_sum = 0
    total_len = len(clean)
    for count in char_counts.values():
        prob = count / total_len
        prob_sum += prob ** 2 
        
    return prob_sum

# ==========================================
# 🧠 LOGIC 3: UTS (URL Typical Score)
# ==========================================
def calculate_uts(df):
    """
    Calculates a weighted suspicion score (0-100).
    """
    uts_score = 0.0
    
    for feature, weight in UTS_WEIGHTS.items():
        if feature in df.columns:
            # Force numeric and fill errors with 0
            val = pd.to_numeric(df[feature], errors='coerce').fillna(0)
            
            # Dynamic Normalization (Scale to 0-1)
            # If the max value in the column is 10, we divide by 10.
            max_val = val.max() if val.max() > 0 else 1
            normalized = val / max_val
            
            uts_score += normalized * weight
            
    # Scale final result to 0-100 for readability
    return uts_score * 100


# ==========================================
# 🛠️ HELPER: BRAND RISK ANALYZER
# ==========================================
def get_brand_features(domain):
    """
    Orchestrates the Whitelist and Levenshtein logic.
    """
    domain = str(domain).lower()
    clean_domain = domain.replace('www.', '')
    
    # A. WHITELIST CHECK (Exact + Subdomain Wildcard)
    # 1. Exact Match
    if domain in SAFE_DOMAINS or clean_domain in SAFE_DOMAINS:
        return 1, 0.0 # Known Brand, Zero Risk
    
    # 2. Extension Match
    for safe_ext in SAFE_EXTENSIONS:
        if domain.endswith(safe_ext):
            return 1, 0.0

    # 3. Subdomain Match (NEW FIX)
    # Allows 'gemini.google.com' because it ends with '.google.com'
    for safe in SAFE_DOMAINS:
        if domain.endswith("." + safe):
            return 1, 0.0

                
    # B. SUBDOMAIN TRAP DETECTION (New!)
    # Catches 'amazon.com.verify.xyz'
    # If a safe domain appears INSIDE the URL, but it wasn't caught by the whitelist check above,
    # it implies it is being used as a subdomain on a fake site.
    for safe in SAFE_DOMAINS:
        if safe in clean_domain:
            # We found 'amazon.com' inside 'amazon.com.verify.xyz'
            # Since we already passed the Whitelist Check (A), we know this isn't the REAL amazon.com
            return 0, 1.0 # MAX RISK

    # C. Check Typosquatting (Levenshtein Loop)
    min_dist = 100
    parts = re.split(r'[.-]', clean_domain)
    
    for part in parts:
        # Skip generic terms
        if part in ['com', 'org', 'net', 'login', 'secure', 'account', 'verify', 'update']: continue
        
        for brand in SENSITIVE_BRANDS_LIST:
            # Optimization: Don't compare if lengths are wildly different
            if abs(len(brand) - len(part)) > 2: continue
            if len(brand) < 3: continue
            
            dist = levenshtein_distance(part, brand)
            
            # Trap: 'citibank' hidden inside 'secure-citibank-login'
            if brand in part and part != brand:
                dist = 0 
            
            if dist < min_dist:
                min_dist = dist
                
    # Normalize Risk Score
    if min_dist == 0: return 0, 1.0  # High Risk (Impersonation)
    if min_dist == 1: return 0, 0.9  # High Risk (Typo)
    if min_dist == 2: return 0, 0.6  # Medium Risk
    
    return 0, 0.0

# ==========================================
# 🚀 MAIN EXTRACTOR
# ==========================================
def extract_url_features(url):
    if pd.isna(url): return pd.Series({})
    url = str(url)
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]
    except:
        return pd.Series({'url_length': len(url)}) 

    # 1. Run Brand Logic
    is_official, brand_risk = get_brand_features(domain)
    
    # 2. Run Entropy Logic
    char_prob = calculate_char_probability(domain)
    
    # 3. Basic RegEx Features
    has_ip = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', domain) else 0
    has_https = 1 if parsed.scheme == 'https' else 0
    
    return pd.Series({
        'url_length': len(url),
        'domain_length': len(domain),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_at': url.count('@'),
        'num_slashes': url.count('/'),
        'num_questionmarks': url.count('?'),
        'num_equals': url.count('='),
        'num_ampersand': url.count('&'),
        'num_underscores': url.count('_'),
        'num_hash': url.count('#'),
        'num_percent': url.count('%'),
        'has_ip': has_ip,
        'has_https': has_https,
        'char_probability': char_prob,  # <--- ENTROPY FEATURE
        'is_known_brand': is_official,
        'typo_similarity': brand_risk
    })
