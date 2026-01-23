"""
Inference Engine.
Wraps Model + Whitelist + Rules into a single decision maker.
"""
import pandas as pd
import numpy as np
import pickle
import re
import requests  # <--- Needed for redirects
from urllib.parse import urlparse
from src.config import MODEL_DIR
from src.utils import load_whitelist
from src.feature_engineering import extract_url_features, calculate_uts

class PhishingPredictor:
    def __init__(self, model_name='xgboost'):
        self.model_name = model_name
        self.model = None
        self.whitelist = load_whitelist()
        self.load_model()

        # 1: The "Bad Neighborhood" TLDs
        # These are rarely used by legit banks/tech companies
        self.SUSPICIOUS_TLDS = ['.xyz', '.tk', '.ml', '.ga', '.cf', '.gq', '.top', '.club', '.vip']
        
        # 2: Sensitive Keywords
        self.SENSITIVE_KEYWORDS = ['login', 'verify', 'account', 'update', 'secure', 'banking', 'signin', 'confirm', 'wallet']

        # 3. URL SHORTENERS (Domains Only)
        # We stick to known domains to avoid false positives (like 'bit.' matching 'orbit.com')
        self.SHORTENERS = {
            'bit.ly', 'goo.gl', 'tinyurl.com', 'is.gd', 'cli.gs', 't.co', 
            'tr.im', 'ow.ly', 'bit.do', 'x.co'
        }
        
    def load_model(self):
        target_path = MODEL_DIR / f"{self.model_name}.pkl"
        if not target_path.exists():
            return
        try:
            with open(target_path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            print(f"❌ Error loading pickle file: {e}")
            
    def resolve_redirect(self, url):
        """
        Follows the URL to see where it lands (e.g., bit.ly -> google.com).
        Returns the FINAL destination URL.
        """
        try:
            # We use a 'HEAD' request which is fast and doesn't download the file body
            response = requests.head(url, allow_redirects=True, timeout=3)
            return response.url
        except:
            # If we can't resolve it (e.g. offline), return original
            return url

    def predict(self, url):
        # ---------------------------------------------------------
        # STEP 0: LINK EXPANSION (The "Scout")
        # ---------------------------------------------------------
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # EXACT MATCH CHECK (Prevents 't.co' matching 'microsoft.com')
            if domain.replace('www.', '') in self.SHORTENERS:
                print(f"   [INFO] Shortener detected ({domain}). Resolving...")
                final_url = self.resolve_redirect(url)
                
                # RECURSION: If it changed, analyze the new URL
                if final_url != url:
                    print(f"   [INFO] Redirected to: {final_url}")
                    
                    # --- 🚨 NEW! CHECK FOR WARNING PAGES ---
                    # Bitly/Google/TinyURL redirect to these pages if the link is bad.
                    lower_dest = final_url.lower()
                    if "warning" in lower_dest or "abuse" in lower_dest or "blocked" in lower_dest:
                        return {
                            "url": url,
                            "status": "PHISHING",
                            "confidence": 100.0,
                            "reason": "Redirected to Abuse/Warning Page",
                            "features": {}
                        }
                    # ---------------------------------------
                        
                    result = self.predict(final_url)
                    result['reason'] = f"{result['reason']} (Resolved from Shortener)"
                    return result
        except:
            pass # Continue if parsing fails
        
        # ---------------------------------------------------------
        # 1. EXTRACT FEATURES
        # ---------------------------------------------------------
        features = extract_url_features(url)
        
        # --- DEBUGGING: SEE WHAT THE SYSTEM SEES ---
        # This will print in your terminal so you know WHY it failed/passed
        risk_score = features.get('typo_similarity', 0)
        is_official = features.get('is_known_brand', 0)
        has_ip = features.get('has_ip', 0)
        url_lower = url.lower()
        
        print(f"   [DEBUG] Risk: {risk_score:.2f} | Official: {is_official} | URL: {url}")

        # ---------------------------------------------------------
        # 2. WHITELIST CHECK (The "Green Lane")
        # ---------------------------------------------------------
        if is_official == 1:
            return {
                "url": url,
                "status": "SAFE",
                "confidence": 100.0,
                "reason": "Official Whitelisted Domain",
                "features": features
            }

        # ---------------------------------------------------------
        # 3. RULE OVERRIDES (The "Red Lane")
        # ---------------------------------------------------------
        
        # Rule A: Raw IP Address (e.g. http://192.168.0.1)
        if has_ip == 1:
            return {
                "url": url,
                "status": "PHISHING",
                "confidence": 100.0,
                "reason": "Suspicious Raw IP Address",
                "features": features
            }
        
        # Rule B: TLD PENALTY (Fix for 'login-verify.ga/account')
        # Logic: If URL ends in a "Bad TLD" exists as an ending OR followed by a slash AND contains a "Sensitive Keyword", block it.
        # Example: secure-banking-login.ml
        for tld in self.SUSPICIOUS_TLDS:
            # Check ".ga" at end OR ".ga/" in middle
            if url_lower.endswith(tld) or (tld + "/" in url_lower):
                # Only flag if it ALSO tries to sound legit (has keywords)
                if any(kw in url_lower for kw in self.SENSITIVE_KEYWORDS):
                    return {
                        "url": url,
                        "status": "PHISHING",
                        "confidence": 100.0,
                        "reason": f"Suspicious TLD ({tld}) + Sensitive Keyword",
                        "features": features
                    }

        # Rule C: Typosquatting & Keyword Traps
        # Catches 'paypa1.com' AND 'paypal-verify.com'
        # We lowered threshold to 0.70 to be safe
        if risk_score >= 0.70:
            return {
                "url": url,
                "status": "PHISHING",
                "confidence": 100.0,
                "reason": f"High-Risk Brand Impersonation (Score: {risk_score:.2f})",
                "features": features
            }

        # Rule D: MESSY URL HEURISTIC (Fix for 'suspicious-long-url...')
        # If URL is huge and messy, it's likely machine-generated.
        if len(url) > 75 and url.count('-') > 3:
             return {
                "url": url,
                "status": "PHISHING",
                "confidence": 90.0,
                "reason": "Suspicious URL Structure (Excessive Length/Hyphens)",
                "features": features
            }
            
        # Removed Not using anymore
        # Rule E: URL Shorteners (Hidden Destination) 
        # Hackers love these to bypass filters.
        #if any(short in url_lower for short in self.SHORTENERS):
        #     return {
        #        "url": url,
        #        "status": "PHISHING",
        #        "confidence": 95.0,
        #        "reason": "URL Shortener Detected (Hidden Destination)",
        #        "features": features
        #    }
            
        # ---------------------------------------------------------
        # 4. AI PREDICTION (The Brain)
        # ---------------------------------------------------------
        if self.model is None:
             # Fallback if model is missing
             return {"status": "UNKNOWN", "reason": "Model not loaded"}

        # Prepare Data
        df = pd.DataFrame([features])
        df['UTS'] = calculate_uts(df)
        
        # Align Columns
        if hasattr(self.model, 'feature_names_in_'):
            expected_cols = self.model.feature_names_in_
            model_df = pd.DataFrame()
            for col in expected_cols:
                model_df[col] = df[col] if col in df.columns else 0
        else:
            model_df = df

        try:
            prob = self.model.predict_proba(model_df)[0, 1]
            status = "PHISHING" if prob >= 0.8 else "SAFE"
            
            return {
                "url": url,
                "status": status,
                "confidence": prob * 100 if status == "PHISHING" else (1 - prob) * 100,
                "reason": "AI Pattern Analysis",
                "features": features
            }
        except Exception as e:
            return {"status": "ERROR", "reason": str(e)}
