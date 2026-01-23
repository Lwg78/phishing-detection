import pytest
from src.feature_engineering import extract_url_features
from src.utils import load_whitelist

def test_whitelist_loading():
    """Test if whitelist loads correctly."""
    wl = load_whitelist()
    assert isinstance(wl, dict)
    assert 'exact_matches' in wl

def test_feature_extraction_safe():
    """Test a known safe URL."""
    url = "https://www.google.com"
    features = extract_url_features(url)
    assert features['is_known_brand'] == 1
    assert features['typo_similarity'] == 0.0

def test_feature_extraction_phishing():
    """Test a known phishing pattern (IP Address)."""
    url = "http://192.168.0.1/login"
    features = extract_url_features(url)
    assert features['has_ip'] == 1

def test_typosquatting_logic():
    """Test if our typo logic catches 'faceb00k'."""
    # This relies on your updated feature_engineering logic
    url = "http://faceb00k.com"
    features = extract_url_features(url)
    # Depending on your exact logic, risk should be > 0
    assert features['typo_similarity'] > 0
