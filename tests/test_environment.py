# Test functions
import sys


def test_python_version():
    """ Test to ensure Python version is correct (3=<) """
    assert sys.version_info >= (3, 0), "Python version must be 3 or higher"

def test_imports():
    """ Test to ensure all required packages can be imported """
    try:
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import sklearn
        import statsmodels
        import scipy
        import streamlit
    except ImportError as e:
        assert False, f"Missing library: {e.name}"

def test_spacy_model_lang():
    """ Ensure that the spacy language model is correctly set to English """
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("This is a test sentence.")
        assert doc.text == "This is a test sentence."
        assert len(doc) > 0  # Check that tokens are created
    except OSError:
        assert False, "SpaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm"