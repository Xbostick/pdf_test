"""
Utilities for text preprocessing, including cleaning, tokenization, and feature extraction.
"""

import re
from pdfminer.high_level import extract_text 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from transformers import BertTokenizer
from nltk.corpus import stopwords 
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import string


# Regular expressions for cleaning text
RE_URL = re.compile(r'(?:http|ftp|https)://[\w_-]+(?:\.[\w_-]+)+[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-]?')
RE_EMAIL = re.compile(r'[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*@'
                      r'(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|'
                      r'\[(?:(?:(2(?:5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}'
                      r'(?:(2(?:5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:'
                      r'(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])')

def clean_header(text):
    """
    Remove email headers from text.
    
    Args:
        text: Input text string.
    
    Returns:
        Text with headers removed.
    """  
    assert isinstance(text, str), "Input must be a string"

    text = re.sub(r'(From:\s+[^\n]+\n)', '', text)
    text = re.sub(r'(Subject:[^\n]+\n)', '', text)
    text = re.sub(r'(([\sA-Za-z0-9\-]+)?[A|a]rchive-name:[^\n]+\n)', '', text)
    text = re.sub(r'(Last-modified:[^\n]+\n)', '', text)
    text = re.sub(r'(Version:[^\n]+\n)', '', text)

    return text

def clean_text(text):  
    """
    Clean text by converting to lowercase, removing URLs, emails, punctuation, and numbers.
    
    Args:
        text: Input text string.
    
    Returns:
        Cleaned text string.
    """
    assert isinstance(text, str), "Input must be a string"

    text = text.lower()
    text = text.strip()
    text = re.sub(RE_URL, '', text)
    text = re.sub(RE_EMAIL, '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'(\d+)', ' ', text)
    text = re.sub(r'(\s+)', ' ', text)
    
    return text

def create_pca_transformer(X, n_components=200, save_path="./models", force_create=False):
    """
    Create and save a PCA transformer for dimensionality reduction.
    
    Args:
        X: Input data (TF-IDF vectors).
        n_components: Number of components to keep.
        save_path: Directory to save PCA model.
        force_create: If True, recreate PCA even if it exists.
    
    Returns:
        PCA transformer object.
    """
    assert n_components > 0, "Number of components must be positive"
    
    pca_path = Path(save_path) / "pca_transformer.pkl"
    if not force_create and pca_path.exists():
        print(f"PCA transformer already exists. Loaded from {pca_path}")
        return joblib.load(pca_path)
    
    pca = PCA(n_components=n_components)
    pca.fit(X.toarray()) 
    joblib.dump(pca, pca_path)
    print(f"PCA transformer saved to {pca_path}")
    return pca

def preprocess_corpus(corpus):
    """
    Preprocess a corpus by cleaning, tokenizing, removing stopwords, and lemmatizing.
    
    Args:
        corpus: List of text strings.
    
    Returns:
        List of preprocessed text strings.
    """
    assert isinstance(corpus, (list, tuple)), "Corpus must be a list or tuple"
    assert all(isinstance(t, str) for t in corpus), "All corpus elements must be strings"

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Clean headers and text
    cleaned_data = [clean_text(clean_header(line)) for line in corpus]
    
    # Tokenize
    tokenized_data = [word_tokenize(line) for line in cleaned_data]
    
    # Remove stopwords
    filtered_data = [[w for w in line if w not in stop_words] for line in tokenized_data]
    
    # Lemmatize
    lemmed_data = [' '.join(lemmatizer.lemmatize(token) for token in line) for line in filtered_data]
    
    return lemmed_data

def create_tokenizer(corpus, save_path = "./models", force_create = False):
    """
    Create and save a TF-IDF tokenizer.
    
    Args:
        corpus: List of text strings to fit the tokenizer.
        save_path: Directory to save the tokenizer (default: './models').
        force_create: If True, recreate tokenizer even if it exists (default: False).
    
    Returns:
        TF-IDF vectorizer object.
    """
    assert isinstance(corpus, (list, tuple)), "Corpus must be a list or tuple"

    path = Path(save_path) / "TfIdf_pretrained.pkl"
    if not force_create and Path(path).exists():
        print(f"Tokenizer already created. Loaded from {path}")
        return joblib.load(path)
    
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=1, use_idf=True)
    vectorizer.fit(corpus)
    joblib.dump(vectorizer, path)
    print(f"TfIdf tokenizer saved to {path}")
    return vectorizer

class Preprocessor():
    """Class to preprocess text data"""
    def __init__(self, corpus = None, save_path = "./models", force_create = False,
                force_create_TfIDF = False,
                force_create_pca = False, n_components_pca = 200):
        """
        Initialize the preprocessor.
        
        Args:
            corpus: List of text strings to fit the preprocessor (default: None).
            save_path: Directory to save the preprocessor (default: './models').
            force_create: If True, recreate preprocessor even if it exists (default: False).
            force_create_TfIDF: If True, recreate TF-IDF tokenizer (default: True).
            force_create_pca: If True, recreate PCA transformer (default: True).
            n_components_pca: Number of PCA components (default: 200).
        """
        
        path = Path(save_path) / "Preprocessor_pretrained.pkl"
        if not force_create and path.exists():
            print(f"Preprocessor exists. Load it from {path} using load_preprocessor")
            return None
        
        if corpus == None:
            print("No corpus provided. Preprocessor not created. try to load it with load_preprocessor function")
            return
        
        print(f"Start creating preprocessor. It would be saved to {path}\n")
        corpus = preprocess_corpus(corpus)
        print(f"Start creating TfIdf tokenizer.\n")
        self.TfIdf = create_tokenizer(corpus, save_path= save_path, force_create = force_create_TfIDF)
        print(f"Start creating pca transformer with {n_components_pca} components.\n")
        self.pca = create_pca_transformer(self.TfIdf.transform(corpus), force_create=force_create_pca,
                                          n_components= n_components_pca, save_path= save_path)
        joblib.dump(self, path)
    
    def preprocess(self, corpus):
        """
        Preprocess a corpus using TF-IDF and PCA.
        
        Args:
            corpus: Input text or list of texts.
        
        Returns:
            Transformed features.
        """
        if isinstance(corpus, str):
            corpus = [corpus]
        assert isinstance(corpus, (list, tuple)), "Corpus must be a list or tuple"
        return self.pca.transform(self.TfIdf.transform(preprocess_corpus(corpus)))
    
def load_preprocessor(save_path = "./models"):
    """
    Load a pretrained preprocessor.
    
    Args:
        save_path: Directory where the preprocessor is saved (default: './models').
    
    Returns:
        Preprocessor object.
    """
    path = Path(save_path) / "Preprocessor_pretrained.pkl"
    assert path.exists(), f"Preprocessor not found at {path}"
    return joblib.load(path)


def tokenize_text_inference(path: Path, preprocessor: Preprocessor) -> list:
    """
    Tokenize and preprocess text from a PDF file for inference.
    
    Args:
        path: Path to the PDF file.
        preprocessor: Preprocessor object with TfIdf and PCA transformers.
    
    Returns:
        Preprocessed text features.
    """
    assert path.exists(), f"PDF file not found at {path}"

    text = extract_text(path)
    return preprocessor.preprocess([text])