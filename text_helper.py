### Все для предобработки текста
import re
from pdfminer.high_level import extract_text as e_t
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from pathlib import Path
from transformers import BertTokenizer
from nltk.corpus import stopwords 
import joblib

def extract_text(path: Path) -> list:
    #assert    

    text = e_t(path)
    # remove punctuation
    text_no_signs = re.sub(r'[\.\?\!\,\:\;\"]', '', text.replace('\n', ' '))
    # remove uneeded spaces
    text_no_signs = re.sub(r' +', r' ', text_no_signs)
    # tokenize



def tokenize_pdf(path: Path, tokenizer: str) -> list:
    #tokenizer can be Bert or BPE(example)
    #test
    text = extract_text(path)
    if tokenizer is 'Bert':
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        tokenized = tokenizer(text)
        return tokenized
    elif tokenizer is 'TfIdf':
        stemmer = EnglishStemmer()
        stemmed = [stemmer.stem(word) for word in text.split(' ')]
        stemmed_unite = [' '.join(word) for word in stemmed]
        #TODO add TfIdf saved tokkenzer
        vectorizer = joblib.load("tokenizer.pkl")
        tokenized = vectorizer.transform(stemmed_unite)
        return tokenized
    #assert

        
    
