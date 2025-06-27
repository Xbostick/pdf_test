### Все для предобработки текста
import re
from pdfminer.high_level import extract_text as e_t
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import RussianStemmer
from pathlib import Path
from transformers import BertTokenizer
from nltk.corpus import stopwords 

def extract_text(path: Path) -> list:
    #assert    

    text = e_t(path)
    # remove punctuation
    text_no_signs = re.sub(r'[\.\?\!\,\:\;\"]', '', text)
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
    elif tokenizer is 'BPE':
        tokenized = word_tokenize(text) #pomin`iat`
        stop_words = set(stopwords.words('russian')) 
        # remove stopwords from tokens in dataset
        statement_no_stop = [word for word in tokenized if word not in stop_words]

        stemmer = RussianStemmer()
        stemmed = [stemmer.stem(token) for token in tokenized]

        #BPE impl

        return stemmed
    #assert

        
    
