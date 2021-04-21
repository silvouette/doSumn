import pandas as pd
import string
import re
pd.options.mode.chained_assignment = None

def get_parser_data(filename):
    data = pd.read_csv('../' + filename)
    return data

def wordcount(text):
    word_count = 0
    for key, value in text.items():
      for i in value:
        word_count += len(i.split())
    return word_count

def clean(text):
    noise = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'tabel', 'gambar', 'yang', 'dan', 'atau']
    text = text.translate(str.maketrans("","",string.punctuation)).strip().lower()
    text = re.sub(r'\w*\d+\w*', '', text)
    text = ' '.join(w for w in text.split() if w not in noise)
    return text