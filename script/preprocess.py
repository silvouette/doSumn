import string
import re

def clean(text):
    noise = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'tabel', 'gambar', 'yang', 'dan', 'atau']
    text = text.translate(str.maketrans("","",string.punctuation)).strip().lower()
    text = re.sub(r'\w*\d+\w*', '', text)
    text = ' '.join(w for w in text.split() if w not in noise)
    return text