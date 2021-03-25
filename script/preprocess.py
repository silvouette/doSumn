import string
import re

def clean(text):
    noise = ['i', 'ii', 'iii', 'iv', 'v', 'vi']
    text = text.translate(str.maketrans("","",string.punctuation)).strip().lower()
    text = re.sub(r"\d+", "", text)
    text = ' '.join(w for w in text.split() if w not in noise)
    return text