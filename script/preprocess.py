import string
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

def clean(text):
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    noise = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'tabel', 'gambar', 'yang']
    text = text.translate(str.maketrans("","",string.punctuation)).strip().lower()
    text = re.sub(r'\w*\d+\w*', '', text)
    text = ' '.join(w for w in text.split() if w not in noise)
    return text