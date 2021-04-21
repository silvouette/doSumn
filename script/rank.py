from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import difflib
 
def ranker(data):
    sets = ['BACKGROUND','TOPIC','METHOD','DATASET','RESULT','CONCLUSION','SUGGESTION']
    summ_collection = {}

    for cat in sets:
        mask = data.loc[data['pred']== cat]
        if len(mask)>=10:
            summ = generate_summary(data.loc[data['pred']== cat], int(len(mask)*0.4))
        elif len(mask)>=2:
            summ = generate_summary(data.loc[data['pred']== cat], int(len(mask)*0.5))
        else:
            summ = data.loc[data['pred']== cat,'sentence'].values

        summ_collection[cat] = summ
        
    return summ_collection

def read_sent(sents):
    sentences = [sentence.replace("[^a-zA-Z]", " ").split(" ") for sentence in sents]
    # sentences.pop()
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def generate_summary(data, top_n):
    stop_words = stopwords.words('indonesian')

    # step 1 - read and split text
    sentencex_col =  data.loc[:,'sentence_x']
    sentences = read_sent(sentencex_col.values)

    sentence_col =  data.loc[:,'sentence']
    real_sentences_pre = sentence_col.values 
    real_sentences =  read_sent(sentence_col.values) #for comparison later

    # step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    # step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    # step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(real_sentences)), reverse=True)   #indexes of top ranked sentences  

    summarize_text = [" ".join(ranked_sentence[i][1]) for i in range(top_n)]

    check = summarize_text
    rid = []
    for index, sent in enumerate(check[:-1]):
        slicer = index+1
        sets = check[slicer:]
        diff = difflib.get_close_matches(sent, sets)

        if len(diff)>0:
            rid = ",".join(diff)

    chosen = [x for x in real_sentences_pre if x in summarize_text and x not in rid]
    return chosen