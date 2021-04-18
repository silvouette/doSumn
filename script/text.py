from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
 
def read_article():
    article = ['Sedangkan dengan menggunakan variasi threshold 0,5 , 0,75 dan 0,95 diperoleh hasil masing - masing 100%, 12,67% dan 0,67%.', 'Proses diagnosa kasus baru dilakukan dengan cara memasukkan data pasien, faktor resiko dan gejala - gejala yang dialami pasien pada kasus baru.', 'Kesamaan masing - masing faktor resiko dan gejala akan dihitung menggunakan persamaan ( 1).', 'Frame berisi relasi antara data pasien, penyakit yang diderita, faktor resiko dan gejala - gejala yang menyertai kasus tersebut .', 'Setiap kasus diberikan tingkat kepercayaan / keyakinan dari pakar terhadap hubungan data - data tersebut , sehingga dengan representasi ini dapat dibuat suatu model kasus untuk sistem CBR, dimana problem space adalah faktor resiko dan gejala - gejala penyakit serta solution space adalah nama penyakit.', 'Tingkat kepercayaan menunjukan kepastian diagnosa dari pakar berdasarkan faktor resiko dan gejala yang dialami pasien.', 'Pada penelitian ini, metode similaritas yang digunakan adalah simple matching coefficient dengan persamaan ( 1) ( Tursina, 2012).', 'Pakar akan merevisi nama penyakit beserta tingkat kepercayaan terhadap penyakit hasil diagnosa memiliki nilai similarity lebih kecil dari 0.8.Setelah kasus direvisi, selanjutnya kasus tersebut akan disimpan ( retain) dan dijadikan sebagai basis kasus baru.', 'Sistem dibagi menjadi 2 kategori berdasarkan jenis pemakai yaitu pakar dan paramedis.', 'Masing - masing kategori pemakai mempunyai hak akses terhadap sistem yang dengan fasilitas yang berbeda - beda.', 'Berdasarkan hasil pengujian sistem case - based reasoning untuk mendiagnosa penyakit kardiovaskuler dapat ditarik beberapa kesimpulan sebagai berikut: 1. Sistem case - based reasoning dengan menggunakan metode simple matching coefficient dapat diimplementasikan untuk melakukan diagnosa awal penyakit cardiovascular berdasarkan kondisi ( gejala dan faktor resiko) seorang pasien.', 'Nilai PPV 86,84% dan NPV 90,00%, dengan tingkat akurasi sebesar 87,50% serta tingkat kesalahan ( error rate) sebesar 12,50%.']
    sentences = []

    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    
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


def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article()
    top_n = int(len(sentences)*0.5)
    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    print(top_n)
    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    article = ['Sedangkan dengan menggunakan variasi threshold 0,5 , 0,75 dan 0,95 diperoleh hasil masing - masing 100%, 12,67% dan 0,67%.', 'Proses diagnosa kasus baru dilakukan dengan cara memasukkan data pasien, faktor resiko dan gejala - gejala yang dialami pasien pada kasus baru.', 'Kesamaan masing - masing faktor resiko dan gejala akan dihitung menggunakan persamaan ( 1).', 'Frame berisi relasi antara data pasien, penyakit yang diderita, faktor resiko dan gejala - gejala yang menyertai kasus tersebut .', 'Setiap kasus diberikan tingkat kepercayaan / keyakinan dari pakar terhadap hubungan data - data tersebut , sehingga dengan representasi ini dapat dibuat suatu model kasus untuk sistem CBR, dimana problem space adalah faktor resiko dan gejala - gejala penyakit serta solution space adalah nama penyakit.', 'Tingkat kepercayaan menunjukan kepastian diagnosa dari pakar berdasarkan faktor resiko dan gejala yang dialami pasien.', 'Pada penelitian ini, metode similaritas yang digunakan adalah simple matching coefficient dengan persamaan ( 1) ( Tursina, 2012).', 'Pakar akan merevisi nama penyakit beserta tingkat kepercayaan terhadap penyakit hasil diagnosa memiliki nilai similarity lebih kecil dari 0.8.Setelah kasus direvisi, selanjutnya kasus tersebut akan disimpan ( retain) dan dijadikan sebagai basis kasus baru.', 'Sistem dibagi menjadi 2 kategori berdasarkan jenis pemakai yaitu pakar dan paramedis.', 'Masing - masing kategori pemakai mempunyai hak akses terhadap sistem yang dengan fasilitas yang berbeda - beda.', 'Berdasarkan hasil pengujian sistem case - based reasoning untuk mendiagnosa penyakit kardiovaskuler dapat ditarik beberapa kesimpulan sebagai berikut: 1. Sistem case - based reasoning dengan menggunakan metode simple matching coefficient dapat diimplementasikan untuk melakukan diagnosa awal penyakit cardiovascular berdasarkan kondisi ( gejala dan faktor resiko) seorang pasien.', 'Nilai PPV 86,84% dan NPV 90,00%, dengan tingkat akurasi sebesar 87,50% serta tingkat kesalahan ( error rate) sebesar 12,50%.']

    for i in article:
        if i in summarize_text:
            print(i)
    # Step 5 - Offcourse, output the summarize texr
    # print("Summarize Text: \n", "".join(summarize_text))

# let's begin

generate_summary( "msft.txt", 2)