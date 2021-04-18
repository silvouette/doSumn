import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

document1 ="""Penyakit kardiovaskuler atau cardiovascular disease ( CVD) menurut definisi WHO adalah istilah bagi serangkaian gangguan jantung dan pembuluh darah.  Data badan kesehatan dunia WHO ( 2012) menunjukan bahwa CVD adalah faktor penyebab kematian nomor satu didunia dan berdasarkan data Riset Kesehatan Dasar ( RISKESDAS)  Kementerian Kesehatan Republik Indonesia tahun 2007 menunjukkan, penyakit yang termasuk kelompok CVD menempati urutan teratas penyebab kematian di Indonesia. Ditinjau dari sisi ketersediaan tenaga ahli dibidang cardiovascular, saat ini Indonesia hanya memiliki sekitar 500 dokter spesialis penyakit jantung dan pembuluh darah. Artinya dengan jumlah penduduk Indonesia yang mencapai 240 juta, rasio dokter spesialis jantung dan pembuluh darah adalah 1:480.000 penduduk. Jumlah ini masih sangat kurang dibandingkan dengan kebutuhan penduduk di Indonesia.  Berdasarkan data RISKESDAS tahun 2007, penyakit yang termasuk kelompok CVD menempati urutan teratas penyebab kematian di Indonesia yaitu sebanyak 31,9%. Kualitas dan peningkatan akses pelayanan penyakit CVD sangat bergantung pada ketersediaan dan distribusi dokter spesialis. Sistem menerima input berbetuk biner ( 1 dan 0) diamana, nilai 1 menunjukan adanya gejala atau faktor resiko dan nilai 0 menunjukan tidak ada gejala atau faktor resiko dalam sebuah kasus. Pengujian tanpa threshold menunjukan tingkat akurasi sebesar 89%."""
parser = PlaintextParser.from_string(document1,Tokenizer("english"))

summarizer = LexRankSummarizer()
#Summarize the document with 2 sentences
summary = summarizer(parser.document, 4)

print("\n")
for sentence in summary:
    print(sentence)