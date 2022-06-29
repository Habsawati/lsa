#!/usr/bin/env python
# coding: utf-8

# ## CRAWLING DATA BERITA

# Web crawling adalah teknik pengumpulan data yang digunakan untuk mengindeks informasi pada halaman menggunakan URL (Uniform Resource Locator) dengan menyertakan API (Application Programming Interface) untuk melakukan penambangan dataset yang lebih besar. Web crawling adalah teknik pengumpulan data yang digunakan untuk mengindeks informasi pada halaman menggunakan URL (Uniform Resource Locator) dengan menyertakan API (Application Programming Interface) untuk melakukan penambangan dataset yang lebih besar. 
# 
# Pada tugas ini untuk melakukan crawling saya menggunakan library scrapy. Scrapy adalah framework Python untuk melakukan web scraping dalam skala besar. Scrapy menyediakan segala tools yang kita butuhkan untuk mengekstrak data dari setiap website secara efisien, memprosesnya, lalu menyimpannya dalam struktur atau format yang kita inginkan.

# In[1]:


import scrapy
class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            'https://nasional.sindonews.com/',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # print(response.url)

        for i in range(1, 30):
            for berita in response.css('body > div:nth-child(6) > section > div.grid_24 > div.homelist-new.scroll'):
                yield{
                    'Topik': berita.css(' ul > li.latest-event.latest-track-' + str(i) + ' > div.homelist-box > div.homelist-top > div.homelist-channel::text').extract(),
                    'Tanggal': berita.css(' ul > li.latest-event.latest-track-' + str(i) + ' > div.homelist-box > div.homelist-top > div.homelist-date::text').extract(),
                    'Judul': berita.css(' ul > li.latest-event.latest-track-' + str(i) + ' > div.homelist-box > div.homelist-title > a::text').extract(),
                    # 'link': response.css('body > div:nth-child(6) > section > div.grid_24 > div.homelist-new.scroll > ul > li.latest-event.latest-track-0 > div.homelist-box > div.homelist-title > a::@href').extract(),
                    'gambar': berita.css('ul > li.latest-event.latest-track-' + str(i) + ' > div.homelist-pict > a > img::text').extract(),
                    'Deskripsi': berita.css(' ul > li.latest-event.latest-track-' + str(i) + ' > div.homelist-box > div.homelist-desc::text').extract(),

                }


# # Import Library yang Diperlukan

# In[2]:


# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

# for named entity recognition (NER)
from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#stop-words
stop_words=set(nltk.corpus.stopwords.words('indonesian'))


# # Load Dataset

# Memanggil data berita yang telah kita crawling tadi dengan file CSV

# In[26]:


df=pd.read_csv('scrapy_berita.csv')


# In[27]:


df.head()


# In[28]:


# Menghapus kolom yang tidak digunakan.
df.drop(['Topik', 'Tanggal', 'Judul'],axis=1,inplace=True)


# In[29]:


df.head(10)


# # DATA CLEANING & PRE-PROCESSING

# Text preprocessing adalah suatu proses untuk menyeleksi data text agar menjadi lebih terstruktur lagi dengan melalui serangkaian tahapan. Tapi, sesungguhnya tidak ada aturan pasti tentang setiap tahapan dalam text preprocessing. Semua itu tergantung dengan jenis serta kondisi data yang kita miliki. Text preprocessing merupakan salah satu implementasi dari text mining. Text mining sendiri adalah suatu kegiatan menambang data, dimana data yang biasanya diambil berupa text yang bersumber dari dokumen-dokumen yang memiliki goals untuk mencari kata kunci yang mewakili dari sekumpulan dokumen tersebut sehingga nantinya dapat dilakukan analisa hubungan antara dokumen-dokumen tersebut. 

# ## Removing Number

# Pada tahap ini akan melakukan penghapusan angka, untuk codenya dapat dilihat dibawah ini :

# In[30]:


import string 
import re #regex library

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

df['remov number'] = df['isi'].apply(remove_number)
df.head(10)


# ## Removing Punctuation

# Pada tahap prepocessing selanjutnya akan melakukan removing punctuation seperti menghapus simbol dan tanda baca yang tidak penting.

# In[31]:


#remove punctuation(simbol dan tanda baca)
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

df['remov punct'] = df['remov number'].apply(remove_punctuation)
df.head(10)


# ## Stopword

# Tahap selanjutnya untuk prepocessing adalah tahapan filtering yang digunakan untuk mengambil kata-kata yang penting dari hasil token tadi. Kata umum yang biasanya muncul dan tidak memiliki makna disebut dengan stopword. Misalnya penggunaan kata penghubung seperti dan, yang,serta, setelah, dan lainnya. Penghilangan stopword ini dapat mengurangi ukuran index dan waktu pemrosesan. Selain itu, juga dapat mengurangi level noise. 

# In[32]:


def clean_text(headline):
  le=WordNetLemmatizer()
  word_tokens=word_tokenize(headline)
  tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
  cleaned_text=" ".join(tokens)
  return cleaned_text


# In[33]:


# time taking
df['stopword']=df['remov punct'].apply(clean_text)


# In[34]:


df.head(10)


# In[35]:


#Menghapus kolom isi, remov number dan remov punct
df.drop(['isi', 'remov number', 'remov punct'],axis=1,inplace=True)
df.head(10)


# In[36]:


# Melihat data ke-0
df['stopword'][0]


# ## MENGEKSTRAK FITUR DAN MEMBUAT DOCUMENT-TERM-MATRIX ( DTM )
# Dalam DTM nilainya adalah nilai TFidf. Term Frequency — Inverse Document Frequency atau TFIDF adalah suatu metode algoritma yang berguna untuk menghitung bobot setiap kata yang umum digunakan. Metode ini juga terkenal efisien, mudah dan memiliki hasil yang akurat. Secara sederhana, metode TF-IDF digunakan untuk mengetahui berapa sering suatu kata muncul di dalam dokumen. Contoh yang dibahas kali ini adalah mengenai penentuan urutan peringkat data berdasarkan query yang digunakan.
# 
# Inti utama dari algoritma ini adalah melakukan perhitungan nilai TF dan nilai IDF dari setiap kata kunci terhadap masing-masing dokumen dalam korpus. 
# 
# Term Frequency (TF) yaitu pembobotan/weight setiap kata (term) pada suatu dokumen berdasarkan jumlah kemunculannya dalam dokumen tersebut. Semakin besar jumlah kemunculan suatu kata dalam dokumen, maka semakin besar pula bobot yang diberikan (TF Tinggi) jadi nilai tertinggi merupakan  jumlah kemunculan/frekuensi.
# 
# Setelah menentukan Tf maka selanjutnya kita tentukan nilai IDF nya dapat dihitung dengan rumus :
# $$
# \operatorname{idf}=\log \left(\frac{D}{d f}\right)
# $$
# 
# Selanjutnya adalah melakukan perkalian antara nilai TF dan IDF untuk mendapatkan jawaban akhir. untuk rumusnya sebagai berikut:
# $$
# \begin{gathered}
# Tf-Idf=t f_{i j} * i d f_{j} \\
# Tf-Idf=t f_{i j} * \log \left(\frac{D}{d f}\right)
# \end{gathered}
# $$
# 
# Keterangan :
# Dimana D adalah jumlah semua dokumen dalam koleksi sedangkan df adalah jumlah dokumen yang mengandung term tertentu.
# 
# Parameter dari vectorizer Tfidf mmiliki beberapa poin penting:
# 1) LSA umumnya diimplementasikan dengan nilai Tfidf di mana-mana dan tidak dengan Count Vectorizer.
# 
# 2) max_features tergantung pada daya komputasi Anda dan juga pada eval. metrik (skor koherensi adalah metrik untuk model topik). Coba nilai yang memberikan evaluasi terbaik. metrik dan tidak membatasi kekuatan pemrosesan.
# 
# 3) Nilai default untuk min_df & max_df bekerja dengan baik.
# 
# 4) Dapat mencoba nilai yang berbeda untuk ngram_range.
# 

# In[37]:


vect =TfidfVectorizer(stop_words=stop_words,max_features=1000)

# to play with. min_df,max_df,max_features etc...


# In[38]:


vect_text=vect.fit_transform(df['stopword'])


# In[39]:


print(vect_text.shape)
print(vect_text)


# Kita sekarang dapat melihat kata-kata yang paling sering dan langka di berita utama berdasarkan skor idf. Semakin kecil nilainya maka kata-kata dalam berita lebih sering muncul.

# In[40]:


vect.get_feature_names()


# In[41]:


vect_text.todense()


# In[42]:


df = pd.DataFrame(vect_text.todense().T, index=vect.get_feature_names(), columns=[f'{i+1}' for i in range (len(df))])
df


# ## LSA
# Latent Semantic Analysis (LSA) merupakan sebuah metode yang memanfaatkan model statistik matematis untuk menganalisa struktur semantik suatu teks. LSA bisa digunakan untuk menilai esai dengan mengkonversikan esai menjadi matriks-matriks yang diberi nilai pada masing-masing term untuk dicari kesamaan dengan term referensi. LSA pada dasarnya adalah dekomposisi nilai tunggal.
# 
# Singular Value Decomposition (SVD) adalah salah satu teknik reduksi dimensi yang bermanfaat untuk memperkecil nilai kompleksitas dalam pemrosesan term-document matrix. SVD merupakan teorema aljabar linier yang menyebutkan bahwa persegi panjang dari term-document matrix dapat dipecah/didekomposisikan menjadi tiga matriks, yaitu :
# 
# – Matriks ortogonal U (matriks dokumen-topik)
# 
# – Matriks diagonal S (Matrik diagonal dengan elemen matriks positif atau nol)
# 
# – Transpose dari matriks ortogonal V (matriks topik-term)
# 
# Yang dirumuskan dengan :
# $$
# A_{m n}=U_{m m} x S_{m n} x V_{n n}^{T}
# $$
# 
# Keterangan : 
# A = Matriks Masukan (Pada Penelitian matriks ini berisi matrik hasil perhitungan TF-IDF)
# 
# U = Matriks Ortogonal U
# 
# S = Matriks Diagonal S (matriks positif atau nol)
# 
# V =  Transpose Ortogonal V
# 
# 
# Setiap baris dari matriks U (matriks istilah dokumen) adalah representasi vektor dari dokumen yang sesuai. Panjang vektor ini adalah jumlah topik yang diinginkan. Representasi vektor untuk suku-suku dalam data kami dapat ditemukan dalam matriks V (matriks istilah-topik).
# 
# Jadi, SVD memberi kita vektor untuk setiap dokumen dan istilah dalam data kita. Panjang setiap vektor adalah k. Kami kemudian dapat menggunakan vektor-vektor ini untuk menemukan kata-kata dan dokumen serupa menggunakan metode kesamaan kosinus.
# 
# Kita dapat menggunakan fungsi truncatedSVD untuk mengimplementasikan LSA. Parameter n_components adalah jumlah topik yang ingin kita ekstrak. Model tersebut kemudian di fit dan ditransformasikan pada hasil yang diberikan oleh vectorizer.
# 
# Terakhir perhatikan bahwa LSA dan LSI (I untuk pengindeksan) adalah sama dan yang terakhir kadang-kadang digunakan dalam konteks pencarian informasi.

# In[43]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[44]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[45]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)


# In[46]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# Sekarang bisa mendapatkan daftar kata-kata penting untuk masing-masing dari 12 topik seperti yang ditunjukkan. Untuk kesederhanaan di sini saya telah menunjukkan 10 kata untuk setiap topik

# In[48]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:12]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# In[ ]:




