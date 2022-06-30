#!/usr/bin/env python
# coding: utf-8

# ## CRAWLING DATA PTA MANAJEMEN

# Web crawling adalah teknik pengumpulan data yang digunakan untuk mengindeks informasi pada halaman menggunakan URL (Uniform Resource Locator) dengan menyertakan API (Application Programming Interface) untuk melakukan penambangan dataset yang lebih besar. Web crawling adalah teknik pengumpulan data yang digunakan untuk mengindeks informasi pada halaman menggunakan URL (Uniform Resource Locator) dengan menyertakan API (Application Programming Interface) untuk melakukan penambangan dataset yang lebih besar. 
# 
# Pada tugas ini untuk melakukan crawling saya menggunakan library scrapy. Scrapy adalah framework Python untuk melakukan web scraping dalam skala besar. Scrapy menyediakan segala tools yang kita butuhkan untuk mengekstrak data dari setiap website secara efisien, memprosesnya, lalu menyimpannya dalam struktur atau format yang kita inginkan.

# import scrapy
# class QuotesSpider(scrapy.Spider):
#     name = "quotes"
# 
#     def start_requests(self):
#         urls = [
#             'https://pta.trunojoyo.ac.id/welcome/detail/080211100070',
#             'https://pta.trunojoyo.ac.id/welcome/detail/090211200001',
#             'https://pta.trunojoyo.ac.id/welcome/detail/080211100050',
#             'https://pta.trunojoyo.ac.id/welcome/detail/100211200002',
#             'https://pta.trunojoyo.ac.id/welcome/detail/080211100044',
#             'https://pta.trunojoyo.ac.id/welcome/detail/080211100119',
#             'https://pta.trunojoyo.ac.id/welcome/detail/080211100103',
#             'https://pta.trunojoyo.ac.id/welcome/detail/080211100098',
#             'https://pta.trunojoyo.ac.id/welcome/detail/090211100079',
#             'https://pta.trunojoyo.ac.id/welcome/detail/090211100089',
#             'https://pta.trunojoyo.ac.id/welcome/detail/090211100013',
#             'https://pta.trunojoyo.ac.id/welcome/detail/090211100020',
#             'https://pta.trunojoyo.ac.id/welcome/detail/090211100064',
#             'https://pta.trunojoyo.ac.id/welcome/detail/090211100064',
#             'https://pta.trunojoyo.ac.id/welcome/detail/090211100018'
#         ]
#         for url in urls:
#             yield scrapy.Request(url=url, callback=self.parse)
# 
#     def parse(self, response):
#         # print(response.url)
#         yield {
#             'judul': response.css('#content_journal > ul > li > div:nth-child(2) > a::text').extract(),
#             'penulis': response.css('#content_journal > ul > li > div:nth-child(2) > div:nth-child(2) > span::text').extract(),
#             'dosen_pembimbing_1': response.css('#content_journal > ul > li > div:nth-child(2) > div:nth-child(3) > span::text').extract(),
#             'dosen_pembimbing_2': response.css('#content_journal > ul > li > div:nth-child(2) > div:nth-child(4) > span::text').extract(),
#             'abstrak': response.css('#content_journal > ul > li > div:nth-child(4) > div:nth-child(2) > p::text').extract(),
#         }
#         # content_journal > ul > li:nth-child(1) > div:nth-child(1) > a
#         # content_journal > ul > li:nth-child(1) > div:nth-child(1) > a

# # Import Library yang Diperlukan

# In[1]:


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

# In[21]:


df=pd.read_csv('scrapylsa.csv')


# In[22]:


df.head()


# In[23]:


# Menghapus kolom yang tidak digunakan.
df.drop(['judul', 'penulis', 'dosen_pembimbing_1', 'dosen_pembimbing_2'],axis=1,inplace=True)


# In[24]:


df.head(10)


# # DATA CLEANING & PRE-PROCESSING

# Text preprocessing adalah suatu proses untuk menyeleksi data text agar menjadi lebih terstruktur lagi dengan melalui serangkaian tahapan. Tapi, sesungguhnya tidak ada aturan pasti tentang setiap tahapan dalam text preprocessing. Semua itu tergantung dengan jenis serta kondisi data yang kita miliki. Text preprocessing merupakan salah satu implementasi dari text mining. Text mining sendiri adalah suatu kegiatan menambang data, dimana data yang biasanya diambil berupa text yang bersumber dari dokumen-dokumen yang memiliki goals untuk mencari kata kunci yang mewakili dari sekumpulan dokumen tersebut sehingga nantinya dapat dilakukan analisa hubungan antara dokumen-dokumen tersebut. 

# ## Removing Number

# Pada tahap ini akan melakukan penghapusan angka, untuk codenya dapat dilihat dibawah ini :

# In[25]:


import string 
import re #regex library

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

df['remov number'] = df['abstrak'].apply(remove_number)
df.head(10)


# ## Removing Punctuation

# In[26]:


def remove_all(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
df['remov all'] = df['remov number'].apply(remove_all)
df.head(10)


# Pada tahap prepocessing selanjutnya akan melakukan removing punctuation seperti menghapus simbol dan tanda baca yang tidak penting.

# In[27]:


#remove punctuation(simbol dan tanda baca)
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

df['remov punct'] = df['remov all'].apply(remove_punctuation)
df.head(10)


# ## Stopword

# Tahap selanjutnya untuk prepocessing adalah tahapan filtering yang digunakan untuk mengambil kata-kata yang penting dari hasil token tadi. Kata umum yang biasanya muncul dan tidak memiliki makna disebut dengan stopword. Misalnya penggunaan kata penghubung seperti dan, yang,serta, setelah, dan lainnya. Penghilangan stopword ini dapat mengurangi ukuran index dan waktu pemrosesan. Selain itu, juga dapat mengurangi level noise. 

# In[28]:


def clean_text(headline):
  le=WordNetLemmatizer()
  word_tokens=word_tokenize(headline)
  tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
  cleaned_text=" ".join(tokens)
  return cleaned_text


# In[29]:


# time taking
df['stopword']=df['remov punct'].apply(clean_text)


# In[30]:


df.head(10)


# In[31]:


#Menghapus kolom isi, remov number dan remov punct
df.drop(['abstrak', 'remov number', 'remov all', 'remov punct'],axis=1,inplace=True)
df.head(10)


# In[32]:


#mengubah huruf menjadi kecil dengan menggunakan Series.str.lower() pada pandas

df['stopword'] = df['stopword'].str.lower()

df.head(10)


# In[33]:


# Melihat data ke-0
df['stopword'][0]


# In[35]:


df.to_csv("abstrak_prepocessing.csv")


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

# In[38]:


import pandas as pd 
import numpy as np

df = pd.read_csv("abstrak_prepocessing.csv", usecols=["stopword"])
df.columns = ["abstrak_akhir"]

df.head()


# In[40]:


from sklearn.feature_extraction.text import CountVectorizer

document = df['abstrak_akhir']
a=len(document)

# Create a Vectorizer Object
vectorizer = CountVectorizer()

vectorizer.fit(document)

# Printing the identified Unique words along with their indices
print("Vocabulary: ", vectorizer.vocabulary_)

# Encode the Document
vector = vectorizer.transform(document)

# Summarizing the Encoded Texts
print("Encoded Document is:")
print(vector.toarray())


# In[41]:


a = vectorizer.get_feature_names()


# # TF IDF

# In[43]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
tf = tfidf.fit_transform(vectorizer.fit_transform(document)).toarray()


# In[44]:


dfb = pd.DataFrame(data=tf,index=list(range(1, len(tf[:,1])+1, )),columns=[a])
dfb


# In[ ]:





# # K-Means

# In[56]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[58]:


#--- Mengubah Variabel Data Frame Menjadi Array ---
x_array =  np.array(dfb)
print(x_array)


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




