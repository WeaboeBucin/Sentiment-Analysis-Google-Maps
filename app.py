import streamlit as st
import pandas as pd
import re
import nltk
import pickle
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

# Download stopwords bahasa Indonesia
nltk.download('stopwords')
nltk.download('punkt')

import en_core_web_sm
nlp = en_core_web_sm.load()

# Buat stemmer untuk bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Ambil stopwords bahasa Indonesia
stop_words = set(stopwords.words('indonesian'))

with open('model.pkl', 'rb') as file:
    decisionTree = pickle.load(file)

with open('tfidf.pkl', 'rb') as file:
    tfidf = pickle.load(file)


# Fungsi utama Streamlit
def preprocess(text):
    # Mengubah teks menjadi lowercase
    text = str(text).lower()

    # Menghapus karakter non-alfanumerik
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Tokenisasi teks
    tokens = word_tokenize(text)

    # Menghapus stop words (kata-kata yang umum dan tidak informatif)
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Stemming kata-kata (mengubah kata-kata menjadi bentuk dasarnya)
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    # Menggabungkan kembali token-token menjadi teks
    preprocessed_text = ' '.join(stemmed_tokens)


    return preprocessed_text


def main():
    st.title("Klasifikasi Sentimen Dengan Decision Tree")

    # Sidebar untuk pengguna memasukkan data
    st.sidebar.header("Masukkan Data")
    input_data = []
    st.sidebar.markdown("sentimen")
    input_datasentimen = st.sidebar.text_input(f"Masukkan isi sentimen", key = "sentimen")


    input_df = pd.DataFrame({"sentimen": [input_datasentimen]})
    st.subheader("Data yang Dimasukkan")
    st.write(input_df)

    # Melatih model jika tombol ditekan
    if st.sidebar.button("Classification"):

        # Latih model
        data1 = preprocess(input_df['sentimen'])
        data1 = [data1]
        data_normal = tfidf.transform(data1)

        # # Lakukan prediksi
        prediction = decisionTree.predict(data_normal)
        st.subheader("Hasil Prediksi")
        if(prediction == 0):
            prediction = 'negatif'
        elif(prediction == 1):
            prediction = 'netral'
        elif(prediction == 2):
            prediction = 'positif'
        st.write("Prediksi Kelas:", prediction)
    st.subheader("Wordcloud Dataset :")
    # Menampilkan word cloud
    images = ['negatif.png', 'netral.png', 'positif.png']
    st.image(images, use_column_width=True, caption=["negatif wordcloud", "netral wordcloud", "positif wordcloud"])
if __name__ == "__main__":
    main()