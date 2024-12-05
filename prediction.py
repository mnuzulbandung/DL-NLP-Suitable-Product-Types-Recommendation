import streamlit as st
import re
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords

def run():
    # Mengambil model yang disimpan
    loaded_model = tf.keras.models.load_model("model_2")

    # Fungsi untuk menghilangkan karakter yang tidak bermakna
    def f_menghilangkan_karakter_tidak_bermakna(text):
        # Mengkecilkan huruf
        text = text.lower()

        # Menghilangkan karakter spesial dan angka
        text = re.sub(r'[^A-Za-z\s]', '', text)

        # Menghilangkan baris ganda
        text = re.sub(r'\\n', ' ',text)

        # Menghilangkan spasi ganda
        text = text.strip()

        # Menghilangkan link website
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"www.\S+", " ", text)

        return text
    
    # Mengunduh vocabulary stopwords dari nltk berbahasa inggris
    nltk.download('stopwords')
    stpwds_en = list(set(stopwords.words('english')))

        # Menghilangkan kata yang tidak bermakna
    def f_menghilangkan_kata_tidak_bermakna(text):
        # Mengubah teks menjadi list berdasarkan spasi
        tokens = re.findall(r'\w+|[^\w\s]', text)

        # Menghilangkan kata stopwords
        tokens = [word for word in tokens if word not in stpwds_en]

        # Menggabungkan kata pada list menjadi teks
        text = ' '.join(tokens)

        return text


    st.title('Prediksi Tipe Produk')

    st.write('''Laman ini dapat digunakan untuk memprediksi tipe produk yang cocok untuk digunakan user berdasarkan deskripsi produk yang user masukan untuk menjual produk user di dalam marketplace.''')



    with st.form(key='form_parameters'):
        input_text = st.text_input('Masukkan Deskripsi Produk:', placeholder="Contoh: Ini adalah teks untuk prediksi.")
        submit = st.form_submit_button('Predict')


    data = pd.DataFrame([{
        'descriptions': input_text
        }])

    st.write('Deskripsi produk yang dimasukkan:')
    st.dataframe(data)

    # Menghilangkan karakter yang tidak bermakna
    df_temp = data['descriptions'].apply(lambda x: f_menghilangkan_karakter_tidak_bermakna(x))
    # Menghilangkan kata yang tidak bermakna dengan stopwords
    inference_data_pre_processed = df_temp.apply(lambda x: f_menghilangkan_kata_tidak_bermakna(x))

    if submit:
        # Memprediksi data inference
        predictions = loaded_model.predict(inference_data_pre_processed)

        # Mencari nilai dengan hasil prediksi tertinggi
        vector_predicted = np.argmax(predictions, axis=1)

        # Vector mapping
        mapping_dict = {0: 'Household', 1: 'Books', 2: 'Clothing & Accessories', 3: 'Electronics'}

        # Mengubah vektor prediksi menjadi tipe produk
        tipe_produk_predicted = np.vectorize(mapping_dict.get)(vector_predicted)

        st.write('Tipe Produk Hasil Prediksi:')
        st.write(tipe_produk_predicted)


if __name__ == '__main__':
    run()