# Memanggil Module
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import wordcloud
from wordcloud import WordCloud



def run():

    # Memanggil data
    data = pd.read_csv(r'https://zenodo.org/record/3355823/files/ecommerceDataset.csv?download=1', header=None, names=['type', 'descriptions'], delimiter=',')

    
    # Bagian atas
    st.title('Exploratory Data Analysis - Product Description of PT. Tokohijau 2020 Januari - 2024 Januari')
    st.write('---')
    st.image('https://portalberita.stekom.ac.id/assets/images/berita/green-marketing.jpg?t=1730810693851')

    # Deskripsi awal
    st.title('Deskripsi')
    st.write('### Laman ini menjelaskan data deskripsi produk yang digunakan untuk melatih model yang dibuat untuk memberikan rekomendasi tipe produk dari produk yang akan dijual oleh user dalam marketplace.')
    st.markdown('Sumber data: click [here](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification/data) untuk mengunjungi website.')

    # Bagian A Isi dari sebagian data
    st.title('A. Isi dari sebagian data')
    st.dataframe(data.head(5))
    st.write('### Data memiliki jumlah baris sebanyak 50425 baris dan jumlah atribut sebanyak 2 atribut.')

    # Bagian B. Data Overview
    st.title('B. Data Overview:')
    st.write('### Data Dimensions:')
    st.write('* Jumlah Fitur: 2')
    st.write('* Jumlah Baris: 50,425')
    st.write('### Attribute Descriptions:')
    st.write('* **descriptions**: Deskripsi dari produk. (Categorical)')
    st.write('* **type**: Tipe dari produk, digunakan sebagai fitur target.  Berisi 4 kategori, Household; Books; Clothing & Accessories; dan Electronics. (Categorical)')

    # Bagian C. Proporsi data per kategori
    st.title('C. Word Cloud berdasarkan fitur descriptions:')
    st.header('Memperlihatkan beberapa kata yang sering muncul dalam fitur descriptions')
    # Menggabungkan seluruh kata dalam fitur descriptions menjadi satu variabel
    text_data = " ".join(data['descriptions'].astype(str))
    # Membuat wordcloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    # Memperlihatkan wordcloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    st.write('### Insights:')
    st.write('* Fitur descriptions didominasi oleh beberapa kata, diantaranya: made, use, book, make, used, one, product, dan feature.')
    st.write('* Terdapat beberapa kata yang tidak memberikan makna khusus seperti made, use, dan product. Artinya, kata-kata ini bisa saja terdapat pada semua tipe produk dan tidak memberikan pola unik pada model.')
    st.write('* Terdapat beberapa kata yang berbeda namun memiliki makna yang sama seperti make-made dan use-used. Kata-kata ini dapat meningkatkan redudansi yang memberatkan model')
    
if __name__ == '__main__':
    run()