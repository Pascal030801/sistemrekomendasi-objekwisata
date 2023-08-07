
# Pertama lakukan import Flask, pandas, Tf idf Vectorizer, Cosine_similarity, os, nltk, dan stopwords
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os 
import nltk
from nltk.corpus import stopwords


# membuat aplikasi Flask
app = Flask(__name__)

# Membaca data dari file Excel
df = pd.read_excel('objek.xlsx')

# Download stopwords and initialize stemmer
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# Preprocess text by removing stop words
def preprocess_text(text):
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered_tokens)


# Apply preprocessing to the 'deskripsi' column
df['deskripsi_preprocessing'] = df['deskripsi'].apply(preprocess_text)

# Inisialisasi TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Melakukan fit_transform pada deskripsi destinasi
tfidf_matrix = vectorizer.fit_transform(df['deskripsi_preprocessing'])

# Mendapatkan cosine similarity matrix dari tfidf_matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Fungsi untuk mendapatkan rekomendasi berdasarkan kata kunci
def get_recommendations(keyword, cosine_sim):

    # Mengubah keyword menjadi tf-idf vector
    keyword_vector = vectorizer.transform([keyword])

    # Menghitung similarity antara keyword vector dan tf-idf matrix
    sim_scores = cosine_similarity(keyword_vector, tfidf_matrix).flatten()

    # Filter destinasi berdasarkan similarity scores yang melebihi threshold
    indices = [i for i, score in enumerate(sim_scores) if score]
    
    # Mengurutkan destinasi berdasarkan similarity scores
    indices = sorted(indices, key=lambda i: sim_scores[i], reverse=True)
    
    for i in indices:
        print(f"Destinasi: {df['destinasi'][i]}, Cosine Similarity: {sim_scores[i]}, Deskripsi : {df['deskripsi_preprocessing']}")

    # Mendapatkan 10 destinasi teratas
    top_destinations = indices[:10]

    # Mengembalikan destinasi yang direkomendasikan
    return df.iloc[top_destinations]

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommendation():
        try:
            keyword = request.form['keyword']
            recommendations = get_recommendations(keyword, cosine_sim)
        
            if recommendations.empty:
                return render_template('negative.html')
            
            return render_template('positive.html', keyword=keyword, recommendations=recommendations)

        except Exception as e:
            # Tangkap dan tampilkan pesan kesalahan
            error_message = str(e)
            return render_template('negative.html', error=error_message)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))
    app.run(host='0.0.0.0',port=port)