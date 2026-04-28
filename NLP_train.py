# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords

# download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# ===============================
# 2. TEXT CLEANING FUNCTION
# ===============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


# ===============================
# 3. STREAMLIT UI
# ===============================
st.title("📊 NLP Sentiment Analysis Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:

    # ===============================
    # 4. LOAD DATA
    # ===============================
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Data")
    st.write(df.head())

    # ===============================
    # 5. COLUMN SELECTION
    # ===============================
    text_column = st.selectbox("Select Text Column", df.columns)
    label_column = st.selectbox("Select Label Column (optional)", ["None"] + list(df.columns))

    # ===============================
    # 6. CLEAN TEXT
    # ===============================
    df['cleaned'] = df[text_column].apply(clean_text)

    st.subheader("🧹 Cleaned Text")
    st.write(df[['cleaned']].head())

    # ===============================
    # 7. TRAIN MODEL (if label exists)
    # ===============================
    if label_column != "None":

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df['cleaned'])
        y = df[label_column]

        model = LogisticRegression()
        model.fit(X, y)

        st.success("✅ Model trained successfully!")

        # ===============================
        # 8. VISUALIZATION
        # ===============================
        st.subheader("📊 Label Distribution")

        fig, ax = plt.subplots()
        df[label_column].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

        # ===============================
        # 9. PREDICT NEW TEXT
        # ===============================
        st.subheader("🔍 Try Your Own Text")

        user_input = st.text_area("Enter text")

        if st.button("Predict"):
            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)

            st.write("Prediction:", prediction[0])

    else:
        st.warning("⚠️ Please select a label column to train model")
