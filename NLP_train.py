import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources once and cache them
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except nltk.downloader.DownloadError:
        nltk.download('vader_lexicon')

download_nltk_data()

stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

def clean_text(text):
    if not isinstance(text, str):
        return "" # Handle non-string input
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def analyze_sentiments(text):
    if not isinstance(text, str):
        return pd.Series({'Compound': 0.0, 'Sentiment': 'Neutral'}) # Handle non-string input
    scores = sia.polarity_scores(text)
    sentiment = 'Positive' if scores['compound'] >= 0.05 else ('Negative' if scores['compound'] <= -0.05 else 'Neutral')
    return pd.Series({'Compound': scores['compound'], 'Sentiment': sentiment})

def main():
    st.title("CSV Sentiment Analysis App")
    st.write("Upload your CSV file below to perform sentiment analysis.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.write("### Preview of the uploaded data:")
            st.dataframe(df_uploaded.head())

            # User selects the text column
            text_columns = [col for col in df_uploaded.columns if df_uploaded[col].dtype == 'object']
            if not text_columns:
                st.warning("No suitable text columns found in the uploaded CSV for sentiment analysis.")
                return

            selected_text_column = st.selectbox("Select the column containing text for sentiment analysis:", text_columns)

            if st.button("Perform Sentiment Analysis"):
                with st.spinner("Analyzing sentiments..."):
                    # Clean text
                    df_uploaded['clean_text'] = df_uploaded[selected_text_column].apply(clean_text)

                    # Perform sentiment analysis
                    sentiment_results = df_uploaded['clean_text'].apply(analyze_sentiments)
                    df_uploaded = pd.concat([df_uploaded, sentiment_results], axis=1)

                    st.write("### Sentiment Analysis Results:")
                    st.dataframe(df_uploaded[[selected_text_column, 'clean_text', 'Compound', 'Sentiment']].head())

                    st.write("#### Sentiment Distribution:")
                    sentiment_counts = df_uploaded['Sentiment'].value_counts()
                    st.bar_chart(sentiment_counts) # Use st.bar_chart for simple plots

                    # Add an option to download the results
                    csv = df_uploaded.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label='Download Analysis Results as CSV',
                        data=csv,
                        file_name='sentiment_analysis_results.csv',
                        mime='text/csv',
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e) # Display full exception for debugging

if __name__ == "__main__":
    main()
""")
