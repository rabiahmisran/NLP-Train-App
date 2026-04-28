import streamlit as st
import pandas as pd

st.set_page_config(page_title="NLP Training App", layout="wide")
st.title("📊 NLP Training")

def main():
    st.title("CSV File Uploader App")
    st.write("Upload your CSV file below:")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read the CSV file into a pandas DataFrame
            df_uploaded = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.write("### Preview of the uploaded data:")
            st.dataframe(df_uploaded.head())

            # Optionally, you can add a download button for the processed data or other functionalities
            # if st.button('Download Processed Data'):
            #     csv = df_processed.to_csv(index=False).encode('utf-8')
            #     st.download_button(
            #         label='Download data as CSV',
            #         data=csv,
            #         file_name='processed_data.csv',
            #         mime='text/csv',
            #     )

        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

if __name__ == "__main__":
    main()