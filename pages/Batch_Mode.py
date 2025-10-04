import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import pandas as pd

@st.cache_resource
def load_resources():
    model = load_model('churn_model.h5')
    with open('label_encoder_gender.pkl', 'rb') as f:
        label_encoder_gender = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, label_encoder_gender, scaler

model, label_encoder_gender, scaler = load_resources()

st.title("Batch Prediction Mode")
st.write("Upload a CSV file containing multiple customers to predict retention probabilities.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df.head())

    try:
        df_copy = df.copy()
        df_copy = df_copy.drop(['RowNumber', 'CustomerId', 'Surname','Geography','Exited'], axis=1)
        df_copy['Gender'] = label_encoder_gender.transform(df['Gender'])
        scaled_data = scaler.transform(df_copy)
        preds = model.predict(scaled_data)

        df['Probability'] = preds
        df['Prediction'] = (df['Probability'] > 0.5).astype(int)

        display_cols = ['CustomerId', 'Surname', 'Gender', 'Age', 'Prediction']

        st.success("Predictions completed successfully!")
        
        df = df[display_cols]
        st.subheader("Filter Results by Prediction")
        filter_option = st.selectbox(
            "Show customers:",
            ("All", "Likely to Stay", "Likely to Leave")
        )

        if filter_option == "Likely to Leave":
            filtered_df = df[df["Prediction"] == 1]
        elif filter_option == "Likely to Stay":
            filtered_df = df[df["Prediction"] == 0]
        else:
            filtered_df = df

        st.write(f"Showing {len(filtered_df)} records:")
        st.dataframe(filtered_df[display_cols])

        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Filtered Predictions as CSV", csv, "filtered_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
