import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="💼 AML Fraud Detection", layout="centered")
st.title("💼 AML Fraud Detection")

model = joblib.load("model/fraud_model.pkl")
scaler = joblib.load("model/scaler.pkl")
data = pd.read_csv("data/Big_Black_Money_Dataset.csv")
transaction_types = sorted(data['Transaction Type'].dropna().unique())
countries = sorted(data['Country'].dropna().unique())

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📝 Input Manual", "📁 Upload CSV", "📊 Visualisasi", 
    "📥 Download Hasil", "❓ Bantuan", "📊 Evaluasi Model"
])

with tab1:
    st.subheader("🔍 Masukkan Transaksi Baru")
    col1, col2 = st.columns(2)
    with col1:
        trans_type = st.selectbox("Transaction Type", transaction_types)
        country = st.selectbox("Country", countries)
    with col2:
        amount = st.number_input("Amount (USD)", min_value=0.0, value=100.0)
    
    if st.button("Prediksi Risiko"):
        input_df = pd.DataFrame({
            'Transaction Type': [trans_type],
            'Amount (USD)': [amount],
            'Country': [country]
        })
        input_encoded = pd.get_dummies(input_df)
        full_encoded = pd.get_dummies(data[['Transaction Type', 'Amount (USD)', 'Country']])
        for col in full_encoded.columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[full_encoded.columns]
        input_scaled = scaler.transform(input_encoded)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]
        st.markdown(f"### 💡 Hasil Prediksi: {'❌ Tinggi' if prediction == 1 else '✅ Rendah'}")
        st.progress(min(int(proba * 100), 100))
        st.markdown(f"**Probabilitas Risiko: {proba:.2f}**")

with tab2:
    uploaded_file = st.file_uploader("Upload file transaksi (CSV)", type="csv")
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        st.write("📄 Data yang diupload:", df_upload.head())
        df_encoded = pd.get_dummies(df_upload[['Transaction Type', 'Amount (USD)', 'Country']])
        full_encoded = pd.get_dummies(data[['Transaction Type', 'Amount (USD)', 'Country']])
        for col in full_encoded.columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[full_encoded.columns]
        df_scaled = scaler.transform(df_encoded)
        df_upload['Prediksi'] = model.predict(df_scaled)
        st.success("✅ Prediksi selesai.")
        st.write(df_upload)
        st.session_state['predicted_batch'] = df_upload

with tab3:
    if 'predicted_batch' in st.session_state:
        result = st.session_state['predicted_batch']
        chart_data = result['Prediksi'].value_counts().rename({0: 'Aman', 1: 'Mencurigakan'})
        fig, ax = plt.subplots()
        chart_data.plot.pie(autopct="%.1f%%", ylabel="", ax=ax)
        st.pyplot(fig)
    else:
        st.info("🔁 Upload file di tab 📁 Upload CSV terlebih dahulu.")

with tab4:
    if 'predicted_batch' in st.session_state:
        csv = st.session_state['predicted_batch'].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV Prediksi", csv, "hasil_prediksi.csv", "text/csv")
    else:
        st.info("⚠️ Belum ada hasil prediksi. Upload file di tab 📁 terlebih dahulu.")

with tab5:
    st.markdown("### ❓ Panduan Aplikasi")
    st.write("- **Transaction Type**: jenis transaksi (misal Transfer, Withdrawal, dsb.)")
    st.write("- **Amount (USD)**: jumlah transaksi dalam USD")
    st.write("- **Country**: negara asal transaksi")
    st.write("- Label 1 = Mencurigakan, Label 0 = Aman")

    st.markdown("### 🔢 Penjelasan Fitur Lanjutan (Untuk Dataset Lengkap)")
    st.table({
        "Fitur": [
            "transaction_type", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "count_txn_by_orig",
            "sum_txn_by_orig", "avg_amount_by_orig", "count_unique_dest", "count_txn_by_dest"
        ],
        "Deskripsi": [
            "Jenis transaksi (TRANSFER, PAYMENT, CASH_OUT, dll)",
            "Jumlah nominal uang yang ditransaksikan",
            "Saldo awal akun pengirim",
            "Saldo akhir akun pengirim setelah transaksi",
            "Saldo awal akun penerima",
            "Saldo akhir akun penerima setelah transaksi",
            "Total transaksi sebelumnya oleh akun pengirim",
            "Total nominal dari semua transaksi pengirim",
            "Rata-rata nominal transaksi oleh pengirim",
            "Banyaknya akun penerima unik dari akun pengirim",
            "Jumlah transaksi yang diterima oleh akun tujuan"
        ]
    })

    st.markdown("💡 Template CSV bisa disesuaikan dari data asli.")

with tab6:
    st.subheader("📊 Evaluasi Model")
    st.table({
        "Metric": ["Accuracy", "ROC AUC", "Recall (Fraud)", "Precision (Fraud)"],
        "Train Set": ["96%", "97.5%", "86%", "3%"],
        "Test Set": ["96%", "97.7%", "85%", "3%"]
    })
    st.markdown("> Model memiliki generalisasi baik dan **tidak overfitting** meskipun data sangat tidak seimbang (fraud hanya <0.1%).")
