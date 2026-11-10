import streamlit as st
import subprocess
import json
import os
import pandas as pd

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Flower Federated Learning Dashboard",
    page_icon="🌸",
    layout="wide"
)

# Judul Utama
st.title("🌸 Flower Federated Learning + Streamlit Dashboard")
st.markdown("---")

# Sidebar Konfigurasi
st.sidebar.header("⚙️ Konfigurasi Simulasi")
num_rounds = st.sidebar.slider("Jumlah Round FL", min_value=1, max_value=20, value=3)
num_clients = st.sidebar.slider("Jumlah Client", min_value=2, max_value=10, value=2)

# Inisialisasi State Status
if "status" not in st.session_state:
    st.session_state.status = "Idle"

# Tombol Memulai Simulasi
if st.button("🚀 Mulai Federated Learning", type="primary"):
    st.session_state.status = "Running"
    
    # Hapus file hasil lama jika ada
    if os.path.exists("results.json"):
        os.remove("results.json")
        
    st.info("Memulai simulasi Flower...")
    
    # Perintah eksekusi run_simulation.py
    cmd = f"python run_simulation.py --rounds {num_rounds} --clients {num_clients}"
    
    with st.spinner("Training sedang berlangsung, mohon tunggu..."):
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True
        )
        stdout, _ = process.communicate()
        
    st.success("🎉 Training Selesai!")
    st.session_state.status = "Completed"
    st.session_state.log_output = stdout

# Tampilan Setelah Training Selesai
if st.session_state.status == "Completed":
    # 1. Menampilkan Grafik dan Tabel Evaluasi
    st.subheader("📈 Hasil Evaluasi (Loss Metric)")
    
    if os.path.exists("results.json"):
        with open("results.json", "r") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        df["rounds"] = df["rounds"].astype(str)  # Memastikan sumbu X terbaca sebagai kategori Round
        df = df.set_index("rounds")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.line_chart(df, use_container_width=True)
            
        with col2:
            st.write("**Ringkasan Metrik Per Round:**")
            st.dataframe(df, use_container_width=True)
            
            # Menampilkan Loss Terakhir
            latest_loss = df["loss"].iloc[-1]
            st.metric(label="Loss Terakhir (Final Round)", value=f"{latest_loss:.4f}")
    else:
        st.warning("File hasil evaluasi (`results.json`) tidak ditemukan.")

    st.markdown("---")

    # 2. Menampilkan Log Eksekusi Lengkap
    st.subheader("📜 Log Output Simulasi")
    if "log_output" in st.session_state:
        st.code(st.session_state.log_output, language="bash")