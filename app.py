import streamlit as st
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io

# --- Header ---
st.set_page_config(
    page_title="Spektrofotometer Alternatif",
    page_icon="🧪",
    layout="centered"
)
st.title("🧪 Spektrofotometer Alternatif")
st.markdown("""
**Deskripsi:**  
Aplikasi ini membantu mengukur konsentrasi larutan dari foto larutan menggunakan pendekatan olah citra digital intensitas warna RGB.  
**Fitur:**  
- Upload foto larutan  
- Crop ROI untuk standar & sampel  
- Simpan data kalibrasi & prediksi otomatis  
- Download hasil dalam CSV/XLSX  
Optimalkan analisis berbasis warna menggunakan spektrofotometri dengan mudah, cepat, dan mobile-friendly
- © Amin Fatoni 2025  
""")

# --- Session state untuk menyimpan data ---
if "data" not in st.session_state:
    st.session_state.data = []

# --- File uploader custom ---
uploaded_file = st.file_uploader(
    "📷 Upload Foto Larutan/Tabung",
    type=["jpg","jpeg","png"],
    help="Di HP, klik akan otomatis menawarkan kamera atau galeri."
)

if uploaded_file:
    image = Image.open(uploaded_file)

    # Mode pilihan
    tab = st.radio(
        "Pilih mode:",
        ["Kalibrasi Standar", "Prediksi Sampel"],
        index=0,
        horizontal=True
    )

    if tab == "Kalibrasi Standar":
        st.subheader("Crop ROI Tabung Standar")
        cropped_img = st_cropper(
            image,
            realtime_update=True,
            box_color="red",
            aspect_ratio=None,
            help="Geser/ubah ukuran kotak crop untuk menyesuaikan tabung standar"
        )
        cropped_arr = np.array(cropped_img)
        mean_rgb = cropped_arr.mean(axis=(0,1)).astype(int)
        st.image(cropped_img, caption="ROI Tabung Standar", use_container_width=True)
        st.info(f"Rata-rata RGB ROI: {mean_rgb}")

        with st.form("input_data"):
            konsentrasi = st.number_input("Konsentrasi standar (mM)", min_value=0.0, step=0.1)
            submitted = st.form_submit_button("💾 Simpan Data")
            if submitted:
                st.session_state.data.append({
                    "Konsentrasi": konsentrasi,
                    "R": mean_rgb[0],
                    "G": mean_rgb[1],
                    "B": mean_rgb[2]
                })
                st.success("Data standar berhasil disimpan!")

    elif tab == "Prediksi Sampel":
        if not st.session_state.data:
            st.warning("Belum ada data standar. Silakan buat kalibrasi dulu.")
        else:
            st.subheader("Crop ROI Tabung Sampel")
            cropped_img = st_cropper(
                image,
                realtime_update=True,
                box_color="blue",
                aspect_ratio=None,
                help="Geser/ubah ukuran kotak crop untuk menyesuaikan tabung sampel"
            )
            cropped_arr = np.array(cropped_img)
            mean_rgb = cropped_arr.mean(axis=(0,1)).astype(int)
            st.image(cropped_img, caption="ROI Tabung Sampel", use_container_width=True)
            st.info(f"Rata-rata RGB ROI Sampel: {mean_rgb}")

            # Prediksi konsentrasi
            df = pd.DataFrame(st.session_state.data)
            X = df["Konsentrasi"].values.reshape(-1, 1)
            results = []
            for channel in ["R","G","B"]:
                y = df[channel].values
                model = LinearRegression().fit(X, y)
                coef = model.coef_[0]
                intercept = model.intercept_
                r2 = model.score(X, y)
                y_sample = mean_rgb[["R","G","B"].index(channel)]
                konsentrasi_pred = (y_sample - intercept)/coef
                results.append({
                    "Channel": channel,
                    "Persamaan": f"y = {coef:.2f}x + {intercept:.2f}",
                    "R²": r2,
                    "Prediksi Konsentrasi": konsentrasi_pred
                })
            results_df = pd.DataFrame(results)
            st.subheader("Hasil Prediksi Sampel")
            st.dataframe(results_df, use_container_width=True)

# --- Tampilkan data kalibrasi & kurva ---
if st.session_state.data:
    df = pd.DataFrame(st.session_state.data)
    st.subheader("Data Kalibrasi")
    st.dataframe(df, use_container_width=True)

    X = df["Konsentrasi"].values.reshape(-1,1)
    fig, ax = plt.subplots()
    colors = {"R":"red","G":"green","B":"blue"}
    for channel in ["R","G","B"]:
        y = df[channel].values
        model = LinearRegression().fit(X,y)
        y_pred = model.predict(X)
        coef = model.coef_[0]
        intercept = model.intercept_
        r2 = model.score(X,y)
        ax.scatter(df["Konsentrasi"],y,color=colors[channel],label=f"{channel} data")
        ax.plot(df["Konsentrasi"],y_pred,color=colors[channel],linestyle="--",
                label=f"{channel} fit: y={coef:.2f}x+{intercept:.2f}, R²={r2:.3f}")
    ax.set_xlabel("Konsentrasi (mM)")
    ax.set_ylabel("Intensitas RGB")
    ax.set_title("Kurva Kalibrasi RGB")
    ax.legend()
    st.pyplot(fig)

# --- Download data ---
with st.expander("📥 Download Data"):
    if st.session_state.data:
        df = pd.DataFrame(st.session_state.data)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Data Kalibrasi (CSV)", data=csv_bytes, file_name="data_kalibrasi.csv")
        try:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Kalibrasi", index=False)
                if 'results_df' in locals():
                    results_df.to_excel(writer, sheet_name="Prediksi", index=False)
            st.download_button("⬇️ Gabungan (XLSX)", data=excel_buffer.getvalue(), file_name="kalibrasi_prediksi.xlsx")
        except Exception:
            st.info("Install `openpyxl` untuk download Excel.")

