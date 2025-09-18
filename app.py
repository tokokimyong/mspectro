import streamlit as st
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("Spektrofotometer Alternatif")
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

# Session state untuk menyimpan data


if "data" not in st.session_state:


    st.session_state.data = []

# Session state untuk menyimpan data
if "data" not in st.session_state:
    st.session_state.data = []


uploaded_file = st.file_uploader("Upload foto larutan/tabung", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    tab = st.radio("Pilih mode:", ["Kalibrasi Standar", "Prediksi Sampel"])

    if tab == "Kalibrasi Standar":
        st.write("### Pilih ROI untuk 1 tabung standar (crop manual)")
        cropped_img = st_cropper(image, realtime_update=True, box_color="red", aspect_ratio=None)
        cropped_arr = np.array(cropped_img)
        mean_rgb = cropped_arr.mean(axis=(0,1)).astype(int)

        st.image(cropped_img, caption="ROI Tabung Standar", use_container_width=True)
        st.write(f"Rata-rata RGB ROI: {mean_rgb}")

        with st.form("input_data"):
            konsentrasi = st.number_input("Masukkan konsentrasi standar (misal mM)", min_value=0.0, step=0.1)
            submitted = st.form_submit_button("Simpan Data")
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
            st.warning("Belum ada data standar. Silakan buat kurva kalibrasi dulu.")
        else:
            st.write("### Pilih ROI untuk tabung sampel")
            cropped_img = st_cropper(image, realtime_update=True, box_color="blue", aspect_ratio=None)
            cropped_arr = np.array(cropped_img)
            mean_rgb = cropped_arr.mean(axis=(0,1)).astype(int)

            st.image(cropped_img, caption="ROI Tabung Sampel", use_container_width=True)
            st.write(f"Rata-rata RGB ROI Sampel: {mean_rgb}")

            # Prediksi konsentrasi untuk semua channel
            df = pd.DataFrame(st.session_state.data)
            X = df["Konsentrasi"].values.reshape(-1, 1)

            results = []
            for channel in ["R", "G", "B"]:
                y = df[channel].values
                model = LinearRegression().fit(X, y)
                coef = model.coef_[0]
                intercept = model.intercept_
                r2 = model.score(X, y)

                y_sample = mean_rgb[["R","G","B"].index(channel)]
                konsentrasi_pred = (y_sample - intercept) / coef

                results.append({
                    "Channel": channel,
                    "Persamaan": f"y = {coef:.2f}x + {intercept:.2f}",
                    "R²": r2,
                    "Prediksi Konsentrasi": konsentrasi_pred
                })

            results_df = pd.DataFrame(results)
            st.write("### Hasil Prediksi Sampel")
            st.dataframe(results_df, use_container_width=True)

# Jika sudah ada data kalibrasi, tampilkan tabel + kurva
if st.session_state.data:
    df = pd.DataFrame(st.session_state.data)
    st.write("### Data Kalibrasi")
    st.dataframe(df, use_container_width=True)

    # Plot kurva kalibrasi
    X = df["Konsentrasi"].values.reshape(-1, 1)
    fig, ax = plt.subplots()
    colors = {"R": "red", "G": "green", "B": "blue"}

    for channel in ["R", "G", "B"]:
        y = df[channel].values
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        coef = model.coef_[0]
        intercept = model.intercept_
        r2 = model.score(X, y)

        ax.scatter(df["Konsentrasi"], y, color=colors[channel], label=f"{channel} data")
        ax.plot(df["Konsentrasi"], y_pred, color=colors[channel], linestyle="--",
                label=f"{channel} fit: y = {coef:.2f}x + {intercept:.2f}, R² = {r2:.3f}")

    ax.set_xlabel("Konsentrasi")
    ax.set_ylabel("Intensitas RGB")
    ax.set_title("Kurva Kalibrasi RGB")
    ax.legend()
    st.pyplot(fig)

import io

with st.expander("📥 Download Data"):
    if st.session_state.data:
        df = pd.DataFrame(st.session_state.data)
        st.write("### Data Kalibrasi")
        st.dataframe(df, use_container_width=True)

        # CSV kalibrasi
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Data Kalibrasi (CSV)",
            data=csv_bytes,
            file_name="data_kalibrasi.csv",
            mime="text/csv"
        )

        # Excel kalibrasi (butuh openpyxl)
        try:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Kalibrasi', index=False)
            st.download_button(
                label="⬇️ Data Kalibrasi (XLSX)",
                data=excel_buffer.getvalue(),
                file_name="data_kalibrasi.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception:
            st.info("Install `openpyxl` untuk download Excel.")

    if 'results_df' in locals():
        st.write("### Hasil Prediksi Sampel")
        st.dataframe(results_df, use_container_width=True)

        # CSV prediksi
        csv_pred = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Prediksi Sampel (CSV)",
            data=csv_pred,
            file_name="prediksi_sampel.csv",
            mime="text/csv"
        )

        # Excel gabungan
        try:
            xls_buf = io.BytesIO()
            with pd.ExcelWriter(xls_buf, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Kalibrasi', index=False)
                results_df.to_excel(writer, sheet_name='Prediksi', index=False)
            st.download_button(
                label="⬇️ Gabungan Kalibrasi+Prediksi (XLSX)",
                data=xls_buf.getvalue(),
                file_name="kalibrasi_dan_prediksi.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception:
            pass

