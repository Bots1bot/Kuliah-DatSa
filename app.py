import streamlit as st
import pandas as pd
import pickle
import os

st.title('Prediksi Tagihan Listrik Jakarta')
st.write('Aplikasi untuk memprediksi jumlah tagihan listrik berdasarkan parameter yang diberikan.')

st.sidebar.header('Input Parameter')

def user_input_features():
    kwh = st.sidebar.slider('Konsumsi KWH (kWh)', 150.0, 600.0, 350.0)
    ac_units = st.sidebar.slider('Jumlah AC', 0, 3, 1)
    ac_hours_per_day = st.sidebar.slider('Jam AC per Hari', 0.0, 10.0, 5.0)
    family_size = st.sidebar.slider('Jumlah Anggota Keluarga', 2, 6, 4)

    month_name = st.sidebar.selectbox('Bulan', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    tariff_class = st.sidebar.selectbox('Kelas Tarif', ['R1', 'R2', 'R3'])

    data = {
        'kwh': kwh,
        'ac_units': ac_units,
        'ac_hours_per_day': ac_hours_per_day,
        'family_size': family_size,
        'month_name': month_name,
        'tariff_class': tariff_class
    }
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()
st.subheader('Parameter Input Pengguna:')
st.write(df_input)

# Kolom dan dtype yang diharapkan model saat training
training_columns_and_dtypes = {
    'kwh': 'float64',
    'ac_units': 'int64',
    'ac_hours_per_day': 'float64',
    'family_size': 'int64',
    'month_name_Aug': 'bool', 'month_name_Dec': 'bool', 'month_name_Feb': 'bool',
    'month_name_Jan': 'bool', 'month_name_Jul': 'bool', 'month_name_Jun': 'bool',
    'month_name_Mar': 'bool', 'month_name_May': 'bool', 'month_name_Nov': 'bool',
    'month_name_Oct': 'bool', 'month_name_Sep': 'bool',
    'tariff_class_R2': 'bool', 'tariff_class_R3': 'bool'
}

# Buat satu baris default dengan tipe sesuai (bool False, numeric 0/0.0)
default_row = {}
for col, dtype in training_columns_and_dtypes.items():
    if dtype == 'bool':
        default_row[col] = False
    elif 'int' in dtype:
        default_row[col] = 0
    else:
        default_row[col] = 0.0

final_input_df = pd.DataFrame([default_row]).astype(training_columns_and_dtypes)

# Isi fitur numerik dari input user
final_input_df.loc[0, 'kwh'] = float(df_input.loc[0, 'kwh'])
final_input_df.loc[0, 'ac_units'] = int(df_input.loc[0, 'ac_units'])
final_input_df.loc[0, 'ac_hours_per_day'] = float(df_input.loc[0, 'ac_hours_per_day'])
final_input_df.loc[0, 'family_size'] = int(df_input.loc[0, 'family_size'])

# Set one-hot untuk month dan tariff (jika kolom tersedia — model mungkin menggunakan baseline)
selected_month_col = f"month_name_{df_input.loc[0, 'month_name']}"
if selected_month_col in final_input_df.columns:
    final_input_df.loc[0, selected_month_col] = True

selected_tariff_col = f"tariff_class_{df_input.loc[0, 'tariff_class']}"
if selected_tariff_col in final_input_df.columns:
    final_input_df.loc[0, selected_tariff_col] = True

# Pastikan urutan kolom sesuai definisi training (opsional tapi aman)
final_input_df = final_input_df[list(training_columns_and_dtypes.keys())]

# Fungsi load model yang benar (gunakan with open(..., 'rb'))
@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file tidak ditemukan: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)

model_path = 'linear_regression_model.pkl'

if st.sidebar.button('Prediksi Tagihan'):
    try:
        model = load_model(model_path)
        prediction = model.predict(final_input_df)
        st.subheader('Hasil Prediksi Tagihan Listrik:')
        st.write(f"Tagihan Diprediksi: Rp {prediction[0]:,.2f}")
    except FileNotFoundError as fnf:
        st.error(str(fnf))
    except Exception as e:
        st.error("Terjadi kesalahan saat melakukan prediksi. Detail di bawah:")
        st.exception(e)
