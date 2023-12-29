import streamlit as st
import pickle

model = pickle.load(open('prediksi_diabetes.sav', 'rb'))

    
st.title('prediksi_diabetes')

Pregnancies = st.number_input(
    'Masukkan berapa kali hamil', min_value=0.0, max_value=17.0, step=1.0)

Glocose = st.number_input(
    'Masukkan Konsentrasi glukosa plasma dalam tes toleransi glukosa oral.', min_value=0.0, max_value=199.0, step=1.0)

BloodPressure  = st.number_input(
    'Masukkan Tekanan darah diastolik (mm Hg).', min_value=0.0, max_value=122.0, step=0.1)

SkinThickness  = st.number_input(
    'Masukkan Ketebalan lipatan kulit trisep (mm).', min_value=0.0, max_value=99.0, step=0.1)

Insulin = st.number_input(
    'Masukkan nilai Insulin selama 2 jam', min_value=0.0, max_value=846.0, step=0.1)

BMI = st.number_input(
    'Masukkan nilai Index Berat Badan.', min_value=0.0, max_value=67.1, step=0.1)

DiabetesPedigreeFunction = st.number_input(
    'Masukkan nilai Fungsi Diabetes.', min_value=0.08, max_value=2.42, step=0.1)

Age = st.number_input(
    'Masukkan nilai Usia.', min_value=21.0, max_value=81.0, step=0.1)


predict = ''

if st.button('Prediksi'):
    input_data = [[Pregnancies, Glocose, BloodPressure , SkinThickness, Insulin ,
                   BMI , DiabetesPedigreeFunction, Age]]
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        predict = 'Pasien diprediksi Terkena Diabtes.'
    elif prediction[0] == 1:
        predict = 'Pasien diprediksi Tidak terkena Diabetes.'
st.write(predict)