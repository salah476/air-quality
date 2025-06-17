import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Air Quality Dashboard - Milan")

uploaded_file = "AirQualityUCI.xlsx"
df = pd.read_excel(uploaded_file)

st.write("Preview of dataset:")
st.dataframe(df.head())

# رسم المتغيرات الخطيرة مقابل حساس CO
columns_to_plot = [
    'CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
    'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
    'PT08.S5(O3)', 'T', 'RH', 'AH'
]

sensor_col = 'PT08.S1(CO)'

st.subheader("Relationships with CO Sensor")
fig, axes = plt.subplots(len(columns_to_plot), 1, figsize=(8, len(columns_to_plot)*2))

for i, col in enumerate(columns_to_plot):
    sns.lineplot(data=df[[col, sensor_col]].dropna(), x=sensor_col, y=col, ax=axes[i])
    axes[i].set_title(f"{col} vs {sensor_col}")

st.pyplot(fig)
