import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# تحميل البيانات
df = pd.read_excel("AirQualityUCI.xlsx")
df.dropna(inplace=True)

# الأعمدة المستخدمة
features = [
    'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
    'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
    'PT08.S5(O3)', 'T', 'RH', 'AH'
]
target = 'CO(GT)'

# البيانات
X = df[features]
y = df[target]

# توحيد المقياس
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# تقسيم
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# التنبؤ
y_pred = model.predict(X_test)

# جدول النتائج
results_df = pd.DataFrame(X_test, columns=features)
results_df['Actual CO'] = y_test.values
results_df['Predicted CO'] = y_pred
results_df['T (Temp)'] = results_df['T']
results_df['RH (%)'] = results_df['RH']

# واجهة Streamlit
st.title("Air Quality Dashboard - Milan")

st.subheader("📊 Actual vs Predicted CO")
st.dataframe(results_df[['Actual CO', 'Predicted CO', 'T (Temp)', 'RH (%)']].head(20))

# رسم بياني CO
st.subheader("📉 Actual vs Predicted CO (GT)")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(results_df['Actual CO'].values[:50], label='Actual CO', marker='o')
ax1.plot(results_df['Predicted CO'].values[:50], label='Predicted CO', marker='x')
ax1.set_xlabel("Sample")
ax1.set_ylabel("CO Concentration")
ax1.legend()
st.pyplot(fig1)

# رسم درجة الحرارة
st.subheader("🌡️ Temperature Over Samples")
fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.plot(results_df['T (Temp)'].values[:50], color='orange', marker='o')
ax2.set_ylabel("Temp (°C)")
st.pyplot(fig2)

# رسم الرطوبة
st.subheader("💧 Humidity Over Samples")
fig3, ax3 = plt.subplots(figsize=(10, 3))
ax3.plot(results_df['RH (%)'].values[:50], color='blue', marker='x')
ax3.set_ylabel("Humidity (%)")
st.pyplot(fig3)

# تحذير عن القيم العالية
st.subheader("⚠️ High CO Levels (Predicted > 10)")
high_risk = results_df[results_df['Predicted CO'] > 10]
st.dataframe(high_risk[['Actual CO', 'Predicted CO', 'T (Temp)', 'RH (%)']])
