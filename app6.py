import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

# تقسيم البيانات
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# التنبؤ
y_pred = model.predict(X_test)

# تحضير البيانات للعرض
results_df = X_test.copy()
results_df['Actual CO'] = y_test.values
results_df['Predicted CO'] = y_pred
results_df['Temperature'] = X_test['T'].values
results_df['Humidity'] = X_test['RH'].values

# واجهة Streamlit
st.title("🌍 Milan Air Quality Dashboard")
st.subheader("📊 Actual vs Predicted CO Levels")
st.dataframe(results_df[['Actual CO', 'Predicted CO', 'Temperature', 'Humidity']].head(20))

# رسم CO
st.subheader("📉 Actual vs Predicted CO (First 50 Samples)")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(results_df['Actual CO'].values[:50], label='Actual CO', marker='o')
ax1.plot(results_df['Predicted CO'].values[:50], label='Predicted CO', marker='x')
ax1.set_xlabel("Sample Index")
ax1.set_ylabel("CO (GT)")
ax1.legend()
st.pyplot(fig1)

# رسم درجة الحرارة الأصلية
st.subheader("🌡️ Temperature (°C) - First 50 Samples")
fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.plot(results_df['Temperature'].values[:50], color='orange', marker='o')
ax2.set_ylabel("Temperature °C")
st.pyplot(fig2)

# رسم الرطوبة الأصلية
st.subheader("💧 Humidity (%) - First 50 Samples")
fig3, ax3 = plt.subplots(figsize=(10, 3))
ax3.plot(results_df['Humidity'].values[:50], color='blue', marker='x')
ax3.set_ylabel("Relative Humidity (%)")
st.pyplot(fig3)

# تحذير
st.subheader("⚠️ High CO Levels (Predicted > 10)")
high_risk = results_df[results_df['Predicted CO'] > 10]
st.dataframe(high_risk[['Actual CO', 'Predicted CO', 'Temperature', 'Humidity']])

# دقة النموذج
mae = mean_absolute_error(y_test, y_pred)
st.markdown(f"### 📏 Mean Absolute Error (MAE): `{mae:.2f}`")
