import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# تحميل البيانات
df = pd.read_excel("AirQualityUCI.xlsx")

# إزالة القيم المفقودة
df.dropna(inplace=True)

# اختيار الأعمدة المؤثرة
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
model = LinearRegression()
model.fit(X_train, y_train)

# التنبؤ
y_pred = model.predict(X_test)

# إعداد النتائج للعرض
results_df = X_test.copy()
results_df['Actual CO'] = y_test.values
results_df['Predicted CO'] = y_pred
results_df['T (Temp)'] = X_test['T'].values
results_df['RH (%)'] = X_test['RH'].values

# واجهة Streamlit
st.title("Air Quality Dashboard - Milan")

st.subheader("📊 Preview of Results (Actual vs Predicted CO)")
st.dataframe(results_df[['Actual CO', 'Predicted CO', 'T (Temp)', 'RH (%)']].head(20))

# رسم بياني للمقارنة بين القيم الفعلية والمتوقعة للـ CO
st.subheader("📉 Actual vs Predicted CO (GT)")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(results_df['Actual CO'].values[:50], label='Actual CO', marker='o')
ax1.plot(results_df['Predicted CO'].values[:50], label='Predicted CO', marker='x')
ax1.set_xlabel("Sample")
ax1.set_ylabel("CO Concentration")
ax1.legend()
st.pyplot(fig1)

# رسم بياني لدرجة الحرارة
st.subheader("🌡️ Temperature Over Samples")
fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.plot(results_df['T (Temp)'].values[:50], color='orange', marker='o')
ax2.set_title("Temperature")
ax2.set_xlabel("Sample")
ax2.set_ylabel("Temperature (°C)")
st.pyplot(fig2)

# رسم بياني للرطوبة
st.subheader("💧 Humidity Over Samples")
fig3, ax3 = plt.subplots(figsize=(10, 3))
ax3.plot(results_df['RH (%)'].values[:50], color='blue', marker='x')
ax3.set_title("Relative Humidity")
ax3.set_xlabel("Sample")
ax3.set_ylabel("Humidity (%)")
st.pyplot(fig3)

# تحذير من القيم العالية
danger_threshold = 10
high_risk = results_df[results_df['Predicted CO'] > danger_threshold]

st.subheader("⚠️ High CO Levels (Predicted > 10)")
st.dataframe(high_risk[['Actual CO', 'Predicted CO', 'T (Temp)', 'RH (%)']])
