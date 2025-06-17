import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# تحميل البيانات
df = pd.read_excel("AirQualityUCI.xlsx")

# تنظيف البيانات: إزالة القيم المفقودة والخاطئة
df.replace(-200, pd.NA, inplace=True)
df.dropna(inplace=True)

# الأعمدة المهمة
features = [
    'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
    'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
    'PT08.S5(O3)', 'T', 'RH', 'AH'
]
target = 'CO(GT)'

# تحديد X و y
X = df[features]
y = df[target]

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# نموذج Random Forest
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# التنبؤ
y_pred = model.predict(X_test)

# MAE
mae = mean_absolute_error(y_test, y_pred)

# تجهيز جدول النتائج
results_df = X_test.copy()
results_df['Actual CO'] = y_test
results_df['Predicted CO'] = y_pred
results_df['T (Temp)'] = X_test['T']
results_df['RH (%)'] = X_test['RH']

# Streamlit واجهة
st.title("🌍 Milan Air Quality Dashboard")
st.markdown("### 📏 Mean Absolute Error (MAE): {:.2f}".format(mae))

# عرض أول 20 عينة
st.subheader("🔍 Sample of Actual vs Predicted CO + Temp & RH")
st.dataframe(results_df[['Actual CO', 'Predicted CO', 'T (Temp)', 'RH (%)']].head(20))

# رسم Actual vs Predicted CO
st.subheader("📈 CO(GT): Actual vs Predicted")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(results_df['Actual CO'].values[:50], label="Actual", marker='o')
ax1.plot(results_df['Predicted CO'].values[:50], label="Predicted", marker='x')
ax1.set_ylabel("CO Concentration")
ax1.legend()
st.pyplot(fig1)

# رسم الحرارة
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

# عينات فيها CO متوقع > 10
danger_threshold = 10
high_risk = results_df[results_df['Predicted CO'] > danger_threshold]

st.subheader("⚠️ High CO Levels (Predicted > 10)")
st.dataframe(high_risk[['Actual CO', 'Predicted CO', 'T (Temp)', 'RH (%)']])
