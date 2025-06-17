import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_excel("AirQualityUCI.xlsx")
df.dropna(inplace=True)

features = [
    'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
    'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
    'PT08.S5(O3)', 'T', 'RH', 'AH'
]
target = 'CO(GT)'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

results_df = X_test.copy()
results_df['Actual CO'] = y_test
results_df['Predicted CO'] = y_pred

st.title("Air Quality Dashboard - Milan")

st.subheader("📊 Preview of Dataset")
st.dataframe(results_df.head(10))

st.subheader("📉 Actual vs Predicted CO (GT)")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(results_df['Actual CO'].values[:50], label='Actual CO', marker='o')
ax.plot(results_df['Predicted CO'].values[:50], label='Predicted CO', marker='x')
ax.set_xlabel("Sample")
ax.set_ylabel("CO Concentration")
ax.legend()
st.pyplot(fig)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.write(f"RMSE: {rmse:.2f}")
st.write(f"R² Score: {r2:.2f}")

danger_threshold = 10
high_risk = results_df[results_df['Predicted CO'] > danger_threshold]

st.subheader("⚠️ High CO Levels (Predicted > 10)")
st.dataframe(high_risk[['Actual CO', 'Predicted CO']])
