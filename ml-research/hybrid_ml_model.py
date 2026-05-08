import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

print("🚀 Loading Data for Hybrid ML-IRT Model...")
df = pd.read_csv("omr_synthetic_data.csv")

# যদি কোনো কারণে IRT এর ডাটা মিসিং থাকে, সেটা বাদ দেওয়া
df = df.dropna(subset=['Estimated_Ability'])

# 🎯 ফিচার সিলেকশন (Features and Target)
# Hybrid Approach: আমরা Q1-Q10 এর উত্তর, টোটাল স্কোর এবং IRT-এর প্রেডিকশন সব একসাথে ফিচার হিসেবে নিচ্ছি
feature_cols = [f"Q{i+1}" for i in range(10)] + ['Total_Raw_Score', 'Estimated_Ability']
X = df[feature_cols]

# Target: স্টুডেন্টের আসল মেধা, যেটা আমরা প্রেডিক্ট করে মেলাবো
y = df['True_Ability_Theta'] 

# ✂️ Train-Test Split (মডেলকে ৮০% ডাটা দিয়ে শেখাবো, ২০% দিয়ে পরীক্ষা নেবো)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 Training Random Forest Regressor Model... Please wait...")
# Random Forest মডেল তৈরি ও ট্রেইনিং
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 🎯 প্রেডিকশন
y_pred_ml = rf_model.predict(X_test) # ML এর প্রেডিকশন
y_pred_irt = X_test['Estimated_Ability'] # পিওর IRT এর প্রেডিকশন

# 📏 এরর ক্যালকুলেশন (RMSE - Root Mean Squared Error)
# যার RMSE যত কম, সে তত ভালো!
rmse_irt = np.sqrt(mean_squared_error(y_test, y_pred_irt))
rmse_ml = np.sqrt(mean_squared_error(y_test, y_pred_ml))

print("\n" + "="*50)
print("🏆 PERFORMANCE COMPARISON (RMSE)")
print(f"📊 Pure IRT Model Error: {rmse_irt:.4f}")
print(f"🤖 Hybrid ML-IRT Model Error: {rmse_ml:.4f}")
print("="*50)

if rmse_ml < rmse_irt:
    print("🎉 SUCCESS! The Hybrid ML model outperformed the Pure IRT model!")
    print("গবেষণার মূল হাইপোথিসিস প্রমাণিত! ML মডেল সাধারণ IRT-এর চেয়ে বেশি নিখুঁত।")
else:
    print("🤔 Interesting. The ML model didn't beat IRT yet. We might need hyperparameter tuning.")

# 📈 ভিজ্যুয়ালাইজেশন: কোন মডেল কতটা নিখুঁত?
plt.figure(figsize=(10, 6))
# পিওর IRT এর পারফরম্যান্স
plt.scatter(y_test, y_pred_irt, alpha=0.5, color='blue', label=f'Pure IRT (RMSE: {rmse_irt:.3f})')
# হাইব্রিড ML এর পারফরম্যান্স
plt.scatter(y_test, y_pred_ml, alpha=0.7, color='red', marker='x', label=f'Hybrid ML (RMSE: {rmse_ml:.3f})')

# একদম পারফেক্ট প্রেডিকশনের লাইন
plt.plot([-3, 3], [-3, 3], color='green', linestyle='--', label='Perfect Prediction (Ground Truth Line)')

plt.title('Prediction Accuracy: Pure IRT vs Hybrid ML-IRT')
plt.xlabel('True Student Ability (Actual Theta)')
plt.ylabel('Predicted Student Ability')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('ml_vs_irt_comparison.png', dpi=300)
print("\n✅ Comparison graph saved successfully as 'ml_vs_irt_comparison.png'!")