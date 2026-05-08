import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

print("🚀 Loading Data for Advanced ML Model...")
df = pd.read_csv("omr_synthetic_data.csv").dropna(subset=['Estimated_Ability'])

# ফিচার এবং টার্গেট সেট করা
feature_cols = [f"Q{i+1}" for i in range(10)] + ['Total_Raw_Score', 'Estimated_Ability']
X = df[feature_cols]
y = df['True_Ability_Theta']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 Tuning Gradient Boosting Model... This might take a few seconds...")
# মডেলকে আরও স্মার্ট করার জন্য বিভিন্ন অপশন দিয়ে টেস্ট করা (Hyperparameter Tuning)
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [2, 3, 4]
}

gb_model = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(gb_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# বেস্ট মডেল দিয়ে প্রেডিকশন
best_ml_model = grid_search.best_estimator_
y_pred_ml = best_ml_model.predict(X_test)
y_pred_irt = X_test['Estimated_Ability']

rmse_irt = np.sqrt(mean_squared_error(y_test, y_pred_irt))
rmse_ml = np.sqrt(mean_squared_error(y_test, y_pred_ml))

print("\n" + "="*50)
print("🏆 ADVANCED PERFORMANCE COMPARISON (RMSE)")
print(f"📊 Pure IRT Model Error: {rmse_irt:.4f}")
print(f"🤖 Tuned Gradient Boosting Error: {rmse_ml:.4f}")
print("="*50)

if rmse_ml < rmse_irt:
    print("🎉 BOOM! The Tuned ML model successfully beat the IRT model!")
    print("আমাদের হাইব্রিড মেকানিজম কাজ করেছে! 🚀")
else:
    print("🤔 IRT is still holding strong. Mathematical baseline is very tough to beat with small data.")