import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from girth import twopl_mml

print("⚙️ Loading Data...")
# ১. ডাটা লোড করা
df = pd.read_csv("omr_synthetic_data.csv")

# Girth লাইব্রেরির জন্য ডাটাকে (Questions x Students) ফরম্যাটে সাজানো
questions_cols = [f"Q{i+1}" for i in range(10)]
responses = df[questions_cols].values.T 

print("🧠 Running IRT Model (2-Parameter Logistic)... Please wait...")
# ২. IRT 2PL মডেল রান করানো (এখানে মেশিন প্রশ্ন এবং স্টুডেন্ট উভয়কেই জাজ করবে)
results = twopl_mml(responses)

# ৩. মেশিনের বের করা রেজাল্টগুলো আলাদা করা
difficulties = results['Difficulty']
discriminations = results['Discrimination']
abilities = results['Ability']

# ৪. ডাটাফ্রেমে মেশিনের প্রেডিক্ট করা Ability যোগ করা
df['Estimated_Ability'] = abilities

print("\n" + "="*50)
print("📊 Question Analysis (First 5 Questions):")
for i in range(5):
    print(f"Q{i+1} -> Difficulty: {difficulties[i]:.2f} | Discrimination (a): {discriminations[i]:.2f}")
print("="*50)

print("\n🧑‍🎓 Student Analysis (First 5 Students):")
# Raw Score (সাধারণ নম্বর) এর সাথে মেশিনের বের করা মেধার (Ability) তুলনা
print(df[['StudentID', 'Total_Raw_Score', 'True_Ability_Theta', 'Estimated_Ability']].head())
print("="*50)

# ৫. Item Characteristic Curve (ICC) বা প্রশ্নের গ্রাফ তৈরি করা (Question 1 এর জন্য)
print("\n📈 Generating Item Characteristic Curve (ICC) for Question 1...")
theta_range = np.linspace(-3, 3, 100)
a = discriminations[0]
b = difficulties[0]

# লজিস্টিক ইকুয়েশন
p_correct = 1 / (1 + np.exp(-a * (theta_range - b)))

plt.figure(figsize=(8, 5))
plt.plot(theta_range, p_correct, label=f'Q1 (Difficulty: {b:.2f})', color='blue', linewidth=2)
plt.axvline(x=b, color='red', linestyle='--', label=f'Difficulty Level = {b:.2f}')
plt.axhline(y=0.5, color='gray', linestyle=':')
plt.title('Item Characteristic Curve (ICC) - Question 1')
plt.xlabel('Student Ability (Theta)')
plt.ylabel('Probability of Correct Answer (0 to 1)')
plt.legend()
plt.grid(True, alpha=0.3)

# গ্রাফটি ছবি হিসেবে সেভ করা
plt.savefig('icc_plot_q1.png', dpi=300)
print("✅ ICC plot saved successfully as 'icc_plot_q1.png'!")
df.to_csv("omr_synthetic_data.csv", index=False)
print("✅ Updated dataset saved with Estimated_Ability!")