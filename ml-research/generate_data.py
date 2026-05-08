import numpy as np
import pandas as pd

# reproducibility এর জন্য seed সেট করা
np.random.seed(42)

# কনফিগারেশন
NUM_STUDENTS = 500
NUM_QUESTIONS = 10

# ১. স্টুডেন্টদের মেধা (Theta) তৈরি করা (Normal Distribution: Mean=0, Std=1)
# পজিটিভ মানে ভালো স্টুডেন্ট, নেগেটিভ মানে দুর্বল
theta = np.random.normal(0, 1, NUM_STUDENTS)

# ২. প্রশ্নের প্যারামিটার সেট করা
# a = Discrimination (প্রশ্নটি ভালো-খারাপ স্টুডেন্ট আলাদা করতে পারে কিনা, সাধারণত 0.5 থেকে 2.0)
a = np.random.uniform(0.5, 2.0, NUM_QUESTIONS)
# b = Difficulty (প্রশ্নের কাঠিন্য, -2 মানে খুব সহজ, +2 মানে খুব কঠিন)
b = np.random.normal(0, 1, NUM_QUESTIONS)
# c = Guessing parameter (আন্দাজে সঠিক হওয়ার সম্ভাবনা, যেহেতু ৪টি অপশন তাই ০.২৫ এর কাছাকাছি)
c = np.random.uniform(0.15, 0.25, NUM_QUESTIONS)

# ৩. রেসপন্স ম্যাট্রিক্স তৈরি করা (0 = ভুল, 1 = সঠিক)
response_matrix = np.zeros((NUM_STUDENTS, NUM_QUESTIONS), dtype=int)

print("⚙️ Generating Synthetic OMR Data using 3PL IRT Model...\n")

for i in range(NUM_STUDENTS):
    for j in range(NUM_QUESTIONS):
        # IRT 3PL গাণিতিক সূত্র:
        # P(correct) = c + (1 - c) / (1 + exp(-a * (theta - b)))
        prob_correct = c[j] + (1 - c[j]) / (1 + np.exp(-a[j] * (theta[i] - b[j])))
        
        # সম্ভাবনার উপর ভিত্তি করে 0 বা 1 বসানো
        random_chance = np.random.uniform(0, 1)
        if random_chance < prob_correct:
            response_matrix[i, j] = 1
        else:
            response_matrix[i, j] = 0

# ৪. ডাটাফ্রেম তৈরি ও সেভ করা
columns = [f"Q{j+1}" for j in range(NUM_QUESTIONS)]
df = pd.DataFrame(response_matrix, columns=columns)

# স্টুডেন্ট আইডি এবং আসল মেধা (True Theta) যোগ করা (পরে ML মডেল টেস্ট করার জন্য লাগবে)
df.insert(0, "StudentID", [f"S_{i+1}" for i in range(NUM_STUDENTS)])
df["True_Ability_Theta"] = theta
df["Total_Raw_Score"] = df[columns].sum(axis=1)

# CSV ফাইলে সেভ
df.to_csv("omr_synthetic_data.csv", index=False)

print("✅ Data Generation Complete!")
print(f"📌 Total Students: {NUM_STUDENTS}, Total Questions: {NUM_QUESTIONS}")
print("📊 First 5 rows of the dataset:")
print(df.head())