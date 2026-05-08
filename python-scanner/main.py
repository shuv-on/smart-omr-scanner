from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 ডেমো Answer Key এবং IRT Difficulty Parameters (রিসার্চ থেকে প্রাপ্ত)
# বাস্তবে এগুলো আপনার ডাটাবেস থেকে আসবে category অনুযায়ী
ANSWER_KEY = {
    1: "A", 2: "B", 3: "C", 4: "D", 5: "A",
    6: "B", 7: "C", 8: "D", 9: "A", 10: "B"
}

# IRT Difficulty (b): নেগেটিভ মানে সহজ, পজিটিভ মানে কঠিন
IRT_DIFFICULTY = {
    1: -1.5, 2: -0.5, 3: 0.0,  4: 0.5,  5: 1.2,  # Q5 সবচেয়ে কঠিন
    6: -1.0, 7: 0.2,  8: 1.5,  9: -0.8, 10: 0.8  # Q8 সবচেয়ে কঠিন
}

def process_omr_actual(image_bytes, category):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    target_width = 800
    ratio = target_width / image.shape[1]
    target_height = int(image.shape[0] * ratio)
    resized_image = cv2.resize(image, (target_width, target_height))

    output_image = resized_image.copy()
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_bubbles = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        ar = w / float(h)
        
        if area > 1000 and 0.6 <= ar <= 1.5 and w < 120 and h < 120:
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            
            if total / float(area) > 0.35:
                raw_bubbles.append((x, y, w, h, c))

    pure_bubbles = []
    for (x, y, w, h, c) in raw_bubbles:
        cx = x + w // 2
        cx_ratio = cx / target_width
        if 0.25 < cx_ratio < 0.95:
            pure_bubbles.append((x, y, w, h, c))

    filled_bubbles = sorted(pure_bubbles, key=lambda b: b[1])

    detected_answers = {}
    category_map = {"science": "sci", "bangla": "ban", "ict": "ict", "gk": "gk"}
    prefix = category_map.get(category, "ict")

    for i in range(1, 11):
        detected_answers[f"{prefix}_{i}"] = "NOT_ANSWERED"

    for idx, (x, y, w, h, c) in enumerate(filled_bubbles):
        if idx >= 10: break 
        
        cx = x + w // 2
        question_num = idx + 1
        q_id = f"{prefix}_{question_num}"
        cx_ratio = cx / target_width
        
        if cx_ratio < 0.42:   ans = "A"
        elif cx_ratio < 0.60: ans = "B"
        elif cx_ratio < 0.78: ans = "C"
        else:                 ans = "D"

        detected_answers[q_id] = ans
        
        # ছবিতে চিহ্নিত করা
        cv2.drawContours(output_image, [c], -1, (0, 255, 0), 3)
        cv2.putText(output_image, f"Q{question_num}:{ans}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    _, buffer = cv2.imencode('.jpg', output_image)
    processed_image_b64 = base64.b64encode(buffer).decode('utf-8')

    return detected_answers, len(filled_bubbles), f"data:image/jpeg;base64,{processed_image_b64}", prefix

@app.post("/api/scan-omr")
async def scan_omr(omrImage: UploadFile = File(...), category: str = Form(...)):
    try:
        contents = await omrImage.read()
        answers, count, img_data, prefix = process_omr_actual(contents, category)
        
        # 🚀 IRT EVALUATION START
        raw_score = 0
        irt_weighted_score = 0
        evaluation_details = {}

        for i in range(1, 11):
            q_id = f"{prefix}_{i}"
            student_ans = answers.get(q_id, "NOT_ANSWERED")
            correct_ans = ANSWER_KEY.get(i)
            difficulty = IRT_DIFFICULTY.get(i, 0)
            
            is_correct = (student_ans == correct_ans)
            
            if is_correct:
                raw_score += 1
                # কঠিন প্রশ্নের জন্য বেশি পয়েন্ট (Base 1 + Difficulty)
                # সহজ প্রশ্নের জন্য পয়েন্ট তুলনামূলক কম
                irt_weighted_score += (1 + difficulty)
                
            evaluation_details[q_id] = {
                "student_answer": student_ans,
                "correct_answer": correct_ans,
                "is_correct": is_correct,
                "difficulty_level": difficulty
            }
            
        # মেধা স্কেলিং (Theta proxy mapping)
        estimated_ability = round(irt_weighted_score, 2)
        
        return {
            "status": "success",
            "category": category,
            "totalBubblesDetected": count,
            "performance": {
                "totalQuestions": 10,
                "rawScore": raw_score,               # সাধারণ নম্বর (যেমন: ১০ এর মধ্যে ৭)
                "irtAbilityScore": estimated_ability # IRT মেধা স্কোর (কঠিন প্রশ্ন পারলে বেশি হবে)
            },
            "evaluationDetails": evaluation_details, # ফ্রন্টএন্ডে কোন প্রশ্ন ভুল/সঠিক দেখানোর জন্য
            "detectedAnswers": answers,
            "processedImage": img_data
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}