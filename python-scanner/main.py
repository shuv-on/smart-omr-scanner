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

# 🧠 ডেমো Answer Key এবং IRT Difficulty Parameters
ANSWER_KEY = {
    1: "A", 2: "B", 3: "C", 4: "D", 5: "A",
    6: "B", 7: "C", 8: "D", 9: "A", 10: "B"
}

# IRT Difficulty (b): নেগেটিভ মানে সহজ, পজিটিভ মানে কঠিন
IRT_DIFFICULTY = {
    1: -1.5, 2: -0.5, 3: 0.0,  4: 0.5,  5: 1.2, 
    6: -1.0, 7: 0.2,  8: 1.5,  9: -0.8, 10: 0.8 
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
    
    # 🚀 ইমেজ নয়েজ দূর করতে GaussianBlur যুক্ত করা হলো
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_bubbles = []
    
    # 🚀 স্টেপ ১: খালি বা ভরাট, সব বাবল খুঁজে বের করা
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        ar = w / float(h)
        
        # সাইজ এবং অনুপাত দিয়ে বৃত্ত শনাক্ত করা
        if area > 800 and 0.6 <= ar <= 1.5 and w < 120 and h < 120:
            cx = x + w // 2
            cx_ratio = cx / target_width
            if 0.25 < cx_ratio < 0.95:
                all_bubbles.append((x, y, w, h, c))

    # ওপর থেকে নিচে সাজানো
    all_bubbles = sorted(all_bubbles, key=lambda b: b[1])

    # 🚀 স্টেপ ২: বাবলগুলোকে সারিতে (Rows) গ্রুপ করা
    rows = []
    current_row = []
    for b in all_bubbles:
        if not current_row:
            current_row.append(b)
        else:
            last_y = current_row[-1][1]
            # যদি Y-কোঅর্ডিনেটের পার্থক্য ৩০ পিক্সেলের কম হয়, তারমানে এরা একই লাইনে (প্রশ্নে) আছে
            if abs(b[1] - last_y) < 30:
                current_row.append(b)
            else:
                rows.append(current_row)
                current_row = [b]
    if current_row:
        rows.append(current_row)

    detected_answers = {}
    category_map = {"science": "sci", "bangla": "ban", "ict": "ict", "gk": "gk"}
    prefix = category_map.get(category, "ict")

    for i in range(1, 11):
        detected_answers[f"{prefix}_{i}"] = "NOT_ANSWERED"

    filled_count = 0

    # 🚀 স্টেপ ৩: সারি ধরে ধরে ভরাট বাবল চেক করা
    for idx, row in enumerate(rows):
        if idx >= 10: break 
        
        question_num = idx + 1
        q_id = f"{prefix}_{question_num}"
        
        # একই সারির বাবলগুলোকে বাম থেকে ডানে সাজানো (A, B, C, D)
        row = sorted(row, key=lambda b: b[0])
        
        best_ans = "NOT_ANSWERED"
        max_filled = 0
        best_bubble = None
        
        for (x, y, w, h, c) in row:
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            
            fill_ratio = total / float(cv2.contourArea(c))
            
            # যদি ৩৫% এর বেশি ভরাট থাকে এবং অন্যান্য অপশনের চেয়ে বেশি কালো হয়
            if fill_ratio > 0.35 and fill_ratio > max_filled:
                max_filled = fill_ratio
                best_bubble = (x, y, w, h, c)
                
                # A, B, C, D নির্ধারণ করা
                cx = x + w // 2
                cx_ratio = cx / target_width
                if cx_ratio < 0.42:   best_ans = "A"
                elif cx_ratio < 0.60: best_ans = "B"
                elif cx_ratio < 0.78: best_ans = "C"
                else:                 best_ans = "D"

        detected_answers[q_id] = best_ans
        
        # ছবিতে মার্ক করা
        if best_ans != "NOT_ANSWERED" and best_bubble is not None:
            filled_count += 1
            bx, by, bw, bh, bc = best_bubble
            cv2.drawContours(output_image, [bc], -1, (0, 255, 0), 3)
            cv2.putText(output_image, f"Q{question_num}:{best_ans}", (bx, by - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    _, buffer = cv2.imencode('.jpg', output_image)
    processed_image_b64 = base64.b64encode(buffer).decode('utf-8')

    return detected_answers, filled_count, f"data:image/jpeg;base64,{processed_image_b64}", prefix

@app.post("/api/scan-omr")
async def scan_omr(omrImage: UploadFile = File(...), category: str = Form(...)):
    try:
        contents = await omrImage.read()
        answers, count, img_data, prefix = process_omr_actual(contents, category)
        
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
                irt_weighted_score += (1 + difficulty)
                
            evaluation_details[q_id] = {
                "student_answer": student_ans,
                "correct_answer": correct_ans,
                "is_correct": is_correct,
                "difficulty_level": difficulty
            }
            
        estimated_ability = round(irt_weighted_score, 2)
        
        return {
            "status": "success",
            "category": category,
            "totalBubblesDetected": count,
            "performance": {
                "totalQuestions": 10,
                "rawScore": raw_score,               
                "irtAbilityScore": estimated_ability 
            },
            "evaluationDetails": evaluation_details, 
            "detectedAnswers": answers,
            "processedImage": img_data
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}