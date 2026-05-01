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

def process_omr_actual(image_bytes, category):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # ১. ইমেজ রিসাইজ (স্ট্যান্ডার্ড ৮০০ পিক্সেল উইডথ)
    target_width = 800
    ratio = target_width / image.shape[1]
    target_height = int(image.shape[0] * ratio)
    resized_image = cv2.resize(image, (target_width, target_height))

    output_image = resized_image.copy()
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # ২. থ্রেশহোল্ড (কালো বাবলগুলোকে সাদা করবে)
    # আপনার ছবির কন্ট্রাস্ট অনুযায়ী ১৫০ মানটি একদম পারফেক্ট
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # ৩. কন্টুর খুঁজে বের করা
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_bubbles = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        ar = w / float(h)
        
        # 🛡️ ভুয়া বর্ডার বা নয়েজ ফিল্টার করার জন্য শর্ত
        # গোল্লাগুলো সাধারণত ২০০০-১৫০০০ এরিয়ার হয় এবং উইডথ ১০০-এর নিচে থাকে
        if area > 1000 and 0.6 <= ar <= 1.5 and w < 120 and h < 120:
            # চেক করা: গোল্লাটা কি আসলেই ভরাট?
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            
            fill_ratio = total / float(area)
            
            # যদি ৩০% এর বেশি কালি থাকে, তবেই বাবল ধরবে
            if fill_ratio > 0.35:
                raw_bubbles.append((x, y, w, h, c))

    # ৪. পজিশন ফিল্টার: বামের নম্বর এবং ডানের বর্ডার বাদ দেওয়া
    # আমরা শুধু মাঝখানের ০.২৫ থেকে ০.৯২ অংশটুকু নিব
    pure_bubbles = []
    for (x, y, w, h, c) in raw_bubbles:
        cx = x + w // 2
        cx_ratio = cx / target_width
        if 0.25 < cx_ratio < 0.95:
            pure_bubbles.append((x, y, w, h, c))

    # ৫. ওপর থেকে নিচে সিরিয়াল করা
    filled_bubbles = sorted(pure_bubbles, key=lambda b: b[1])

    detected_answers = {}
    prefix = category if category else "ict"
    # ক্যাটাগরি অনুযায়ী আইডি সেট করা
    category_map = {"science": "sci", "bangla": "ban", "ict": "ict"}
    prefix = category_map.get(category, "ict")

    # শুরুতে সব NOT_ANSWERED সেট করা
    for i in range(1, 6):
        detected_answers[f"{prefix}_{i}"] = "NOT_ANSWERED"

    print("\n" + "="*50)
    print(f"🎯 SCANNER LOG: Found {len(filled_bubbles)} Potential Bubbles")

    for idx, (x, y, w, h, c) in enumerate(filled_bubbles):
        if idx >= 5: break # শুধু প্রথম ৫টা প্রশ্ন নিবে
        
        cx = x + w // 2
        question_num = idx + 1
        q_id = f"{prefix}_{question_num}"
        cx_ratio = cx / target_width
        
        # 📐 আপনার দেওয়া রেশিও অনুযায়ী উত্তর বের করা
        if cx_ratio < 0.42:   ans = "A"
        elif cx_ratio < 0.60: ans = "B"
        elif cx_ratio < 0.78: ans = "C"
        else:                 ans = "D"

        detected_answers[q_id] = ans
        print(f"✅ Q{question_num}: {ans} (Ratio: {cx_ratio:.3f})")

        # ছবিতে চিহ্নিত করা
        cv2.drawContours(output_image, [c], -1, (0, 255, 0), 3)
        cv2.putText(output_image, f"Q{idx+1}:{ans}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    print("="*50 + "\n")

    # ৬. ইমেজ এনকোড করে ফ্রন্টএন্ডে পাঠানো
    _, buffer = cv2.imencode('.jpg', output_image)
    processed_image_b64 = base64.b64encode(buffer).decode('utf-8')

    return detected_answers, len(filled_bubbles), f"data:image/jpeg;base64,{processed_image_b64}"

@app.post("/api/scan-omr")
async def scan_omr(omrImage: UploadFile = File(...), category: str = Form(...)):
    try:
        contents = await omrImage.read()
        answers, count, img_data = process_omr_actual(contents, category)
        return {
            "status": "success",
            "category": category,
            "totalBubblesDetected": count, 
            "detectedAnswers": answers,
            "processedImage": img_data
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}