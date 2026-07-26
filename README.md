## Demo Link
[Watch Demo Video](https://youtu.be/6P73GazZVs4)
# Smart OMR Project

An intelligent, automated **Optical Mark Recognition (OMR)** examination system that uses Computer Vision and AI-driven Item Response Theory (IRT) to scan, evaluate, and grade multiple-choice answer sheets — instantly.

The project is a **full-stack, multi-service** application:

| Service | Technology | Port | Purpose |
|--------|-----------|------|---------|
| **Frontend** | Next.js 16 (React 19) | `3000` | Student UI (Login / Register / Dashboard / Exam / Scanner) |
| **Backend** | Spring Boot 3 (Java 17) | `8080` | Auth, JWT, Business logic, bridges UI to scanner |
| **Scanner** | Python FastAPI + OpenCV | `8000` | OMR image processing & bubble detection |
| **ML Research** | Python (scikit-learn, girth) | — | Synthetic data generation, IRT 2PL/3PL, hybrid ML-IRT |

> **Repositery name (বাংলা):** অটোমেটিক ওএমআর পরীক্ষা গ্রেডিং সিস্টেম — যেখানে কম্পিউটার ভিশন দিয়ে OMR শিট স্ক্যান করে সঠিক/ভুল উত্তর শনাক্ত করা হয় এবং IRT মডেল দিয়ে শিক্ষার্থীর প্রকৃত মেধা (Ability) পরিমাপ করা হয়।

---

## Table of Contents

1. [Features](#features)
2. [Project Architecture](#project-architecture)
3. [Tech Stack](#tech-stack)
4. [Folder Structure](#folder-structure)
5. [Prerequisites](#prerequisites)
6. [Installation & Setup](#installation--setup)
   - [1. Database Setup (MySQL)](#1-database-setup-mysql)
   - [2. Backend Setup (Spring Boot)](#2-backend-setup-spring-boot)
   - [3. Scanner Setup (Python FastAPI)](#3-scanner-setup-python-fastapi)
   - [4. Frontend Setup (Next.js)](#4-frontend-setup-nextjs)
   - [5. (Optional) ML Research](#5-optional-ml-research)
7. [How to Run the Full Project](#how-to-run-the-full-project)
8. [Environment Variables](#environment-variables)
9. [API Endpoints](#api-endpoints)
10. [How It Works](#how-it-works)
11. [Troubleshooting](#troubleshooting)
12. [License](#license)

---

## Features

### Frontend (Next.js)
- **Landing Page** — Modern hero landing page with auto-redirect for logged-in users.
- **Authentication** — Secure Sign Up / Sign In forms with toast notifications.
- **Dashboard** — Real-time stats: Total Exams, Scanned OMRs, Average Raw Score, Average IRT Ability Score, and an exam history table (status: green/yellow/red).
- **Exam Module** — Two modes:
  - **Online Exam** — Timer-based MCQ test (10 minutes), selectable subject.
  - **Scan Mode** — Upload a paper OMR image to get AI-analyzed answers.
- **Scanner Module** — Subject-specific OMR upload (ICT, Science, GK, Bangla) with live animated scanning.
- **Results View** — Score breakdown (correct/wrong/penalty), IRT Ability Score, AI-annotated OMR image (green boxes & labels), digital answer-sheet visualization (green = correct, red = wrong, blue = skipped correct).
- **Persistent State** — Exam history is saved in `localStorage`.

### Backend (Spring Boot)
- **JWT-based Authentication** — Register & Login with BCrypt-hashed passwords and JWT tokens (jjwt 0.11.5).
- **CORS Configuration** — Pre-configured for `localhost:3000` with credentials.
- **Stateless Session** — JWT filter integrated with Spring Security.
- **Bridge Service** — Forwards uploaded OMR images to the Python scanner via `RestTemplate`.
- **User Entity** — Persisted in MySQL via Spring Data JPA (`ddl-auto: update`).
- **Password Encoder** — BCrypt hashing.
- **`.env` Support** — Loads secrets via `spring-dotenv`.

### Scanner (Python FastAPI + OpenCV)
- **Image Pre-processing** — GaussianBlur, Resize (800px width), grayscale conversion.
- **Adaptive Thresholding** — Binary inverse threshold (150/255).
- **Contour Detection** — Identifies filled & empty bubbles (area > 800, aspect ratio 0.6–1.5).
- **Row Grouping** — Groups bubbles into rows of 4 (A, B, C, D options).
- **Fill Ratio Analysis** — A bubble is "marked" if filled > 35%.
- **Visual Output** — Returns base64-encoded annotated image with detected answers highlighted in green.
- **Multi-subject Support** — Prefix mapping: `science → sci`, `bangla → ban`, `ict → ict`, `gk → gk`.
- **Grading Engine** — Calculates raw score, negative marking (0.25 per wrong), and weighted IRT ability score.

### ML Research
- **Synthetic Data Generator** — Creates 500-student × 10-question dataset using **IRT 3PL** model (`generate_data.py`).
- **Pure IRT 2PL Analyzer** — Fits Item Response Theory model using `girth.twopl_mml`, computes Difficulty & Discrimination, plots ICC (`analyze_irt.py`).
- **Hybrid ML-IRT Model** — Random Forest Regressor combining Q-answers + Raw Score + IRT-predicted Ability, compares RMSE vs. pure IRT (`hybrid_ml_model.py`).
- **Advanced ML Model** — Gradient Boosting Regressor with `GridSearchCV` hyperparameter tuning (`advanced_ml.py`).
- **Visualizations** — `icc_plot_q1.png`, `ml_vs_irt_comparison.png` — research-grade plots saved at 300 DPI.

---

## Project Architecture

```
                ┌──────────────────────┐
                │   Browser (User)     │
                │  http://localhost:    │
                │       3000           │
                └──────────┬───────────┘
                           │  (Axios / Fetch)
                           ▼
        ┌────────────────────────────────────────┐
        │  Spring Boot Backend (Java 17)         │
        │   - Auth (JWT, BCrypt)                 │
        │   - User Entity (MySQL)                │
        │   - OMR Controller (proxy)             │
        │  http://localhost:8080                 │
        └──────────┬─────────────────────────────┘
                   │  RestTemplate (multipart)
                   ▼
        ┌────────────────────────────────────────┐
        │  Python FastAPI Scanner (OpenCV)       │
        │   - Bubble Detection                   │
        │   - Grading + IRT Weighting            │
        │  http://localhost:8000                 │
        └────────────────────────────────────────┘

        ┌────────────────────────────────────────┐
        │  ML Research (offline analysis)        │
        │   - IRT 2PL/3PL                        │
        │   - Random Forest / Gradient Boosting  │
        └────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 16, React 19, Tailwind CSS 4, Axios, Lucide Icons, React Hot Toast |
| Backend | Spring Boot 3.2.5, Spring Security, Spring Data JPA, jjwt 0.11.5, Lombok, MySQL Connector |
| Scanner | FastAPI, OpenCV, NumPy, Pillow |
| ML | Python 3, scikit-learn, pandas, NumPy, matplotlib, girth |
| Database | MySQL (or any JPA-compatible DB) |
| Auth | JWT (HS256), BCrypt |
| Build Tools | Maven (backend), npm/pnpm (frontend), pip/venv (python) |

---

## Folder Structure

```
smart-omr-project/
├── frontend/                 # Next.js client app
│   ├── src/
│   │   ├── app/              # Routes (/, /login, /register, /dashboard, /exam, /scanner)
│   │   ├── components/       # Navbar.js
│   │   └── data/             # questions.json (4 categories × 10 Qs)
│   ├── package.json
│   └── tailwind.config.js
│
├── smart-omr-backend/        # Spring Boot server
│   ├── src/main/java/com/smartomr/backend/
│   │   ├── config/           # SecurityConfig, CorsConfig, JwtAuthenticationFilter
│   │   ├── controller/       # AuthController, OmrController
│   │   ├── dto/              # RegisterRequest, LoginRequest, AuthResponse
│   │   ├── entity/           # User
│   │   ├── repository/       # UserRepository
│   │   ├── service/          # AuthService, OmrService
│   │   ├── utils/            # JwtUtil
│   │   └── SmartOmrApplication.java
│   ├── src/main/resources/application.yml
│   ├── .env                  # DB creds + JWT secret
│   └── pom.xml
│
├── python-scanner/           # FastAPI OMR service
│   ├── main.py               # /api/scan-omr endpoint
│   ├── debug_omr.jpg         # Sample debug image
│   ├── debug_thresh.jpg      # Threshold debug
│   └── venv/ or .venv/
│
├── ml-research/              # IRT & ML experiments
│   ├── generate_data.py      # Synthetic data generator
│   ├── analyze_irt.py        # IRT 2PL analyzer
│   ├── hybrid_ml_model.py    # Random Forest
│   ├── advanced_ml.py        # Gradient Boosting + GridSearchCV
│   ├── omr_synthetic_data.csv
│   ├── icc_plot_q1.png       # Generated ICC plot
│   └── ml_vs_irt_comparison.png
│
└── science.jpeg              # Reference image
```

---

## Prerequisites

Make sure the following are installed:

| Tool | Minimum Version | Check Command |
|------|-----------------|---------------|
| **Node.js** | 18+ | `node -v` |
| **npm / pnpm** | latest | `npm -v` |
| **Java JDK** | 17 | `java -version` |
| **Maven** | 3.8+ | `mvn -version` |
| **Python** | 3.10+ | `python --version` |
| **MySQL** | 8.0+ | `mysql --version` |
| **pip** | 23+ | `pip --version` |

> **For ML research only (optional):** `pip install pandas numpy scikit-learn matplotlib girth`

---

## Installation & Setup

### 1. Database Setup (MySQL)

Start your MySQL server, then create the database:

```sql
CREATE DATABASE smart_omr;
```

The Spring Boot app will auto-create the `users` table on first run (`ddl-auto: update`).

### 2. Backend Setup (Spring Boot)

```bash
cd smart-omr-backend

# (Optional) Check / edit credentials:
cat .env
# DB_USERNAME=root
# DB_PASSWORD=your_mysql_password
# JWT_SECRET=your_hex_secret
# JWT_EXPIRATION=86400000

# Run with Maven
./mvnw spring-boot:run
# or
mvn spring-boot:run
```

**Backend will start on:** `http://localhost:8080`

### 3. Scanner Setup (Python FastAPI)

```bash
cd python-scanner

# Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# .\venv\Scripts\Activate.ps1    # Windows PowerShell

# Install dependencies
pip install fastapi uvicorn opencv-python numpy python-multipart

# Run the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Scanner will start on:** `http://localhost:8000`

### 4. Frontend Setup (Next.js)

```bash
cd frontend

# Install dependencies
npm install
# or
pnpm install
# or
yarn install

# Run development server
npm run dev
# or
pnpm dev
```

**Frontend will start on:** `http://localhost:3000`

### 5. (Optional) ML Research

```bash
cd ml-research

# Activate venv (or create new one)
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install pandas numpy scikit-learn matplotlib girth

# 1) Generate synthetic OMR data (creates omr_synthetic_data.csv)
python generate_data.py

# 2) Fit IRT 2PL model (produces icc_plot_q1.png)
python analyze_irt.py

# 3) Train hybrid ML-IRT (Random Forest) — produces ml_vs_irt_comparison.png
python hybrid_ml_model.py

# 4) Train advanced Gradient Boosting model with hyperparameter tuning
python advanced_ml.py
```

---

## How to Run the Full Project

Open **4 terminals** in parallel and run each service:

| Terminal | Directory | Command | URL |
|----------|-----------|---------|-----|
| 1 | `smart-omr-backend` | `mvn spring-boot:run` | http://localhost:8080 |
| 2 | `python-scanner` | `uvicorn main:app --host 0.0.0.0 --port 8000 --reload` | http://localhost:8000 |
| 3 | `frontend` | `npm run dev` | http://localhost:3000 |
| 4 (optional) | `ml-research` | any `python *.py` script | — |

**Then:**

1. Open `http://localhost:3000` in your browser.
2. Click **Create Account** → fill the form → submit.
3. You'll be redirected to **Login** → sign in → land on **Dashboard**.
4. Try the **Scanner** tab → pick a subject → upload an OMR image → see AI analysis.
5. Or try the **Exam** tab → take an online timed MCQ test.

> **Note:** In some flows the frontend calls the FastAPI scanner **directly** (port `8000`). Make sure `python-scanner` is also running, otherwise scanning will fail with a connection error.

---

## Environment Variables

Create/edit `smart-omr-backend/.env`:

```env
DB_USERNAME=root
DB_PASSWORD=your_mysql_password
JWT_SECRET=your_long_random_hex_string_here
JWT_EXPIRATION=86400000
```

The `application.yml` reads these via `${DB_USERNAME}`, `${DB_PASSWORD}`, `${JWT_SECRET}`, `${JWT_EXPIRATION}`.

---

## API Endpoints

### Auth (Spring Boot · `:8080`)

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/api/auth/register` | `{ name, email, password }` | Create new user, returns JWT |
| `POST` | `/api/auth/login` | `{ email, password }` | Authenticate user, returns JWT |

### OMR Proxy (Spring Boot · `:8080`)

| Method | Endpoint | Params | Description |
|--------|----------|--------|-------------|
| `POST` | `/api/omr/scan` | `omrImage` (multipart), `category` (string) | Forwards upload to Python scanner |

### Scanner (FastAPI · `:8000`)

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/api/scan-omr` | `omrImage` (file), `category` (form) | Detect bubbles, grade, return JSON + base64 image |

**Sample response (`/api/scan-omr`):**

```json
{
  "status": "success",
  "category": "ict",
  "totalBubblesDetected": 10,
  "performance": {
    "totalQuestions": 10,
    "rawScore": 7,
    "irtAbilityScore": 4.32
  },
  "evaluationDetails": {
    "ict_1": { "student_answer": "C", "correct_answer": "C", "is_correct": true, "difficulty_level": 0.0 },
    "ict_2": { "student_answer": "A", "correct_answer": "A", "is_correct": true, "difficulty_level": -0.5 }
  },
  "detectedAnswers": { "ict_1": "C", "ict_2": "A", ... },
  "processedImage": "data:image/jpeg;base64,..."
}
```

---

## How It Works

### Step 1 — Upload
The user uploads a photo of a filled OMR answer sheet through the Next.js frontend.

### Step 2 — Spring Boot Proxy
The Next.js app sends the file as `multipart/form-data` to the Spring Boot backend (`/api/omr/scan`), which forwards it to the Python scanner using `RestTemplate`.

### Step 3 — Image Processing (Python)
- Resize image (width = 800px, height auto).
- Convert to grayscale → `GaussianBlur` → `THRESH_BINARY_INV` (150/255).
- Find external contours → filter for bubble candidates (area > 800, aspect ratio 0.6–1.5).
- Sort top-to-bottom, group into rows by Y-coordinate (tolerance 30 px).
- Within each row, compute **fill ratio** = filled pixels / bubble area.
- The bubble with fill ratio > 35% AND the highest fill in its row wins.
- Bubble's center X-coordinate → mapped to A / B / C / D.

### Step 4 — Grading
- For each question, compare detected answer vs. answer key.
- **Raw Score** = count of correct answers.
- **Penalty** = wrong × 0.25.
- **Final Score** = correct − penalty.
- **IRT Weighted Ability** = Σ `(correct ? 1 + difficulty_i : 0)`.

### Step 5 — Frontend Display
- Summary panel (Score / Correct / Wrong / Total).
- **AI-Predicted IRT Ability Score** (positive = above average, negative = below).
- Toggleable **Digital Answer Sheet** (green/red/blue bubbles).
- **Annotated OMR image** returned as base64 — green boxes highlight detected bubbles with labels like `Q1:C`.

### Step 6 — Persistence
- Exam results are stored in `localStorage` → rendered on the Dashboard.
- User credentials are stored in **MySQL** with BCrypt-hashed passwords and JWT tokens for stateless auth.

### ML Research Pipeline
1. **Generate synthetic data** based on the 3PL IRT model (500 students, 10 questions).
2. **Fit pure IRT 2PL** using `girth` to estimate per-student ability and per-question difficulty/discrimination.
3. **Train hybrid Random Forest** combining ML features + IRT ability.
4. **Compare RMSE** of pure IRT vs. hybrid ML — generate comparison plot.
5. **Advanced:** Tune a Gradient Boosting Regressor via `GridSearchCV` (5-fold CV).

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| **CORS error in browser** | Ensure `python-scanner` is running on port `8000` and that the FastAPI middleware `allow_origins=["*"]` is active. Backend also pre-configures CORS for `localhost:3000`. |
| **`Connection failed! Python FastAPI server (main.py) is running?`** | Start the scanner: `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`. |
| **Backend can't connect to MySQL** | Make sure MySQL is running, the DB `smart_omr` exists, and `.env` credentials match. |
| **`401 Unauthorized` on `/api/omr/scan`** | The endpoint is publicly permitted via `SecurityConfig`; restart backend after changes. |
| **Empty / wrong detection** | Improve photo quality: flat lighting, no skew, full sheet in frame, dark fill inside bubbles. |
| **Hydration error in Next.js** | `Navbar.js` and other components use the `isClient` state check — already handled. |
| **`JWT secret too short`** | Replace with a long random hex string (≥ 32 bytes). Use `openssl rand -hex 32`. |
| **girth install fails** | `pip install girth` may need `numpy` first; on Apple Silicon try `conda install -c conda-forge girth`. |

---

## Future Roadmap (Suggestions)

- Replace simple contour detection with a deep-learning bubble detector (YOLO / UNet).
- Multi-page OMR sheets, QR-coded student ID.
- Admin panel for managing questions, answer keys, and difficulty parameters.
- Cloud storage for OMR images and full server-side exam history.
- Export results to PDF/CSV.
- Replace local JWT with OAuth2 / SSO.
- Deploy with Docker Compose.

---

## License

This project is provided **as-is for educational and research purposes**.

---

> **Made with ❤️** — combining Computer Vision, Item Response Theory, and Modern Web Tech to make exam grading fast, accurate, and intelligent.
