# **Sentiment Analysis API**

*A complete end-to-end project plan, architecture, and weekly roadmap*

---

## **1. Overview**

This project builds a fully functional, production-grade **Sentiment Analysis API** that:

- Performs **real-time sentiment analysis** (positive / negative / neutral)
- Automatically stores **new weak/low-confidence samples**
- Collects **user feedback** for corrections
- Retrains itself **every week for free** (GitHub Actions)
- Uses **free resources only**
- Deploys a FastAPI server on **HuggingFace Spaces** (always free CPU)
- Stores datasets + models using **HuggingFace Model Hub & Datasets**
- Supports optional **LLM-assisted self-labeling** (NVIDIA NIM free-tier)

The system continuously improves accuracy without spending money on cloud infra.

---

# **2. High-Level Architecture (Zero-Cost)**

```
                ┌────────────────────────────┐
                │   HF Space (FastAPI API)   │
                │  - Real-time inference     │
                │  - Stores low-confidence   │
                │  - Stores user feedback    │
                └──────────────┬─────────────┘
                               │
                       push new feedback
                               │
                   ┌─────────────────────────┐
                   │ HF Datasets Repository  │
                   │  - All labeled data     │
                   │  - New feedback         │
                   └──────────────┬──────────┘
                                  │ weekly
                   ┌────────────────────────────────┐
                   │ GitHub Actions (Free Compute)  │
                   │ 1. Pull dataset                │
                   │ 2. Train model                 │
                   │ 3. Evaluate                    │
                   │ 4. Push new model if improved  │
                   └──────────────┬─────────────────┘
                                  │
                   ┌─────────────────────────────┐
                   │ HF Model Hub                │
                   │ - model_v1.joblib           │
                   │ - model_v2.joblib           │
                   └──────────────┬──────────────┘
                                  │ auto reload
                    ┌──────────────────────────────┐
                    │ HF Space API                 │
                    │ Loads latest model           │
                    └──────────────────────────────┘

```

This gives you **a full production ML workflow with $0 spent**.

---

# **3. Features**

### **Core Features**

- Real-time sentiment prediction
- Batch prediction
- Feedback endpoint (user-corrected labels)
- Low-confidence sample logging
- Model versioning and auto-promotion
- Weekly automatic retraining
- Free CI/CD pipeline
- Free dataset and model storage
- LLM-assisted self-labeling (NIM/others)

---

# **4. Technology Stack**

### **Runtime / API**

- FastAPI
- Uvicorn
- Python 3.10+

### **ML**

- scikit-learn
- TF-IDF + Logistic Regression (fast, light, CPU-friendly)
- Optional: DistilBERT (if HF Space allows)

### **Storage (Free)**

- HuggingFace Model Hub → model artifacts
- HuggingFace Datasets → training data + feedback
- GitHub → CI/CD + retraining

### **Self-Labelling (Optional, Free)**

- NVIDIA NIM (free trial + daily limits)
- Google Gemini Nano
- OpenAI o3-mini free tier

---

# **5. System Components**

---

## **5.1 FastAPI Application (HF Space)**

Endpoints:

### **1. POST /v1/sentiment**

Input:

```json
{ "text": "I love this!" }
```

Response:

```json
{
  "label": "positive",
  "confidence": 0.93
}
```

### **2. POST /v1/sentiment/batch**

Analyzes multiple texts.

### **3. POST /v1/feedback**

Used for:

- user-corrected labels
- storing low-confidence predictions

### **4. GET /health**

Health check.

---

## **5.2 Self-Learning Logic**

### When model prediction is low-confidence:

```
confidence < 0.55  → log to HF dataset
```

### When user corrects a label:

Store:

```
text
model_label
correct_label
confidence
timestamp
```

### Weekly pipeline uses new rows to learn.

---

## **5.3 Weekly Retraining (GitHub Actions)**

GitHub Actions YAML:

- Pull dataset
- Retrain model
- Evaluate performance
- If improved → push new model
- If worse → discard

Auto-deployment → HF Space auto reloads the new model.

---

# **6. Free Deployment Setup**

This is the **official zero-cost stack** you’ll use.

### **Hosting:**

HuggingFace Spaces (FastAPI template)

### **Dataset Storage:**

HuggingFace Datasets repo

### **Model Storage:**

HuggingFace Model Hub

### **Retraining:**

GitHub Actions (weekly cron)

### **Self-Labelling:**

(Optional)

NVIDIA NIM (free limits)

---

# **7. Directory Structure (Full Project)**

```
sentiment-api/
│
├── app/
│   ├── main.py
│   ├── schemas.py
│   ├── service.py
│   ├── auth.py
│   ├── utils.py
│   ├── model_loader.py
│
├── model/
│   ├── train_local.py
│   ├── retrain_weekly.py
│   ├── evaluate.py
│   └── model.joblib
│
├── datasets/
│   ├── base_dataset.csv
│   ├── feedback_buffer.csv
│
├── scripts/
│   ├── push_model_hf.py
│   ├── pull_dataset_hf.py
│   └── self_label_nim.py
│
├── .github/
│   └── workflows/
│       └── weekly_retrain.yml
│
├── requirements.txt
├── README.md
└── Dockerfile
```

---

# **8. Weekly Retraining Workflow**

## **Step 1 — Collect feedback + weak samples**

Stored in HF Datasets.

## **Step 2 — GitHub Actions Cron Trigger**

```
schedule:
  - cron: "0 3 * * 0"
```

## **Step 3 — Retrain**

- Merge old + new data
- Train new model
- Evaluate
- If better → promote

## **Step 4 — Push model**

To HF Model Hub.

## **Step 5 — HF Space auto-reloads**

New model active within ~10 seconds.

---

# **9. Timeline — 6 Weeks (Realistic + Achievable)**

This timeline matches:

- your constraints
- your free tools
- your skill level
- zero-cost infra

---

# **WEEK 1 — Core API + Basic Model**

Build FastAPI endpoints

Train TF-IDF + Logistic Regression baseline

Create HF Space

Deploy basic API with static model

Test inference speed

**Output:**

- Running live API
- Manual sentiment analysis working

---

# **WEEK 2 — Feedback System + Low Confidence Logging**

Add `/feedback` endpoint

Implement confidence threshold logic

Store feedback to HF Dataset

Add logging and request metadata

**Output:**

- API logging uncertain samples
- Feedback storing mechanism complete

---

# **WEEK 3 — Dataset + Model Hub Integration**

Create HF Dataset repo

Push base dataset

Link HF Space to load models from HF

Add script to pull dataset from HF

Add script to push model to HF

**Output:**

- Data + models separated from code
- Production-style artifact management

---

# **WEEK 4 — Automatic Weekly Retraining**

Write training script

Build evaluation + promotion logic

Write GitHub Actions workflow

Test weekly pipeline manually

Ensure new model auto-loads in HF Space

**Output:**

- Fully automated model training
- Deployment without manual intervention

---

# **WEEK 5 — Self-Labelling System**

Add optional free LLM-assisted labeling

- NIM/Gemini/O3-mini
    
    Integrate self-labeler into weekly training
    
    Add heuristics for quality checks
    

**Output:**

- Model “self-learns” new examples
- Better generalization over time

---

# **WEEK 6 — Optimization & Monitoring**

Add logging dashboard (Streamlit or simple HTML)

Add model performance tracking

Add alert triggers if model degrades

Improve TF-IDF parameters

Add pipeline caching

**Output:**

- Polished, stable, production-grade project
- Ready to show recruiters and teams

---

# **10. Deliverables at the End of 6 Weeks**

You will have:

- **A fully deployed public API**
- **Zero hosting cost (HF Space + GitHub Actions)**
- **A self-improving ML model retrained weekly**
- **Model versioning + auto-promotion**
- **Self-labelling using free LLM APIs**
- **A system architecture used by real ML teams**
- **A portfolio project strong enough for interviews**

---

# **11. Future Enhancements (If Free GPU Becomes Available)**

If you get HF free GPU or Colab GPU:

- Move to DistilBERT
- Add multi-language support
- Add emotion classification
- Add topic modeling
- Add embeddings cache

---

# **12. Summary**

This project gives you:

- Professional architecture
- Zero cloud cost
- Continual ML improvement
- Deployment + MLOps pipeline
- Real, practical self-learning
- Maintainable long-term system

It’s *exactly* what companies expect from ML engineers and MLOps engineers — but **built with absolutely no money spent**.