# Sentiment Analysis Self-Learning System - Execution Plan

**A realistic, zero-cost ML system with continuous learning**

---

## Executive Summary

This is a **production-grade sentiment analysis API** that improves itself over time. Unlike typical portfolio projects, this system:

- Collects its own training data from user feedback
- Automatically retrains weekly using GitHub Actions
- Deploys updates without manual intervention
- Costs $0 to run indefinitely

**What makes this different:** Most ML projects are static snapshots. This one has a **continuous learning loop** that mirrors real production ML systems.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERACTION LAYER                    │
│  ┌────────────────┐              ┌──────────────────┐       │
│  │  Web UI Demo   │◄────────────►│  FastAPI Backend │       │
│  │  (HF Space)    │              │  (HF Space)      │       │
│  └────────────────┘              └──────────┬───────┘       │
└──────────────────────────────────────────────┼──────────────┘
                                               │
                        ┌──────────────────────┼──────────────────────┐
                        │                      ▼                      │
                        │         ┌─────────────────────────┐         │
                        │         │  Model Inference         │         │
                        │         │  + Confidence Tracking   │         │
                        │         └──────────┬───────────────┘         │
                        │                    │                         │
                        │         Low confidence or user correction?   │
                        │                    │                         │
                        │                    ▼                         │
                        │         ┌─────────────────────────┐         │
                        │         │  HF Datasets Repo        │         │
                        │         │  - feedback_data.csv     │         │
                        │         └──────────┬───────────────┘         │
                        └────────────────────┼──────────────────────────┘
                                             │
                              Weekly Trigger (Sunday 3 AM)
                                             │
                        ┌────────────────────▼──────────────────────┐
                        │     GitHub Actions (Free Runner)          │
                        │  1. Pull dataset from HF                  │
                        │  2. Merge with base data                  │
                        │  3. Train model_v{N+1}                    │
                        │  4. Evaluate on test set                  │
                        │  5. If improved → push to HF Model Hub    │
                        │  6. If worse → skip deployment            │
                        └────────────────────┬──────────────────────┘
                                             │
                                             ▼
                        ┌─────────────────────────────────┐
                        │  HF Model Hub                   │
                        │  - model_v1.joblib (baseline)   │
                        │  - model_v2.joblib (week 1)     │
                        │  - model_v3.joblib (week 2)     │
                        └─────────────────┬───────────────┘
                                          │
                              Auto-reload by HF Space
                                          │
                        ┌─────────────────▼───────────────┐
                        │  API uses latest model          │
                        └─────────────────────────────────┘
```

---

## Technology Stack (All Free)

| Component | Technology | Why |
|-----------|-----------|-----|
| **API Framework** | FastAPI | Fast, modern, auto-generates docs |
| **Model** | TF-IDF + Logistic Regression | CPU-efficient, fast inference, good baseline |
| **Hosting** | HuggingFace Spaces | Free forever, supports FastAPI + Gradio |
| **Dataset Storage** | HuggingFace Datasets | Version-controlled, free unlimited storage |
| **Model Storage** | HuggingFace Model Hub | Version-controlled model artifacts |
| **CI/CD** | GitHub Actions | 2000 free minutes/month (public repos = unlimited) |
| **Frontend** | Gradio | Dead simple UI, works seamlessly with HF Spaces |

---

## Critical Prerequisites

### Before You Start Week 1

1. **Base Dataset** - You MUST have labeled training data. Options:
   - **Recommended:** [Stanford SST-2](http://nlp.stanford.edu/sentiment/) - 67k movie reviews, clean labels
   - Alternative: [IMDb 50k](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
   - Alternative: [Twitter Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) - 1.6M tweets

2. **HuggingFace Account** - Sign up at [huggingface.co](https://huggingface.co)
   - Create access token with write permissions
   - Store as GitHub secret for Actions

3. **GitHub Account** - For Actions CI/CD

4. **Local Dev Environment**
   - Python 3.10+
   - Git
   - Basic familiarity with FastAPI

---

## Project Structure

```
sentiment-api/
│
├── app/
│   ├── main.py              # FastAPI app + endpoints
│   ├── schemas.py           # Pydantic models
│   ├── model_service.py     # Model loading + inference
│   ├── feedback_service.py  # Feedback collection logic
│   └── config.py            # Environment vars
│
├── ui/
│   └── gradio_app.py        # Simple web interface
│
├── model/
│   ├── train.py             # Initial training script
│   ├── retrain.py           # Used by GitHub Actions
│   ├── evaluate.py          # Model evaluation logic
│   └── vectorizer.joblib    # Saved TF-IDF vectorizer
│
├── data/
│   ├── base_train.csv       # Initial training data
│   ├── base_test.csv        # Held-out test set
│   └── README.md            # Data documentation
│
├── scripts/
│   ├── download_dataset.py  # Get SST-2 or IMDb
│   ├── push_to_hf.py        # Upload model to HF Hub
│   └── pull_from_hf.py      # Download feedback data
│
├── .github/
│   └── workflows/
│       └── weekly_retrain.yml
│
├── requirements.txt
├── app.py                   # HF Space entry point
├── README.md
└── .env.example
```

---

## Weekly Execution Plan

### **Week 1: Foundation - API + Model + Dataset**

**Goal:** Get a working API deployed with a trained baseline model.

#### Tasks
- [ ] Download and clean base dataset (SST-2 recommended)
- [ ] Split into train/test sets (80/20)
- [ ] Train initial TF-IDF + Logistic Regression model
- [ ] Build FastAPI with these endpoints:
  - `POST /predict` - Single text prediction
  - `POST /predict/batch` - Multiple texts
  - `GET /health` - Health check
  - `GET /model_info` - Current model version + metrics
- [ ] Local testing with curl/Postman
- [ ] Create HF Space and deploy

#### Deliverables
- ✅ Live API at `https://huggingface.co/spaces/{username}/sentiment-api`
- ✅ Baseline model accuracy on test set (target: 80%+)
- ✅ API responds in <500ms per request

#### Key Files to Create
- `app/main.py`
- `model/train.py`
- `requirements.txt`
- `app.py` (HF Space entry)

---

### **Week 2: Feedback System + UI**

**Goal:** Enable users to interact with the system and collect training data.

#### Tasks
- [ ] Add feedback endpoint:
  - `POST /feedback` - User corrections
  - Schema: `{text, predicted_label, correct_label, confidence}`
- [ ] Implement confidence threshold logic (< 0.55 = auto-log)
- [ ] Store feedback to local CSV buffer
- [ ] Build Gradio UI with:
  - Text input
  - Prediction display (label + confidence %)
  - "This is wrong" button → correction form
  - History of predictions
- [ ] Deploy UI to same HF Space (Gradio + FastAPI together)

#### Deliverables
- ✅ Live demo UI at `https://huggingface.co/spaces/{username}/sentiment-api`
- ✅ Feedback collection working
- ✅ CSV accumulating low-confidence samples

#### Key Files to Create
- `app/feedback_service.py`
- `app/schemas.py` (add FeedbackRequest)
- `ui/gradio_app.py`
- Update `app.py` to launch both FastAPI + Gradio

---

### **Week 3: HuggingFace Integration**

**Goal:** Separate data/models from code. Enable version control for both.

#### Tasks
- [ ] Create HF Dataset repository: `{username}/sentiment-feedback-data`
- [ ] Push base dataset to HF Datasets
- [ ] Create HF Model repository: `{username}/sentiment-model`
- [ ] Push initial model + vectorizer to HF Model Hub
- [ ] Update API to load model from HF Hub (not local disk)
- [ ] Update feedback system to push to HF Datasets (not just local CSV)
- [ ] Add model versioning logic (v1, v2, v3...)

#### Deliverables
- ✅ Dataset repo live: `https://huggingface.co/datasets/{username}/sentiment-feedback-data`
- ✅ Model repo live: `https://huggingface.co/models/{username}/sentiment-model`
- ✅ API pulling model from HF Hub on startup
- ✅ Feedback automatically syncing to HF Datasets

#### Key Files to Create
- `scripts/push_to_hf.py`
- `scripts/pull_from_hf.py`
- Update `app/model_service.py` to load from HF Hub
- Update `app/feedback_service.py` to push to HF Datasets

---

### **Week 4: Automated Retraining Pipeline**

**Goal:** Make the system self-improving with zero manual intervention.

#### Tasks
- [ ] Write retraining script:
  - Pull latest feedback from HF Datasets
  - Merge with base training data
  - Remove duplicates
  - Train model_v{N+1}
  - Evaluate on held-out test set
  - Compare to previous version
  - If accuracy improved → push new model
  - If accuracy same/worse → discard
- [ ] Create GitHub Actions workflow:
  - Cron trigger: `0 3 * * 0` (every Sunday 3 AM)
  - Steps: checkout → setup python → run retrain script
  - Conditional push to HF Model Hub
- [ ] Add secrets to GitHub:
  - `HF_TOKEN` (HuggingFace access token)
- [ ] Test pipeline manually (GitHub Actions → "Run workflow")
- [ ] Verify HF Space auto-reloads new model

#### Deliverables
- ✅ GitHub Actions workflow running weekly
- ✅ Successful test run (manually triggered)
- ✅ New model version deployed after training
- ✅ Model version visible in API `/model_info` endpoint

#### Key Files to Create
- `model/retrain.py`
- `model/evaluate.py`
- `.github/workflows/weekly_retrain.yml`

---

### **Week 5: Monitoring & Metrics Tracking**

**Goal:** Understand how the system is performing over time.

#### Tasks
- [ ] Add metrics endpoint:
  - `GET /metrics` - Returns JSON with:
    - Total predictions made
    - Total feedback collected
    - Current model version
    - Test set accuracy
    - Average confidence score
- [ ] Log all predictions to file (cache last 1000)
- [ ] Create simple monitoring dashboard in Gradio:
  - Chart: Model accuracy over versions
  - Chart: Feedback volume over time
  - Chart: Confidence distribution
  - Table: Recent low-confidence samples
- [ ] Add model performance degradation alert:
  - If test accuracy drops >5% → log warning
  - If confidence avg drops >10% → log warning

#### Deliverables
- ✅ Metrics dashboard visible in UI
- ✅ Performance tracking over model versions
- ✅ Alerts for degradation (logged to console)

#### Key Files to Create
- `app/metrics_service.py`
- Update `ui/gradio_app.py` with dashboard tab
- `model/performance_tracker.py`

---

### **Week 6: Polish, Documentation, & Demo**

**Goal:** Make this portfolio-ready and recruiter-friendly.

#### Tasks
- [ ] Write comprehensive README.md:
  - Project overview
  - Architecture diagram (use mermaid)
  - Live demo link
  - API documentation
  - How to run locally
  - How the retraining works
- [ ] Add API authentication (optional but impressive):
  - Simple API key validation
  - Rate limiting (10 req/min for free tier)
- [ ] Improve UI aesthetics:
  - Better styling
  - Add examples ("Try these sample texts")
  - Show model explanation (TF-IDF top features)
- [ ] Create demo video (3-5 min):
  - Show API usage
  - Show UI demo
  - Show feedback collection
  - Show GitHub Actions retraining
  - Show model version upgrade
- [ ] Write blog post explaining:
  - Why continuous learning matters
  - How you built it with $0 cost
  - Technical challenges you solved

#### Deliverables
- ✅ Professional README with live demo
- ✅ Polished UI that looks production-grade
- ✅ Demo video uploaded (YouTube/LinkedIn)
- ✅ Blog post published (Medium/Dev.to)
- ✅ GitHub repo public and organized

#### Key Files to Create/Update
- `README.md` (comprehensive)
- `docs/ARCHITECTURE.md`
- `docs/API.md`
- Update `ui/gradio_app.py` (styling improvements)

---

## Decision Points & Constraints

### What We're NOT Doing (At Least Initially)

❌ **LLM-assisted self-labeling** - Adds complexity, questionable value, rate limits make it unreliable  
❌ **Transformer models (BERT, etc.)** - HF Spaces free tier CPU is too slow  
❌ **Multi-language support** - Scope creep, focus on English first  
❌ **User authentication system** - Optional, can add in Week 6 if time allows  
❌ **Production-scale monitoring** - We'll do basic metrics, not full observability stack  

### What We're Deferring to Later

⏭️ **A/B testing different models** - Interesting but not MVP  
⏭️ **Explainability features** (LIME/SHAP) - Nice-to-have, Week 7+  
⏭️ **Fine-tuned DistilBERT** - If you get HF GPU access later  
⏭️ **Kubernetes deployment** - Not needed for free HF Spaces  

---

## Success Metrics

By end of Week 6, you should have:

1. **A live, public API** that anyone can use
2. **A working web UI** that looks professional
3. **Automated weekly retraining** (verify with GitHub Actions history)
4. **At least 3 model versions** deployed (baseline + 2 retrains)
5. **Proof of improvement** (model_v3 > model_v1 in accuracy)
6. **Documentation** good enough to share with recruiters
7. **A demo video** showing the full system

---

## Risk Mitigation

### If HuggingFace Spaces is too slow:
- **Fallback:** Deploy FastAPI to [Railway.app](https://railway.app) (free tier: 500 hrs/month)
- **Fallback 2:** Use [Render.com](https://render.com) (free tier with cold starts)

### If GitHub Actions fails:
- **Fallback:** Use HF Spaces with scheduled Python script (less elegant but works)
- **Fallback 2:** Manual retraining once a week (defeats automation goal but proves concept)

### If feedback collection is too slow:
- **Solution:** Generate synthetic feedback using rule-based augmentation
- **Solution 2:** Manually add 50-100 edge cases to kickstart learning

### If model accuracy doesn't improve:
- **Expected:** Early weeks may not show improvement (need more data)
- **Fix:** Ensure feedback data quality is good (validate labels)
- **Fix:** Tune hyperparameters (C value for LogisticRegression, ngram range for TF-IDF)

---

## What Makes This Portfolio-Worthy

When you show this to recruiters/interviewers, emphasize:

1. **MLOps Skills** - You built a full ML lifecycle, not just a model
2. **Production Mindset** - Version control, automated testing, monitoring
3. **Resource Constraints** - Solved a real problem with zero budget
4. **Continuous Learning** - System improves itself, not static
5. **End-to-End Ownership** - You handled data, model, API, UI, deployment, CI/CD

This is **not a Kaggle notebook**. This is a **real system** that could handle actual users.

---

## Next Steps

1. **Review this plan** - Any changes needed?
2. **Set up prerequisites** - Download dataset, create HF account
3. **Start Week 1** - Focus on getting the baseline working
4. **Track progress** - Use GitHub issues or a Notion board

**Let's build this.**
