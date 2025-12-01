# 🎭 Sentiment Analysis Self-Learning System

A production-grade sentiment analysis API that **continuously improves itself** through user feedback and automated weekly retraining—built entirely with **free, zero-cost infrastructure**.

[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-sentiment--analysis--model-yellow)](https://huggingface.co/AnuragShirke/sentiment-analysis-model)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20Dataset-sentiment--analysis--data-blue)](https://huggingface.co/datasets/AnuragShirke/sentiment-analysis-data)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features

- **Real-time Sentiment Analysis** - Predict positive/negative sentiment with confidence scores
- **Self-Learning System** - Automatically collects low-confidence predictions and user corrections
- **Weekly Auto-Retraining** - GitHub Actions pipeline retrains model with new feedback (Week 4)
- **Web UI** - Beautiful Gradio interface for easy interaction
- **RESTful API** - FastAPI with auto-generated docs
- **Cloud Storage** - Models and datasets version-controlled on HuggingFace Hub
- **Zero Cost** - Runs entirely on free infrastructure

---

## 📊 Current Status

- ✅ **Week 1:** Core API + Model (81.88% accuracy)
- ✅ **Week 2:** Feedback System + Gradio UI
- ✅ **Week 3:** HuggingFace Integration
- ⏳ **Week 4:** Automated Retraining Pipeline (In Progress)
- ⏸️ **Week 5:** Monitoring & Metrics
- ⏸️ **Week 6:** Polish & Documentation

**Progress: 50% Complete (3/6 weeks)**

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- HuggingFace account (free) - [Sign up here](https://huggingface.co/join)
- Git

### Installation

**1. Clone the repository:**
```bash
git clone <your-repo-url>
cd sentiment-api
```

**2. Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Download the dataset:**
```bash
python scripts/download_dataset.py
```
This downloads the SST-2 dataset (67,349 training samples).

**5. Train the model:**
```bash
python model/train.py
```
Expected output: ~81% accuracy on test set.

---

## 🎯 Running the Application

### Option A: Run Everything Together (Recommended)

```bash
chmod +x start.sh
./start.sh
```

This launches:
- **API** at http://localhost:8000
- **Web UI** at http://localhost:7860

Press **Ctrl+C** to stop both servers.

### Option B: Run Separately

**Terminal 1 - Start API:**
```bash
uvicorn app.main:app --reload
```

**Terminal 2 - Start Web UI:**
```bash
python ui/gradio_app.py
```

---

## 📚 Usage Guide

### 1. Web Interface (Easiest)

Open http://localhost:7860 in your browser.

**Analyze Tab:**
- Enter text or try example phrases
- Click "Analyze Sentiment"
- View prediction + confidence score
- Submit corrections if wrong

**Model Info Tab:**
- View current model version
- Check accuracy metrics
- See total predictions and feedback collected

### 2. API Endpoints

**Interactive Documentation:**  
Visit http://localhost:8000/docs for Swagger UI.

**Examples:**

**Predict sentiment:**
```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely loved this movie!"}'
```

Response:
```json
{
  "text": "I absolutely loved this movie!",
  "label": "positive",
  "confidence": 0.904,
  "model_version": "v1"
}
```

**Submit feedback (correction):**
```bash
curl -X POST http://localhost:8000/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "text": "It was okay I guess",
    "predicted_label": "positive",
    "correct_label": "neutral",
    "confidence": 0.62
  }'
```

**Batch prediction:**
```bash
curl -X POST http://localhost:8000/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "I love this!",
      "Terrible experience",
      "Not bad, not great"
    ]
  }'
```

**Get model info:**
```bash
curl http://localhost:8000/v1/model_info
```

**Health check:**
```bash
curl http://localhost:8000/health
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file (or export in terminal):

```bash
# HuggingFace Credentials (optional, for cloud features)
export HF_USERNAME="your_username"
export HF_TOKEN="hf_xxxxxxxxxxxxx"

# Model Loading Strategy
export LOAD_MODEL_FROM_HUB=false  # true = load from HF Hub, false = local
```

### Model Configuration

Edit `app/config.py`:
- `CONFIDENCE_THRESHOLD` - Below this, predictions are logged (default: 0.55)
- `MODEL_VERSION` - Current model version identifier

---

## 🤗 HuggingFace Integration (Optional)

To use cloud storage for models and datasets:

### 1. Get HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Select **Write** access
4. Copy the token

### 2. Set Credentials

```bash
export HF_USERNAME="your_username"
export HF_TOKEN="hf_xxxxxxxxxxxxx"
```

### 3. Upload Model

```bash
python scripts/push_model_hf.py
```

Creates: `https://huggingface.co/{username}/sentiment-analysis-model`

### 4. Upload Dataset

```bash
python scripts/push_dataset_hf.py
```

Creates: `https://huggingface.co/datasets/{username}/sentiment-analysis-data`

### 5. Load from HF Hub

```bash
export LOAD_MODEL_FROM_HUB=true
uvicorn app.main:app --reload
```

The API will now download and cache models from HuggingFace.

---

## 📁 Project Structure

```
sentiment-api/
├── app/
│   ├── main.py              # FastAPI application
│   ├── schemas.py           # Request/response models
│   ├── model_service.py     # ML inference
│   ├── feedback_service.py  # Feedback collection
│   └── config.py            # Configuration
├── model/
│   ├── train.py             # Model training script
│   ├── model_v1.joblib      # Trained model
│   └── metrics_v1.joblib    # Model metrics
├── data/
│   ├── base_train.csv       # Training data (67,349 samples)
│   ├── base_test.csv        # Test data (872 samples)
│   └── feedback_buffer.csv  # Collected feedback
├── scripts/
│   ├── download_dataset.py  # Download SST-2 dataset
│   ├── push_model_hf.py     # Upload model to HF Hub
│   └── push_dataset_hf.py   # Upload dataset to HF Hub
├── ui/
│   └── gradio_app.py        # Web interface
├── requirements.txt
├── start.sh                 # Launch script
└── README.md
```

---

## 🧪 Testing

### Unit Tests (Coming in Week 5)
```bash
pytest tests/
```

### Manual Testing

**Test low-confidence logging:**
1. Enter text with mixed sentiment: "It was okay I guess"
2. Should predict with <70% confidence
3. Check `data/feedback_buffer.csv` - sample should be logged

**Test feedback collection:**
1. Get a prediction
2. Submit correction via UI
3. Verify entry in `data/feedback_buffer.csv`

**Test batch prediction:**
```bash
curl -X POST http://localhost:8000/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Awful!", "Meh"]}'
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│              User Interaction Layer               │
│  ┌─────────────┐          ┌──────────────┐      │
│  │  Gradio UI  │◄────────►│  FastAPI     │      │
│  │  Port 7860  │          │  Port 8000   │      │
│  └─────────────┘          └──────┬───────┘      │
└────────────────────────────────────┼─────────────┘
                                     │
                         ┌──────────▼───────────┐
                         │  Model Service       │
                         │  - Local loading     │
                         │  - HF Hub loading    │
                         └──────────┬───────────┘
                                    │
       ┌────────────────────────────┼───────────────────────┐
       │                            │                       │
       ▼                            ▼                       ▼
┌─────────────┐          ┌────────────────┐    ┌──────────────────┐
│  Local Disk │          │  HF Model Hub  │    │  HF Datasets Hub │
│  - model/   │          │  - model_v*.   │    │  - train/test    │
│  - data/    │          │    joblib      │    │  - feedback CSV  │
└─────────────┘          └────────────────┘    └──────────────────┘
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Algorithm** | TF-IDF + Logistic Regression |
| **Test Accuracy** | 81.88% |
| **Precision (Positive)** | 0.80 |
| **Precision (Negative)** | 0.84 |
| **Recall (Positive)** | 0.85 |
| **Recall (Negative)** | 0.78 |
| **Training Time** | ~30 seconds |
| **Inference Time** | <100ms per request |

---

## 🔄 How Self-Learning Works

1. **User submits text** → API predicts sentiment
2. **Low confidence?** (<55%) → Automatically logged to `feedback_buffer.csv`
3. **Prediction wrong?** → User submits correction via UI
4. **Weekly retraining** (GitHub Actions - Week 4) → Model improves
5. **Auto-deployment** → New model version deployed if accuracy improved

---

## 🛠️ Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'app'**

Solution: Run API using uvicorn, not python directly:
```bash
uvicorn app.main:app --reload  # ✅ Correct
python app/main.py             # ❌ Wrong
```

**2. Model not found error**

Solution: Make sure you've trained the model first:
```bash
python model/train.py
```

**3. HuggingFace upload fails**

Check:
- Token has **write** permissions
- `HF_USERNAME` and `HF_TOKEN` are set
- Network connection is stable

**4. Port already in use**

Change ports in respective files:
- API: Edit `start.sh` or use `--port 8001`
- UI: Edit `ui/gradio_app.py` line: `demo.launch(server_port=7861)`

---

## 🗺️ Roadmap

- [x] **Week 1:** Core API + Model Training
- [x] **Week 2:** Feedback System + Gradio UI
- [x] **Week 3:** HuggingFace Cloud Integration
- [ ] **Week 4:** Automated Retraining Pipeline (GitHub Actions)
- [ ] **Week 5:** Monitoring Dashboard + Metrics
- [ ] **Week 6:** Production Deployment + Documentation

---

## 📖 Documentation

- **API Docs:** http://localhost:8000/docs (when running)
- **Progress Report:** See [PROGRESS.md](PROGRESS.md)
- **Execution Plan:** See [EXECUTION_PLAN.md](EXECUTION_PLAN.md)

---

## 🤝 Contributing

This is a learning project, but suggestions are welcome!

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📜 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **Dataset:** Stanford SST-2 (GLUE Benchmark)
- **Framework:** FastAPI by Sebastián Ramírez
- **UI:** Gradio by HuggingFace
- **Hosting:** HuggingFace Spaces (Free Tier)

---

## 📧 Contact

For questions or feedback, open an issue on GitHub.

---

**Built with ❤️ as a demonstration of MLOps practices on zero-cost infrastructure.**
