# GitHub Actions Workflows

This directory contains automated CI/CD workflows for the Sentiment Analysis project.

## Weekly Retraining Workflow

**File:** `weekly_retrain.yml`

### What It Does

Automatically retrains the sentiment analysis model every week by:

1. **Downloading latest feedback** from HuggingFace Datasets
2. **Merging** user corrections with base training data
3. **Training** a new model version (model_v2, v3, etc.)
4. **Evaluating** performance on held-out test set
5. **Comparing** with current production model
6. **Deploying** to HuggingFace Hub **only if accuracy improves** by ≥0.1%

### Schedule

- **Automatic:** Every Sunday at 3:00 AM UTC
- **Manual:** Can be triggered via "Actions" tab → "Weekly Retrain" → "Run workflow"

### Required Secrets

Add these in: **Settings → Secrets and variables → Actions → New repository secret**

| Secret Name | Description |
|-------------|-------------|
| `HF_TOKEN` | Your HuggingFace access token (with write permissions) |
| `HF_USERNAME` | Your HuggingFace username |

### How to Manually Trigger

1. Go to **Actions** tab in GitHub
2. Click **Weekly Model Retraining** workflow
3. Click **Run workflow** button
4. Select branch (usually `main`)
5. Click green **Run workflow** button

### Monitoring

**View workflow runs:**
- Actions tab → Weekly Model Retraining → Click on a specific run

**Check logs:**
- Click on a run → Click on "retrain" job → Expand steps to see detailed logs

**Download artifacts:**
- Scroll down on run page → "Artifacts" section → Download logs

### Expected Outcomes

#### Scenario 1: Model Improved ✅
- New model version (e.g., v2) is created
- Uploaded to HuggingFace Model Hub
- Logs show: "SUCCESS: Model v2 deployed!"
- Improvement percentage logged

#### Scenario 2: No Improvement 🔄
- Current model kept in production
- Logs show: "Skipping deployment (no improvement)"
- Reason logged (e.g., "No significant change: 0.8188 vs 0.8190")

#### Scenario 3: Insufficient Feedback ⚠️
- No retraining occurs
- Logs show: "Not enough feedback (minimum: 10 user corrections)"
- Workflow exits early

### Troubleshooting

**Workflow fails with "HF_TOKEN not set":**
- Check that secrets are added in repository settings
- Verify secret names exactly match: `HF_TOKEN` and `HF_USERNAME`

**Workflow fails during model upload:**
- Check HF_TOKEN has **write** permissions
- Verify HuggingFace model repository exists
- Check network connectivity in GitHub Actions logs

**No new model despite feedback:**
- Check if feedback meets minimum threshold (10 samples)
- Verify accuracy improvement is >0.1%
- Review comparison logic in logs

**Model downloads but doesn't deploy:**
- Promotion logic may have rejected it (accuracy worse or same)
- Check evaluation metrics in logs

### Testing Locally

Before relying on automated workflow, test locally:

```bash
# Set credentials
export HF_TOKEN="hf_xxxxx"
export HF_USERNAME="your_username"

# Run retraining
python model/retrain.py
```

Expected output: Step-by-step progress with decision logging.

### Workflow Artifacts

Each run stores:
- Trained model files (model_vN.joblib)
- Metrics files (metrics_vN.joblib)
- Retention: 30 days

Access via: Run page → "Artifacts" section

### Cost

- **GitHub Actions:** Free for public repos, 2000 min/month for private
- **This workflow:** ~3-5 minutes per run = ~20 min/month
- **Well within free limits** ✅

### Future Enhancements

Planned improvements:
- Slack/email notifications on successful deployment
- Performance comparison charts
- Automated rollback if deployed model degrades
- Multi-metric evaluation (not just accuracy)
