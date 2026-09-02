# RM Remark Quality API

Production inference service for scoring Relationship Manager (RM) visit remarks on a **0–100 quality scale** and mapping them to **Trash / Average / Good** labels.

## Architecture

```
Remark (raw string)
  → Text cleaning (lowercase, whitespace/symbol normalisation)
  → 37 semantic feature extraction (keyword/pattern-based)
  → Sentence embedding via BAAI/bge-large-en-v1.5 (1024-dim)
  → PCA transform (1024 → 247 dims)
  → Concatenate: [247 PCA] + [37 semantic] = 284 features
  → Reindex to exact feature_columns.pkl order
  → StackedEnsembleModel.predict() → raw score
  → Clip to [0, 100]
  → Map to label via label_mapping.json
  → Generate human-readable explanation
  → Return response
```

**Model**: Stacked Ensemble (LightGBM + XGBoost + CatBoost + RandomForest) with Ridge meta-learner, trained on 3,054 labeled RM visit remarks.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env as needed (API key, port, etc.)
```

### 3. Run the service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

> **Note**: The first startup will download the `BAAI/bge-large-en-v1.5` embedding model (~1.3 GB). Subsequent starts use the cached model.

---

## API Endpoints

### `POST /predict` — Score a single remark

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "remark": "Visited Croma store and met SEC regarding Samsung display. Discussed sell-out performance and trained staff on new Galaxy lineup. SEC committed to improving attachment rate. Will follow up next week."
  }'
```

**Response:**

```json
{
  "quality_score": 78.3,
  "quality_label": "Good",
  "story_elements": {
    "visit_purpose": true,
    "stakeholder": true,
    "discussion": true,
    "response": true,
    "action": true,
    "outcome": true,
    "followup": true
  },
  "strengths": [
    "Visit purpose clearly stated",
    "Stakeholder identified",
    "Business discussion documented",
    "Stakeholder response captured",
    "Action taken documented",
    "Business outcome mentioned",
    "Follow-up plan stated"
  ],
  "missing_elements": []
}
```

### `POST /predict-batch` — Score multiple remarks

```bash
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "records": [
      {"remark": "Visited store. Everything is fine."},
      {"remark": "Met SEC at Croma HSR Layout. Discussed Galaxy S25 sell-out targets. Current achievement is 45%. Trained 2 promoters on features. SEC agreed to push display compliance. Will revisit Friday to check progress."}
    ]
  }'
```

### `GET /health` — Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "artifact_count": 6
}
```

### `GET /version` — Version info

```bash
curl http://localhost:8000/version
```

### `GET /metrics` — Request metrics

```bash
curl http://localhost:8000/metrics
```

### `POST /admin/reload-artifacts` — Hot-swap artifacts

```bash
curl -X POST http://localhost:8000/admin/reload-artifacts \
  -H "X-API-Key: your-api-key"
```

---

## Docker Deployment

### Build

```bash
docker build -t rm-quality-api .
```

### Run

```bash
docker run -p 8000:8000 --env-file .env rm-quality-api
```

The Docker build pre-downloads the embedding model, so the container starts serving immediately.

---

## Configuration

All settings are controlled via environment variables (prefixed with `RM_`). See [`.env.example`](.env.example) for the full list.

| Variable | Default | Description |
|---|---|---|
| `RM_ARTIFACT_DIR` | `./artifacts` | Path to model artifacts directory |
| `RM_API_KEY` | *(empty — auth disabled)* | API key for authenticated endpoints |
| `RM_RATE_LIMIT_REQUESTS` | `100` | Max requests per rate-limit window |
| `RM_RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window duration |
| `RM_CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `RM_PORT` | `8000` | Server port |
| `RM_MAX_BATCH_SIZE` | `500` | Maximum batch size |
| `RM_MODEL_VERSION` | `1.0.0` | Reported model version |

---

## Project Structure

```
comments_api_wrapper/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app, routes, startup
│   ├── config.py                  # Environment configuration
│   ├── schemas.py                 # Pydantic request/response models
│   ├── stacked_ensemble.py        # StackedEnsembleModel class definition
│   ├── services/
│   │   ├── __init__.py
│   │   ├── artifact_loader.py     # Singleton artifact manager
│   │   ├── predictor.py           # RemarkQualityEngine (core inference)
│   │   └── inference_pipeline.py  # Pipeline orchestration
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── preprocessing.py       # Text cleaning
│   │   └── feature_engineering.py # 37 semantic features + explainability
│   └── middleware/
│       ├── __init__.py
│       └── security.py            # API key auth, rate limiting
├── artifacts/                     # Model artifacts (not in git)
│   ├── best_model.pkl
│   ├── pca_model.pkl
│   ├── feature_columns.pkl
│   ├── embedding_model_reference.txt
│   ├── label_mapping.json
│   └── model_architecture.txt
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

## Score Interpretation

| Score Range | Label | Meaning |
|---|---|---|
| 0 – 40 | **Trash** | Little to no usable business information |
| 41 – 70 | **Average** | Some useful information, but incomplete |
| 71 – 100 | **Good** | A genuinely complete, specific business story |

Scores above 75 are intentionally rare. A perfect 100 essentially never happens — even a strong remark usually misses one story element.

**Important**: Scores reflect *remark completeness*, not visit quality. A low score means the remark failed to convey a business story — not that the visit went poorly.
