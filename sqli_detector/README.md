# ML-Based SQL Injection Detection for APIs

SVM + XGBoost + MLP detectors, a FastAPI detection endpoint, SQLite security logs and a Streamlit security dashboard.

```
Incoming API request -> preprocessing & feature extraction (TF-IDF + engineered features)
  -> SVM | XGBoost | MLP -> Benign / Malicious / Suspicious -> Allow / Flag / Block
  -> SQLite security logs -> Streamlit analytics dashboard (heatmaps, model comparison, reports)
```

## Files
| File | Purpose |
|---|---|
| `train.py` | samples the dataset, builds features, trains the 3 models, saves metrics + an evaluation set |
| `features.py` | input cleaning, HTTP/JSON parsing, TF-IDF + engineered features, attack-signature patterns |
| `augment.py` | optional clean-benign / textbook-attack augmentation (~10%) |
| `detector.py` | loads models, produces label, probability, action, timing, explanation |
| `api.py` | FastAPI endpoint: `POST /detect`, `POST /detect/batch`, `GET /health` |
| `database.py` | SQLite logging (`data/security_logs.db`) |
| `dashboard.py` | Streamlit dashboard (7 pages) |
| `analytics.py`, `charts.py`, `report.py` | metrics, Plotly charts/heatmaps, HTML + PDF report |
| `config.py` | paths, thresholds (Allow < 0.40 <= Flag < 0.80 <= Block), categories |

## Setup & run
1. Python 3.9+ (3.10-3.12 recommended). In this folder:
   ```
   python -m venv venv
   venv\Scripts\activate          (Windows)      source venv/bin/activate   (macOS/Linux)
   pip install -r requirements.txt
   ```
2. Extract `sqli_by_type.zip` into `data/` so these exist: `data/sqli_by_type/union_based.csv`, ..., `none_type_benign.csv`.
3. Train (one time):
   ```
   python train.py --quick      # ~1-2 min smoke test
   python train.py              # full run (default 120k malicious + 120k benign samples)
   ```
   Low on RAM? `python train.py --per-type 8000 --benign 50000`.  More data? `--per-type 40000 --benign 240000`.
4. Terminal A - detection API:  `uvicorn api:app --port 8000`
5. Terminal B - dashboard:      `streamlit run dashboard.py`  (opens http://localhost:8501)
6. In the dashboard open **Automated Evaluation -> Run evaluation**, then explore the other pages.
   (If the API is not running the dashboard falls back to the in-process detector.)

## Try the API directly
```
curl -X POST http://127.0.0.1:8000/detect -H "Content-Type: application/json" \
     -d "{\"payload\": \"SELECT * FROM users WHERE username = 'admin' OR '1'='1'\", \"algorithm\": \"all\"}"
```
`algorithm`: `svm`, `xgboost`, `mlp` or `all`. `payload` can be raw SQL, a JSON body, or a complete HTTP request.

## Notes
* SVM = LinearSVC wrapped in probability calibration (a kernel SVM does not scale to hundreds of thousands of rows).
* Evaluation requests are timestamped across the last N hours (simulated) so time charts are meaningful.
* Attack category in the dashboard is assigned by signature matching on requests the models flagged malicious.
* The dataset is synthetic and very separable, so held-out scores are ~100%. Treat them as upper bounds,
  not production accuracy. `augment.py` helps on everyday inputs; for real deployments retrain on real traffic.
