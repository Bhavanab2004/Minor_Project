"""Central configuration for the ML-based SQL injection detection system."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "sqli_by_type"      # extracted dataset folder
MODEL_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "data" / "security_logs.db"
EVAL_SET = BASE_DIR / "data" / "eval_set.csv"      # held-out test queries (made by train.py)
METRICS_JSON = MODEL_DIR / "metrics.json"

API_URL = os.getenv("SQLI_API_URL", "http://127.0.0.1:8000")

# dataset file -> dashboard attack category
ATTACK_FILES = {
    "union_based.csv": "Union-based",
    "boolean_based.csv": "Boolean-based",
    "error_based.csv": "Error-based",
    "time_based.csv": "Time-based",
    "meta_based.csv": "Other",
    "stackqueries_based.csv": "Other",
}
BENIGN_FILE = "none_type_benign.csv"
CATEGORIES = ["Union-based", "Boolean-based", "Error-based", "Time-based", "Other"]

ALGOS = {
    "svm": "Support Vector Machine",
    "xgboost": "XGBoost",
    "mlp": "Multilayer Perceptron",
}

# Decision policy on P(malicious)
MALICIOUS_T = 0.50   # predicted label = Malicious if p >= this
FLAG_T = 0.40        # p >= FLAG_T  -> Flag (suspicious)
BLOCK_T = 0.80       # p >= BLOCK_T -> Block

ENDPOINTS = ["/api/v1/users", "/api/v1/login", "/api/v1/search", "/api/v1/orders",
             "/api/v1/products", "/api/v1/payments", "/api/v1/reports", "/api/v1/admin/query"]
METHODS = ["GET", "POST", "PUT", "DELETE"]
