# backend/app.py
import os
import glob
import json
import math
import traceback
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# ---------------------- Config ----------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

FEATURES: List[str] = ["Seed","Fertilizer","Chemicals","Services","FLE","Repairs","Water","Interest"]
TARGET_ITEM = "Value"

# Gradient descent hyperparams (your notebook values)
GD_ITERS = 1000
GD_LR = 1e-3

os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------- Utilities ----------------------

def _to_native(o):
    import numpy as _np
    if isinstance(o, dict):
        return {k: _to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_native(v) for v in o]
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.bool_,)):
        return bool(o)
    return o

def _clean_items(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Item"] = df["Item"].astype(str)
    df["Item"] = df["Item"].str.replace(r"\s*[¹²³]\s*$", "", regex=True).str.strip()
    df["Item"] = df["Item"].replace({
        "Fertilizer ¹ ": "Fertilizer",
        "Custom services ² ": "Services",
        "Fuel, lube, and electricity": "FLE",
        "Purchased irrigation water": "Water",
        "Interest on operating capital": "Interest",
        "Value of production less operating costs": "Value",
    })
    return df

def _load_crop_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"Category", "Item", "Region", "Year", "Value"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = _clean_items(df)

    op_data = df[df["Category"] == "Operating costs"].copy()
    val_data = df[df["Category"] == "Net value"].copy()

    op_data = op_data[op_data["Item"] != "Total, operating costs"]
    op_data = op_data[op_data["Region"] != "U.S. total"]

    val_data = val_data[val_data["Item"] != "Value of production less total costs listed"]
    val_data = val_data[val_data["Region"] != "U.S. total"]
    val_data["Item"] = "Value"

    op_pivot = op_data.pivot_table(index=["Year","Region"], columns="Item", values="Value").reset_index()
    val_pivot = val_data.pivot_table(index=["Year","Region"], columns="Item", values="Value").reset_index()

    for f in FEATURES:
        if f not in op_pivot.columns:
            op_pivot[f] = 0.0

    merged = pd.merge(op_pivot, val_pivot, on=["Year","Region"])
    keep = ["Year","Region"] + FEATURES + [TARGET_ITEM]
    merged = merged[keep].dropna().reset_index(drop=True)
    return merged

def _available_crop_files() -> Dict[str, str]:
    files = glob.glob(os.path.join(DATA_DIR, "*CostReturn.csv"))
    mapping = {}
    for p in files:
        name = os.path.basename(p)
        crop = name.replace("CostReturn.csv","").lower()
        mapping[crop] = p
    return mapping

def _gd_train(Xz: np.ndarray, y: np.ndarray, iters: int = GD_ITERS, lr: float = GD_LR) -> Tuple[np.ndarray, np.ndarray]:
    """ Minimize 0.5 * ||Xz*theta - y||^2 via gradient descent. Xz has a bias column of ones. """
    m, n = Xz.shape
    theta = np.zeros(n, dtype=float)
    loss_hist = np.zeros(iters, dtype=float)
    for t in range(iters):
        r = Xz.dot(theta) - y
        loss_hist[t] = 0.5 * float(np.dot(r, r))
        g = Xz.T.dot(r)
        theta = theta - lr * g
    return theta, loss_hist


# ---------------------- Fit/Load per-crop model ----------------------

def _fit_model_for_crop(
    crop: str,
    df: pd.DataFrame,
    mode: str = "none",                 # "none" (train on all rows, no test) OR "random" (random split)
    test_size: float = 0.2,
    random_state: int = 42,
    normalize_test_with: str = "train"  # "train" (proper) OR "test" (replicate old leaky normalization, if desired)
) -> Dict:
    """
    Matches your classic pipeline:
      - features: 8 costs only
      - standardize (mean/std)
      - add bias
      - gradient descent
    If mode="random", we compute test metrics using a random row split.
    If normalize_test_with="test", we scale X_test with its own mean/std (replicates past setups that often yield higher R²).
    """
    X_full = df[FEATURES].values.astype(float)
    y_full = df[TARGET_ITEM].values.astype(float)

    if mode == "random":
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_full, y_full, test_size=test_size, random_state=random_state, shuffle=True
        )
    else:
        X_tr, y_tr = X_full, y_full
        X_te, y_te = np.empty((0, len(FEATURES))), np.empty((0,))

    # --- fit scaler on TRAIN ---
    means_tr = X_tr.mean(axis=0)
    stds_tr = X_tr.std(axis=0)
    stds_tr_safe = stds_tr.copy()
    stds_tr_safe[stds_tr_safe == 0] = 1.0

    Z_tr = (X_tr - means_tr) / stds_tr_safe
    Xz_tr = np.c_[np.ones(Z_tr.shape[0]), Z_tr]

    # train GD
    theta, loss_hist = _gd_train(Xz_tr, y_tr, GD_ITERS, GD_LR)

    # train metrics
    yhat_tr = Xz_tr.dot(theta)
    r2_tr = float(r2_score(y_tr, yhat_tr)) if len(y_tr) else None
    rmse_tr = float(np.sqrt(mean_squared_error(y_tr, yhat_tr))) if len(y_tr) else None

    # test metrics (optional)
    r2_te: Optional[float] = None
    rmse_te: Optional[float] = None
    if mode == "random" and len(y_te):
        if normalize_test_with == "test":
            means_te = X_te.mean(axis=0)
            stds_te = X_te.std(axis=0)
            stds_te_safe = stds_te.copy()
            stds_te_safe[stds_te_safe == 0] = 1.0
            Z_te = (X_te - means_te) / stds_te_safe
        else:
            # Proper scaling: use TRAIN means/stds
            Z_te = (X_te - means_tr) / stds_tr_safe
        Xz_te = np.c_[np.ones(Z_te.shape[0]), Z_te]
        yhat_te = Xz_te.dot(theta)
        r2_te = float(r2_score(y_te, yhat_te))
        rmse_te = float(np.sqrt(mean_squared_error(y_te, yhat_te)))

    # save artifacts
    crop_dir = os.path.join(MODELS_DIR, crop)
    os.makedirs(crop_dir, exist_ok=True)

    meta = {
        "crop": crop,
        "features": FEATURES,
        "theta": theta.tolist(),              # [bias, w1..w8] in standardized space (TRAIN scaler)
        "means": means_tr.tolist(),           # TRAIN means
        "stds": (stds_tr_safe).tolist(),      # TRAIN stds (zeros mapped to 1.0)
        "gd": {"iters": GD_ITERS, "lr": GD_LR},
        "split": {
            "mode": mode,
            "test_size": test_size if mode == "random" else None,
            "random_state": random_state if mode == "random" else None,
            "normalize_test_with": normalize_test_with if mode == "random" else None,
            "rows_train": int(len(y_tr)),
            "rows_test": int(len(y_te)),
        },
        "metrics": {
            "r2_train": r2_tr,
            "rmse_train": rmse_tr,
            "r2_test": r2_te,
            "rmse_test": rmse_te,
        },
        "rows": int(len(df)),
        "years": {
            "min": int(df["Year"].min()),
            "max": int(df["Year"].max()),
            "held_out": []  # classic setup (no time holdout here)
        }
    }

    meta_path = os.path.join(crop_dir, "meta.json")
    tmp_path = meta_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(_to_native(meta), f, indent=2)
    os.replace(tmp_path, meta_path)

    return {"crop": crop, "meta": meta}

def _is_gd_meta(meta: dict) -> bool:
    return isinstance(meta, dict) and all(k in meta for k in ("theta", "means", "stds"))

def _load_or_train_all() -> Dict[str, Dict]:
    registry = {}
    crop_files = _available_crop_files()
    for crop, path in crop_files.items():
        try:
            crop_dir = os.path.join(MODELS_DIR, crop)
            meta_p   = os.path.join(crop_dir, "meta.json")

            if os.path.exists(meta_p):
                try:
                    meta = json.load(open(meta_p))
                    if not _is_gd_meta(meta):
                        raise ValueError("Outdated meta schema")
                    entry = {"crop": crop, "meta": meta}
                except Exception as e:
                    print(f"[INFO] Artifacts for {crop} unreadable/outdated ({e}). Retraining (mode=none)...")
                    df = _load_crop_csv(path)
                    entry = _fit_model_for_crop(crop, df, mode="none")
            else:
                df = _load_crop_csv(path)
                entry = _fit_model_for_crop(crop, df, mode="none")

            registry[crop] = entry

        except Exception as e:
            print(f"[WARN] Failed to prepare crop {crop}: {e}")
            traceback.print_exc()
    return registry


# ---------------------- Prediction ----------------------

def _predict_single(crop_entry: Dict, inputs: Dict[str, float]) -> Dict:
    meta = crop_entry["meta"]
    theta = np.array(meta["theta"], dtype=float)  # [bias, w1..w8]
    means = np.array(meta["means"], dtype=float)  # TRAIN means
    stds  = np.array(meta["stds"], dtype=float)   # TRAIN stds (safe)

    x = np.array([float(inputs.get(f, 0.0)) for f in FEATURES], dtype=float)
    z = (x - means) / stds
    xz = np.r_[1.0, z]

    y_hat = float(np.dot(theta, xz))

    w = theta[1:]
    contribs = {f: float(w[i] * z[i]) for i, f in enumerate(FEATURES)}
    marginal = {f: float(w[i] / (stds[i] if stds[i] != 0 else 1.0)) for i, f in enumerate(FEATURES)}

    return {
        "prediction": y_hat,
        "interval_80": [None, None],   # not part of original project
        "interval_95": [None, None],
        "contributions": contribs,
        "marginal_per_dollar": marginal,
        "coef": {f: float(w[i]) for i, f in enumerate(FEATURES)},
        "intercept": float(theta[0]),
    }


# ---------------------- Flask App ----------------------

app = Flask(__name__)
CORS(app)

REGISTRY = _load_or_train_all()


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "crops": sorted(list(REGISTRY.keys()))})


@app.get("/api/crops")
def crops():
    return jsonify(sorted(list(REGISTRY.keys())))


@app.get("/api/debug/files")
def debug_files():
    files = _available_crop_files()
    out = {}
    for crop, p in files.items():
        try:
            df = _load_crop_csv(p)
            out[crop] = {
                "path": p,
                "rows": int(len(df)),
                "years": [int(df["Year"].min()), int(df["Year"].max())],
                "features_present": [f for f in FEATURES if f in df.columns],
            }
        except Exception as e:
            out[crop] = {"path": p, "error": str(e)}
    return jsonify(out)


@app.post("/api/predict")
def predict():
    data = request.get_json(force=True)
    crop = str(data.get("crop","")).lower()
    inputs = data.get("inputs", {}) or {}

    if crop not in REGISTRY:
        return jsonify({"error": f"Unknown crop '{crop}'"}), 400

    for f in FEATURES:
        inputs.setdefault(f, 0.0)

    out = _predict_single(REGISTRY[crop], inputs)
    return jsonify({
        "crop": crop,
        "prediction": out["prediction"],
        "interval_80": out["interval_80"],
        "interval_95": out["interval_95"],
        "contributions": out["contributions"],
        "marginal_per_dollar": out["marginal_per_dollar"],
        "model_meta": REGISTRY[crop]["meta"]
    })


@app.post("/api/predict-multi")
def predict_multi():
    data = request.get_json(force=True)
    items = data.get("crops", [])
    if not items:
        return jsonify({"error":"Provide 'crops': [...]"}), 400

    shares = [max(0.0, float(it.get("share", 0.0))) for it in items]
    total = sum(shares)
    shares = ([1.0/len(items)] * len(items)) if total <= 0 else [s/total for s in shares]

    per_crop = []
    agg_pred = 0.0
    agg_contrib = {f: 0.0 for f in FEATURES}

    for idx, it in enumerate(items):
        crop = str(it.get("crop","")).lower()
        if crop not in REGISTRY:
            return jsonify({"error": f"Unknown crop '{crop}'"}), 400
        inputs = it.get("inputs", {}) or {}
        for f in FEATURES:
            inputs.setdefault(f, 0.0)

        res = _predict_single(REGISTRY[crop], inputs)
        w = shares[idx]
        agg_pred += w * res["prediction"]
        for f, v in res["contributions"].items():
            agg_contrib[f] += w * float(v)

        per_crop.append({
            "crop": crop,
            "share": w,
            "prediction": res["prediction"],
            "interval_80": res["interval_80"],
            "interval_95": res["interval_95"]
        })

    return jsonify({
        "aggregate_prediction": agg_pred,
        "aggregate_contributions": agg_contrib,
        "per_crop": per_crop
    })


@app.post("/api/retrain")
def retrain():
    """
    Body examples:
    - {"crop":"all"}                          -> retrain all with defaults (mode="none")
    - {"crop":"corn","mode":"random"}         -> random 80/20 split, scale test with TRAIN stats
    - {"crop":"corn","mode":"random","normalize_test_with":"test"}  -> replicate old 'test-based' scaling
    - {"crop":"corn","mode":"random","test_size":0.25,"random_state":7}
    """
    data = request.get_json(force=True)
    target = str(data.get("crop","all")).lower()
    mode = str(data.get("mode","none")).lower()                 # "none" or "random"
    test_size = float(data.get("test_size", 0.2))
    random_state = int(data.get("random_state", 42))
    normalize_test_with = str(data.get("normalize_test_with","train")).lower()  # "train" or "test"

    crop_files = _available_crop_files()
    updated = []

    def _do_train(one_crop: str):
        path = crop_files.get(one_crop)
        if not path:
            return f"no-data:{one_crop}"
        df = _load_crop_csv(path)
        entry = _fit_model_for_crop(
            one_crop, df,
            mode=mode,
            test_size=test_size,
            random_state=random_state,
            normalize_test_with=normalize_test_with
        )
        REGISTRY[one_crop] = entry
        return "ok"

    if target == "all":
        for c in crop_files:
            _do_train(c)
            updated.append(c)
    else:
        if target not in crop_files:
            return jsonify({"error": f"No CSV found for crop '{target}'"}), 400
        _do_train(target)
        updated.append(target)

    return jsonify({"updated": updated})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
