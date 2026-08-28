#!/usr/bin/env python3
"""
Retraining script for the Explainable Multi-Disease Diagnosis System.

Removes Diabetes class, adds Parkinson's, and retrains the full pipeline:
    GSE data → StandardScaler → VarianceThreshold → SelectKBest → PSO → ACO → DNN

Optimized for speed: uses staged feature reduction.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

DISEASE_MAP: dict[int, str] = {
    0: "Coronary Artery Disease",
    1: "Autoimmune Disease",
    2: "Alzheimer's Disease",
    3: "Parkinson's Disease",
}

GSE_FILES: dict[int, str] = {
    0: "GSE66360_(Cad)series_matrix (3).txt.gz",
    1: "GSE50772_(Auto-Immune)series_matrix.txt.gz",
    2: "GSE48350_(Alzheimer)series_matrix.txt.gz",
    3: "GSE6613_(Parkinson's)series_matrix.txt.gz",
}

PARKINSON_DISEASE_KEYWORDS = {"parkinson", "parkinson's disease", "parkinsons disease"}

TARGET_FEATURES = 80
PSO_POP = 20
PSO_ITER = 15
ACO_POP = 15
ACO_ITER = 10
ACO_EVAPORATION = 0.5
ACO_ALPHA = 1.0
ACO_BETA = 2.0
RANDOM_STATE = 42
DNN_EPOCHS = 80
DNN_BATCH = 32


# ═══════════════════════════════════════════════════════════════════════════════
# GEO PARSING
# ═══════════════════════════════════════════════════════════════════════════════


def parse_gse_file(filepath: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Parse GEO Series Matrix file. Returns (expression_df, sample_ids, sample_chars)."""
    with gzip.open(filepath, "rt", errors="replace") as fh:
        lines = fh.readlines()

    begin_idx = end_idx = None
    for i, line in enumerate(lines):
        if "!series_matrix_table_begin" in line:
            begin_idx = i
        if "!series_matrix_table_end" in line:
            end_idx = i
            break

    if begin_idx is None or end_idx is None:
        raise ValueError(f"Data table not found in {filepath}")

    header = lines[begin_idx + 1].strip().split("\t")
    sample_ids = [s.strip('"') for s in header[1:]]

    # Sample characteristics (for Parkinson's filtering)
    sample_chars = []
    for line in lines:
        if line.startswith("!Sample_characteristics_ch1"):
            sample_chars = [c.strip('"').lower() for c in line.strip().split("\t")[1:]]
            break

    # Parse expression data
    probes = []
    data_rows = []
    for i in range(begin_idx + 2, end_idx):
        parts = lines[i].strip().split("\t")
        probe_id = parts[0].strip('"')
        if probe_id.startswith("!") or probe_id.startswith('"') or not probe_id:
            continue
        vals = []
        for v in parts[1:]:
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                vals.append(np.nan)
        probes.append(probe_id)
        data_rows.append(vals)

    df = pd.DataFrame(data_rows, index=probes, columns=sample_ids).T
    df = df.fillna(df.median())
    return df, sample_ids, sample_chars


def load_all_datasets() -> tuple[pd.DataFrame, np.ndarray]:
    """Load all 4 GSE datasets, filter Parkinson's controls, return aligned data."""
    print("=" * 70)
    print("STEP 1: Loading GEO datasets")
    print("=" * 70)

    all_dfs = []
    all_labels = []

    for label, fname in GSE_FILES.items():
        fpath = str(PROJECT_ROOT / fname)
        print(f"  Loading {DISEASE_MAP[label]}...")
        t0 = time.time()

        df, sids, chars = parse_gse_file(fpath)

        # Filter Parkinson's to disease samples only
        if label == 3:
            mask = [any(kw in c for kw in PARKINSON_DISEASE_KEYWORDS) for c in chars]
            n_pd = sum(mask)
            n_ctrl = len(mask) - n_pd
            print(f"    Disease: {n_pd}, Controls excluded: {n_ctrl}")
            df = df.iloc[mask]

        print(f"    {df.shape[0]} samples, {df.shape[1]} probes ({time.time()-t0:.1f}s)")
        all_dfs.append(df)
        all_labels.extend([label] * len(df))

    # Common probes
    probe_sets = [set(df.columns) for df in all_dfs]
    common = probe_sets[0]
    for ps in probe_sets[1:]:
        common &= ps
    common = sorted(common)
    print(f"\n  Common probes: {len(common)}")

    # Align
    aligned = [df.reindex(columns=common, fill_value=0.0) for df in all_dfs]
    X = pd.concat(aligned, ignore_index=True)
    y = np.array(all_labels, dtype=np.int32)

    print(f"  Total: {X.shape[0]} samples × {X.shape[1]} features")
    for k in sorted(set(y)):
        print(f"    {DISEASE_MAP[k]}: {(y==k).sum()}")

    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════


def preprocess(X: pd.DataFrame, y: np.ndarray):
    """StandardScaler → VarianceThreshold → SelectKBest. Returns everything needed."""
    print("\n" + "=" * 70)
    print("STEP 2: Preprocessing")
    print("=" * 70)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train_raw)}, Test: {len(X_test_raw)}")

    # StandardScaler
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw.values.astype(np.float64))
    X_test_sc = scaler.transform(X_test_raw.values.astype(np.float64))
    print(f"  StandardScaler: {X_train_sc.shape[1]} features")

    # VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    X_train_var = selector.fit_transform(X_train_sc)
    X_test_var = selector.transform(X_test_sc)
    kept_mask = selector.get_support()
    kept_features = [f for f, k in zip(X.columns, kept_mask) if k]
    print(f"  VarianceThreshold: {len(kept_features)} features")

    # SelectKBest to pre-reduce for PSO/ACO speed
    n_kbest = min(2000, len(kept_features))
    kbest = SelectKBest(f_classif, k=n_kbest)
    X_train_kb = kbest.fit_transform(X_train_var, y_train)
    X_test_kb = kbest.transform(X_test_var)
    kb_mask = kbest.get_support()
    kb_features = [f for f, k in zip(kept_features, kb_mask) if k]
    print(f"  SelectKBest: {len(kb_features)} features")

    return {
        "X_train_kb": X_train_kb,
        "X_test_kb": X_test_kb,
        "X_train_var": X_train_var,
        "X_test_var": X_test_var,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "selector": selector,
        "kept_features": kept_features,
        "kb_features": kb_features,
        "kb_mask": kb_mask,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PSO FEATURE SELECTION (on KBest-reduced set)
# ═══════════════════════════════════════════════════════════════════════════════


def run_pso(X: np.ndarray, y: np.ndarray, n_features: int, n_select: int) -> np.ndarray:
    """PSO feature selection. Returns indices into the input X."""
    print(f"\n{'='*70}")
    print(f"STEP 3: PSO ({n_features} → {n_select})")
    print(f"{'='*70}")

    np.random.seed(RANDOM_STATE)

    def fitness(position):
        top_idx = np.argsort(position)[-n_select:]
        X_sub = X[:, top_idx]
        rf = RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=-1, random_state=RANDOM_STATE)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        return np.mean([
            accuracy_score(y[v], rf.fit(X_sub[t], y[t]).predict(X_sub[v]))
            for t, v in skf.split(X_sub, y)
        ])

    pop = PSO_POP
    dim = n_features
    pos = np.random.uniform(0, 1, (pop, dim))
    vel = np.random.uniform(-0.1, 0.1, (pop, dim))
    fit = np.array([fitness(pos[i]) for i in range(pop)])

    pb_pos, pb_fit = pos.copy(), fit.copy()
    gb_idx = np.argmax(fit)
    gb_pos, gb_fit = pos[gb_idx].copy(), fit[gb_idx]

    w, c1, c2 = 0.7, 1.5, 1.5
    print(f"  Pop: {pop}, Iter: {PSO_ITER}, Initial best: {gb_fit:.4f}")

    for it in range(PSO_ITER):
        r1, r2 = np.random.uniform(0, 1, (pop, dim)), np.random.uniform(0, 1, (pop, dim))
        vel = w * vel + c1 * r1 * (pb_pos - pos) + c2 * r2 * (gb_pos - pos)
        pos = np.clip(pos + vel, 0, 1)

        for i in range(pop):
            f = fitness(pos[i])
            fit[i] = f
            if f > pb_fit[i]:
                pb_fit[i] = f
                pb_pos[i] = pos[i].copy()

        bi = np.argmax(fit)
        if fit[bi] > gb_fit:
            gb_fit = fit[bi]
            gb_pos = pos[bi].copy()

        if (it + 1) % 3 == 0:
            print(f"  Iter {it+1:2d}/{PSO_ITER} — best: {gb_fit:.4f}")

    selected = np.argsort(gb_pos)[-n_select:]
    selected.sort()
    print(f"  PSO done — {len(selected)} features, fitness: {gb_fit:.4f}")
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# ACO FEATURE SELECTION (on PSO-reduced set)
# ═══════════════════════════════════════════════════════════════════════════════


def run_aco(X: np.ndarray, y: np.ndarray, n_features: int, n_select: int) -> np.ndarray:
    """ACO feature selection. Returns indices into the input X."""
    print(f"\n{'='*70}")
    print(f"STEP 4: ACO ({n_features} → {n_select})")
    print(f"{'='*70}")

    np.random.seed(RANDOM_STATE + 1)

    # Heuristic: RF feature importance
    print("  Computing RF importance...")
    rf_h = RandomForestClassifier(n_estimators=80, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE)
    rf_h.fit(X, y)
    heur = rf_h.feature_importances_
    heur = heur / (heur.max() + 1e-10)

    tau = np.ones(n_features) * 0.1
    best_fit, best_sol = -1.0, None

    def aco_fit(mask):
        if mask.sum() == 0:
            return 0.0
        Xs = X[:, mask]
        rf = RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=-1, random_state=RANDOM_STATE)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        return np.mean([
            accuracy_score(y[v], rf.fit(Xs[t], y[t]).predict(Xs[v]))
            for t, v in skf.split(Xs, y)
        ])

    print(f"  Ants: {ACO_POP}, Iter: {ACO_ITER}")

    for it in range(ACO_ITER):
        sols, fits = [], []
        for _ in range(ACO_POP):
            prob = (tau ** ACO_ALPHA) * ((heur + 0.01) ** ACO_BETA)
            prob /= prob.sum()
            sel = np.random.choice(n_features, size=n_select, replace=False, p=prob)
            mask = np.zeros(n_features, dtype=bool)
            mask[sel] = True
            f = aco_fit(mask)
            sols.append(mask)
            fits.append(f)

        fits = np.array(fits)
        bi = np.argmax(fits)
        if fits[bi] > best_fit:
            best_fit = fits[bi]
            best_sol = sols[bi].copy()

        tau *= (1 - ACO_EVAPORATION)
        top_k = max(3, ACO_POP // 3)
        for idx in np.argsort(fits)[-top_k:]:
            tau[sols[idx]] += fits[idx] * 2.0

        if (it + 1) % 3 == 0:
            print(f"  Iter {it+1:2d}/{ACO_ITER} — best: {best_fit:.4f}")

    selected = np.where(best_sol)[0]
    selected.sort()
    print(f"  ACO done — {len(selected)} features, fitness: {best_fit:.4f}")
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# DNN TRAINING
# ═══════════════════════════════════════════════════════════════════════════════


def train_dnn(X_train, y_train, X_test, y_test, n_classes):
    """Train the DNN and return the model."""
    print(f"\n{'='*70}")
    print("STEP 5: Training DNN")
    print(f"{'='*70}")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=DNN_EPOCHS, batch_size=DNN_BATCH,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)],
        verbose=1,
    )

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test accuracy: {acc:.4f}")
    names = [DISEASE_MAP[i] for i in range(n_classes)]
    print(classification_report(y_test, y_pred, target_names=names))
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════


def save_artifacts(model, scaler, selector, pso_indices_kbest, aco_indices_pso,
                   kb_features, kept_features, X_train_final):
    """Save all artifacts, converting indices back to variance-filtered space."""
    print(f"\n{'='*70}")
    print("STEP 6: Saving artifacts")
    print(f"{'='*70}")

    # Convert PSO indices (in KBest space) back to variance-filtered space
    # pso_indices_kbest are indices into kb_features
    # kb_features are the feature names kept by SelectKBest from kept_features
    pso_feature_names = [kb_features[i] for i in pso_indices_kbest]

    # Convert ACO indices (in PSO space) back to variance-filtered space
    aco_feature_names = [pso_feature_names[i] for i in aco_indices_pso]

    # Map feature names to indices in kept_features
    kept_name_to_idx = {name: i for i, name in enumerate(kept_features)}
    pso_in_var = np.array([kept_name_to_idx[n] for n in pso_feature_names], dtype=np.int32)
    aco_in_pso = np.array([kept_name_to_idx[n] for n in aco_feature_names], dtype=np.int32)

    # Save
    model.save(str(ARTIFACTS_DIR / "hybrid_pso_aco_dl_model.keras"))
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")
    joblib.dump(selector, ARTIFACTS_DIR / "variance_selector.pkl")
    joblib.dump(pso_in_var, ARTIFACTS_DIR / "pso_features.pkl")
    joblib.dump(aco_in_pso, ARTIFACTS_DIR / "aco_features.pkl")
    np.save(ARTIFACTS_DIR / "X_train_fs.npy", X_train_final)

    info = {
        "diseases": DISEASE_MAP,
        "n_input_features": int(scaler.n_features_in_),
        "n_after_variance": len(kept_features),
        "n_after_pso": int(len(pso_in_var)),
        "n_final": int(len(aco_in_pso)),
        "selected_probe_ids": aco_feature_names,
    }
    with open(ARTIFACTS_DIR / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"  Model: hybrid_pso_aco_dl_model.keras")
    print(f"  Scaler: scaler.pkl ({scaler.n_features_in_} features)")
    print(f"  Variance selector: variance_selector.pkl")
    print(f"  PSO features: pso_features.pkl ({len(pso_in_var)} → indices in variance space)")
    print(f"  ACO features: aco_features.pkl ({len(aco_in_pso)} → indices in variance space)")
    print(f"  X_train_fs: {X_train_final.shape}")
    print(f"  Training info: training_info.json")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    print("\n" + "█" * 70)
    print("  EXPLAINABLE MULTI-DISEASE DIAGNOSIS SYSTEM — RETRAINING")
    print("  Removing: Diabetes | Adding: Parkinson's Disease")
    print("█" * 70)

    t_start = time.time()

    # 1. Load
    X, y = load_all_datasets()

    # 2. Preprocess
    data = preprocess(X, y)

    # 3. PSO on KBest features
    pso_in_kb = run_pso(
        data["X_train_kb"], data["y_train"],
        data["X_train_kb"].shape[1], 500
    )

    X_train_pso = data["X_train_kb"][:, pso_in_kb]
    X_test_pso = data["X_test_kb"][:, pso_in_kb]

    # 4. ACO on PSO features
    aco_in_pso = run_aco(
        X_train_pso, data["y_train"],
        X_train_pso.shape[1], TARGET_FEATURES
    )

    X_train_final = X_train_pso[:, aco_in_pso]
    X_test_final = X_test_pso[:, aco_in_pso]

    print(f"\n  Final features: {X_train_final.shape[1]}")

    # 5. Train DNN
    model = train_dnn(X_train_final, data["y_train"], X_test_final, data["y_test"], len(DISEASE_MAP))

    # 6. Save
    save_artifacts(
        model=model,
        scaler=data["scaler"],
        selector=data["selector"],
        pso_indices_kbest=pso_in_kb,
        aco_indices_pso=aco_in_pso,
        kb_features=data["kb_features"],
        kept_features=data["kept_features"],
        X_train_final=X_train_final,
    )

    t_total = time.time() - t_start
    print(f"\n{'█'*70}")
    print(f"  RETRAINING COMPLETE — {t_total:.1f}s")
    print(f"  Classes: {', '.join(DISEASE_MAP.values())}")
    print(f"  Features: {X.shape[1]} → variance → KBest → PSO({len(pso_in_kb)}) → ACO({len(aco_in_pso)})")
    print(f"  Train: {len(X_train_final)}, Test: {len(X_test_final)}")
    print(f"{'█'*70}\n")


if __name__ == "__main__":
    main()
