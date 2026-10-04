"""
L&T Finance Farmer Income Prediction Challenge
Team: katinnitk

End-to-end pipeline: load train/test, engineer features, train a 3-way
blended LightGBM ensemble, apply an MAPE-optimal shrinkage factor, and
write the submission CSV.

Usage:
    python solution.py --data LTF_Farmer_Income_data_with_dictionary_For_Share_KGP_v2.xlsx \
                        --sample sample_submission_file.csv \
                        --out submission.csv

Requires: pandas, numpy, scikit-learn, lightgbm, openpyxl
"""
import argparse
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

from features import build_features, add_target_encoding, TARGET

warnings.filterwarnings("ignore")
SEED = 42
NFOLD = 5

PARAMS = dict(
    objective="mae", metric="mae", learning_rate=0.05, num_leaves=80,
    min_data_in_leaf=35, feature_fraction=0.75, bagging_fraction=0.8,
    bagging_freq=1, lambda_l2=3.0, verbose=-1, n_jobs=-1, seed=SEED,
)


def mape(actual, pred):
    return np.mean(np.abs(actual - pred) / actual) * 100


def train_one_formulation(name, Xtr, Xte, y_model, inverse, folds, rounds=1400):
    """Train NFOLD LightGBM models on a given target transform, returning
    out-of-fold predictions (for validation) and test predictions
    (averaged across folds)."""
    oof = np.zeros(len(Xtr))
    test_pred_folds = np.zeros((NFOLD, len(Xte)))
    for k, (ti, vi) in enumerate(folds):
        ds_t = lgb.Dataset(Xtr.iloc[ti], y_model[ti])
        ds_v = lgb.Dataset(Xtr.iloc[vi], y_model[vi])
        m = lgb.train(PARAMS, ds_t, num_boost_round=rounds, valid_sets=[ds_v],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[vi] = m.predict(Xtr.iloc[vi], num_iteration=m.best_iteration)
        test_pred_folds[k] = m.predict(Xte, num_iteration=m.best_iteration)
    return oof, test_pred_folds


def main(args):
    xls = pd.ExcelFile(args.data)
    tr = pd.read_excel(xls, "TrainData")
    te = pd.read_excel(xls, "TestData")

    y = tr[TARGET].astype(float).values
    nonagri_tr = tr["Non_Agriculture_Income"].astype(float).fillna(0).values
    nonagri_te = te["Non_Agriculture_Income"].astype(float).fillna(0).values
    land_tr = tr["Total_Land_For_Agriculture"].astype(float)
    land_tr = land_tr.replace(0, np.nan).fillna(land_tr.median()).values
    land_te = te["Total_Land_For_Agriculture"].astype(float)
    land_te = land_te.replace(0, np.nan).fillna(pd.Series(land_tr).median()).values

    print("Building features...")
    Xtr = build_features(tr)
    Xte = build_features(te)
    Xte = Xte.reindex(columns=Xtr.columns)

    # native categoricals need a shared dtype across train/test
    for c in [c for c in Xtr.columns if c.startswith("cat_")]:
        cats = sorted({str(v) for v in Xtr[c]} | {str(v) for v in Xte[c]})
        dt = pd.CategoricalDtype(cats)
        Xtr[c] = Xtr[c].astype(str).astype(dt)
        Xte[c] = Xte[c].astype(str).astype(dt)

    kf = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
    folds = list(kf.split(Xtr))

    # out-of-fold target encoding for high-cardinality geography columns,
    # fit on log(income) so the tail doesn't dominate the encoding
    Xtr, Xte = add_target_encoding(Xtr, Xte, tr, te, pd.Series(np.log(y)), folds)
    print(f"Feature count: {Xtr.shape[1]}")

    # ---- three target formulations, blended -----------------------------
    # A: model log(total income) directly
    # B: model log1p(agricultural income) and add back the KNOWN
    #    Non_Agriculture_Income (target = Non_Agri + Agri income exactly,
    #    verified on training data: residual is never negative)
    # C: model log(income per hectare) and multiply back by landholding
    print("Training formulation A (log total income)...")
    pA_oof, pA_te = train_one_formulation("A", Xtr, Xte, np.log(y),
                                          lambda p, idx: np.exp(p), folds)

    print("Training formulation B (log agri-income + known non-agri)...")
    agri = np.maximum(y - nonagri_tr, 0)
    pB_oof, pB_te = train_one_formulation("B", Xtr, Xte, np.log1p(agri),
                                          lambda p, idx: np.expm1(p) + nonagri_tr[idx], folds)

    print("Training formulation C (log income-per-hectare * land)...")
    pC_oof, pC_te = train_one_formulation("C", Xtr, Xte, np.log(y / land_tr),
                                          lambda p, idx: np.exp(p) * land_tr[idx], folds)

    predA_oof, predB_oof, predC_oof = (
        np.exp(pA_oof), np.expm1(pB_oof) + nonagri_tr, np.exp(pC_oof) * land_tr)
    predA_te, predB_te, predC_te = (
        np.exp(pA_te.mean(axis=0)), np.expm1(pB_te.mean(axis=0)) + nonagri_te,
        np.exp(pC_te.mean(axis=0)) * land_te)

    # blend weights + MAPE-optimal global shrinkage, chosen by OOF grid search
    # (MAPE's optimum is a weighted median, not a mean, so a slight downward
    # shrink of the blended prediction reduces OOF MAPE)
    best = (None, 1e9)
    for wa in np.arange(0, 1.01, 0.02):
        for wb in np.arange(0, 1.01 - wa + 1e-9, 0.02):
            wc = 1 - wa - wb
            bl = wa * predA_oof + wb * predB_oof + wc * predC_oof
            for g in np.arange(0.85, 1.05, 0.0025):
                s = mape(y, bl * g)
                if s < best[1]:
                    best = ((wa, wb, wc, g), s)
    (wa, wb, wc, g), score = best
    print(f"Blend weights: A={wa:.2f} B={wb:.2f} C={wc:.2f}  shrink={g:.4f}")
    print(f"OOF MAPE: {score:.3f}")

    bl_te = wa * predA_te + wb * predB_te + wc * predC_te
    # income can never be lower than the known non-agricultural component
    final_te = np.maximum(bl_te * g, nonagri_te)

    sample = pd.read_csv(args.sample)
    id_to_pred = dict(zip(te["FarmerID"].values, final_te))
    sample["Income"] = sample["FarmerID2"].map(id_to_pred).round(0).astype(int)
    sample.to_csv(args.out, index=False)
    print(f"Saved {args.out} ({sample.shape[0]} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to the LTF training/test workbook (.xlsx)")
    ap.add_argument("--sample", required=True, help="path to the sample submission CSV")
    ap.add_argument("--out", default="submission.csv")
    main(ap.parse_args())
