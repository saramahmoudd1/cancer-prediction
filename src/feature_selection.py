import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier


def select_features(
        X_train,
        X_test,
        y_train,
        feature_names,
        k=200):

    feature_names = np.asarray(feature_names)

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("FEATURE SELECTION")
    print("=" * 60)

    print(f"Original Features : {X_train.shape[1]}")

    selector = SelectKBest(
        score_func=f_classif,
        k=k
    )

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    selected_indices = selector.get_support(indices=True)

    selected_feature_names = feature_names[selected_indices]

    print(f"Selected Features : {len(selected_feature_names)}")

    joblib.dump(selector, "models/selectkbest.pkl")

    return (
        X_train_selected,
        X_test_selected,
        np.asarray(selected_feature_names)
    )

def random_forest_importance(
        X_train,
        y_train,
        feature_names,
        top_n=20):

    feature_names = np.asarray(feature_names)

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    importance = rf.feature_importances_

    sorted_idx = np.argsort(importance)[::-1]

    top_idx = sorted_idx[:top_n]

    top_features = feature_names[top_idx]
    top_importance = importance[top_idx]

    plt.figure(figsize=(14,6))

    plt.bar(range(top_n), top_importance)

    plt.xticks(
        range(top_n),
        top_features,
        rotation=90
    )

    plt.title("Top Important Genes")

    plt.tight_layout()

    plt.savefig("outputs/top_genes.png")

    plt.close()

    print("\nTop Important Genes\n")

    for i, (gene, score) in enumerate(zip(top_features, top_importance), start=1):
        print(f"{i}. {gene} : {score:.5f}")