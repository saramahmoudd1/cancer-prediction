import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def get_models():

    models = {

        "Logistic Regression":
        LogisticRegression(
            max_iter=3000,
            random_state=42
        ),

        "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

        "Support Vector Machine":
        SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ),

        "XGBoost":
        XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42
        )

    }

    return models


def train_model(model,
                X_train,
                y_train):

    model.fit(X_train, y_train)

    return model


def train_all_models(
        X_train,
        y_train):

    os.makedirs("models", exist_ok=True)

    models = get_models()

    trained_models = {}

    print("=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)

    for name, model in models.items():

        print(f"\nTraining {name} ...")

        trained_models[name] = train_model(
            model,
            X_train,
            y_train
        )

        print(f"{name} Finished.")

    return trained_models


def save_model(model,
               file_name):

    joblib.dump(
        model,
        f"models/{file_name}"
    )

    print(f"Saved -> models/{file_name}")
    
def save_all_models(
        trained_models,
        encoder,
        scaler,
        selector):

    print("\nSaving Models...\n")

    for name, model in trained_models.items():

        file_name = (
            name.lower()
            .replace(" ", "_")
            + ".pkl"
        )

        save_model(
            model,
            file_name
        )

    save_model(
        encoder,
        "label_encoder.pkl"
    )

    save_model(
        scaler,
        "scaler.pkl"
    )

    save_model(
        selector,
        "feature_selector.pkl"
    )

    print("\nAll Models Saved Successfully.\n")    