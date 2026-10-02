from src.eda import run_eda
from src.preprocessing import preprocess
from src.feature_selection import (
    select_features,
    random_forest_importance
)
from src.train import train_all_models
import joblib
from src.evaluate import evaluate_models
from src.train import (
    train_all_models,
    save_all_models
)


run_eda()

(
    X_train,
    X_test,
    y_train,
    y_test,
    encoder,
    scaler,
    feature_names
) = preprocess()

(
    X_train_selected,
    X_test_selected,
    selected_features
) = select_features(
    X_train,
    X_test,
    y_train,
    feature_names,
    k=200
)

random_forest_importance(
    X_train_selected,
    y_train,
    selected_features
)

print("\nFinal Train Shape :", X_train_selected.shape)
print("Final Test Shape  :", X_test_selected.shape)


trained_models = train_all_models(
    X_train_selected,
    y_train
)

selector = joblib.load(
    "models/selectkbest.pkl"
)

save_all_models(
    trained_models,
    encoder,
    scaler,
    selector
)
evaluate_models(
    trained_models,
    X_test_selected,
    y_test,
    encoder
)

