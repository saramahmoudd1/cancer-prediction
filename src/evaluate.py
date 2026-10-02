import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def evaluate_models(
        trained_models,
        X_test,
        y_test,
        encoder):

    os.makedirs("outputs", exist_ok=True)

    results = []

    best_accuracy = 0
    best_model = None
    best_name = ""

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    for name, model in trained_models.items():

        print(f"\nEvaluating {name}...")

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted"
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted"
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted"
        )

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\nClassification Report\n")

        print(
            classification_report(
                y_test,
                predictions,
                target_names=encoder.classes_
            )
        )

        cm = confusion_matrix(
            y_test,
            predictions
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=encoder.classes_
        )

        fig, ax = plt.subplots(figsize=(7,7))

        disp.plot(ax=ax)

        plt.title(name)

        plt.tight_layout()

        file_name = (
            name.lower()
            .replace(" ", "_")
            + "_confusion_matrix.png"
        )

        plt.savefig(
            f"outputs/{file_name}"
        )

        plt.close()

        results.append({

            "Model": name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1 Score": f1

        })

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_model = model

            best_name = name

    results_df = pd.DataFrame(results)

    print("\n")
    print(results_df)

    results_df.to_csv(
        "outputs/model_results.csv",
        index=False
    )

    plt.figure(figsize=(8,5))

    plt.bar(
        results_df["Model"],
        results_df["Accuracy"]
    )

    plt.ylabel("Accuracy")

    plt.title("Model Comparison")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        "outputs/model_comparison.png"
    )

    plt.close()

    joblib.dump(
        best_model,
        "models/best_model.pkl"
    )

    print("\nBest Model :", best_name)

    print("Accuracy :", best_accuracy)

    print("\nBest Model Saved Successfully.")

    return results_df