import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_eda():

    # ==========================
    # Create outputs folder
    # ==========================

    os.makedirs("outputs", exist_ok=True)

    # ==========================
    # Load Data
    # ==========================

    gene_data = pd.read_csv("data/data.csv")
    labels = pd.read_csv("data/labels.csv")

    data = pd.merge(gene_data, labels, on="Unnamed: 0")

    data.rename(columns={"Unnamed: 0": "Sample"}, inplace=True)

    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(data.info())

    print()

    print("=" * 60)
    print("DATASET SHAPE")
    print("=" * 60)

    print(data.shape)

    print()

    print("=" * 60)
    print("FIRST FIVE ROWS")
    print("=" * 60)

    print(data.head())

    print()

    print("=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    print(data.isnull().sum().sum())

    print()

    print("=" * 60)
    print("DUPLICATED ROWS")
    print("=" * 60)

    print(data.duplicated().sum())

    print()

    print("=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)

    print(data["Class"].value_counts())

    print()

    print("=" * 60)
    print("STATISTICS")
    print("=" * 60)

    print(data.describe())

    # ===================================================
    # Cancer Distribution
    # ===================================================

    plt.figure(figsize=(8,5))

    sns.countplot(data=data,
                  x="Class",
                  order=data["Class"].value_counts().index)

    plt.title("Cancer Type Distribution")
    plt.xlabel("Cancer Type")
    plt.ylabel("Number of Samples")

    plt.tight_layout()

    plt.savefig("outputs/cancer_distribution.png")

    plt.close()

    # ===================================================
    # Boxplot
    # ===================================================

    plt.figure(figsize=(15,6))

    data.iloc[:,1:21].boxplot(rot=90)

    plt.title("First 20 Gene Expression")

    plt.tight_layout()

    plt.savefig("outputs/gene_boxplot.png")

    plt.close()

    # ===================================================
    # Histogram
    # ===================================================

    plt.figure(figsize=(14,8))

    data.iloc[:,1:6].hist(figsize=(14,8))

    plt.tight_layout()

    plt.savefig("outputs/gene_histograms.png")

    plt.close()

    # ===================================================
    # Correlation
    # ===================================================

    plt.figure(figsize=(12,10))

    corr = data.iloc[:,1:51].corr()

    sns.heatmap(corr,
                cmap="coolwarm")

    plt.title("Correlation Heatmap (First 50 Genes)")

    plt.tight_layout()

    plt.savefig("outputs/correlation_heatmap.png")

    plt.close()

    # ===================================================
    # PCA Visualization
    # ===================================================

    X = data.drop(columns=["Sample","Class"])

    y = data["Class"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame()

    pca_df["PC1"] = X_pca[:,0]
    pca_df["PC2"] = X_pca[:,1]
    pca_df["Class"] = y

    plt.figure(figsize=(10,7))

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Class",
        s=70
    )

    plt.title("PCA Visualization")

    plt.tight_layout()

    plt.savefig("outputs/pca_visualization.png")

    plt.close()

    # ===================================================
    # Variance Explained
    # ===================================================

    pca_full = PCA()

    pca_full.fit(X_scaled)

    variance = pca_full.explained_variance_ratio_

    plt.figure(figsize=(10,5))

    plt.plot(range(1,31),
             variance[:30],
             marker='o')

    plt.title("Explained Variance")

    plt.xlabel("Principal Components")

    plt.ylabel("Variance Ratio")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("outputs/explained_variance.png")

    plt.close()

    print("\nEDA Finished Successfully.")

    print("\nPlots Saved Inside outputs Folder.")