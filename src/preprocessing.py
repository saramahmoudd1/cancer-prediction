import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
import joblib
import os


def load_data():

    # قراءة الملفات
    gene_data = pd.read_csv("data/data.csv")
    labels = pd.read_csv("data/labels.csv")

    # دمجهم باستخدام اسم الـ sample
    data = pd.merge(gene_data, labels, on="Unnamed: 0")

    # تغيير اسم العمود
    data.rename(columns={"Unnamed: 0": "Sample"}, inplace=True)

    return data


def preprocess():

    data = load_data()

    print("Dataset Shape:", data.shape)

    # Features
    # Features
    X = data.drop(columns=["Sample", "Class"])

# احفظي أسماء الأعمدة
    feature_names = X.columns

# إزالة الأعمدة الثابتة
    selector = VarianceThreshold(threshold=0)

    X = selector.fit_transform(X)

    os.makedirs("models", exist_ok=True)

    joblib.dump(
    selector,
    "models/variance_selector.pkl"
    )

# تحديث أسماء الجينات بعد حذف الأعمدة الثابتة
    feature_names = feature_names[selector.get_support()].to_numpy()

    # Labels
    y = data["Class"]

    # تحويل أسماء السرطانات لأرقام
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        encoder,
        scaler,
        feature_names
    )