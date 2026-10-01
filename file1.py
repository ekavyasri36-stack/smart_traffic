import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# 🚦 SMART TRAFFIC MANAGEMENT USING ML
# ==========================================

print("=" * 60)
print("🚦 SMART TRAFFIC MANAGEMENT SYSTEM")
print("=" * 60)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("traffic.csv")

print("\n✅ Dataset loaded successfully!")

print("\n📊 Dataset Shape:")
print(df.shape)

print("\n📋 Available Columns:")
print(df.columns.tolist())


# ==========================================
# 2. CREATE TRAFFIC LEVEL
# ==========================================

def traffic_level(volume):

    if volume < 2000:
        return "Low Traffic"

    elif volume < 5000:
        return "Medium Traffic"

    else:
        return "High Traffic"


df["Traffic Level"] = df["traffic_volume"].apply(traffic_level)

print("\n🚦 Traffic Levels Created Successfully!")

print("\n📊 Traffic Level Distribution:")
print(df["Traffic Level"].value_counts())


# ==========================================
# 3. REMOVE TRAFFIC VOLUME
# ==========================================

df = df.drop(columns=["traffic_volume"])


# ==========================================
# 4. PROCESS DATE AND TIME
# ==========================================

if "date_time" in df.columns:

    print("\n🕒 Processing Date and Time...")

    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce"
    )

    df["Hour"] = df["date_time"].dt.hour
    df["Month"] = df["date_time"].dt.month
    df["DayOfWeek"] = df["date_time"].dt.dayofweek

    df = df.drop(columns=["date_time"])


# ==========================================
# 5. DEFINE INPUT AND OUTPUT
# ==========================================

target_column = "Traffic Level"

X = df.drop(columns=[target_column])

y = df[target_column]


print("\n🎯 WE ARE PREDICTING:")
print("🚦 Traffic Congestion Level")

print("\nPossible Outputs:")
print(y.unique())


# ==========================================
# 6. IDENTIFY COLUMN TYPES
# ==========================================

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()


categorical_features = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()


print("\n📊 Numerical Features:")
print(numeric_features)

print("\n📋 Categorical Features:")
print(categorical_features)


# ==========================================
# 7. PREPROCESS NUMERICAL DATA
# ==========================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# ==========================================
# 8. PREPROCESS TEXT DATA
# ==========================================

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ==========================================
# 9. COMBINE PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),

        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ==========================================
# 10. SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\n📊 DATASET SPLIT")

print("Training Records:", len(X_train))

print("Testing Records:", len(X_test))


# ==========================================
# 11. CREATE RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42,

    n_jobs=-1
)


# ==========================================
# 12. CREATE COMPLETE ML PIPELINE
# ==========================================

pipeline = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )

    ]
)


# ==========================================
# 13. TRAIN MODEL
# ==========================================

print("\n🤖 Training Traffic Prediction Model...")

pipeline.fit(
    X_train,
    y_train
)

print("✅ Model Training Completed!")


# ==========================================
# 14. MAKE PREDICTIONS
# ==========================================

y_pred = pipeline.predict(
    X_test
)


# ==========================================
# 15. MODEL PERFORMANCE
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)

print("📊 MODEL PERFORMANCE")

print("=" * 60)


print(
    f"\n🎯 Accuracy: {accuracy * 100:.2f}%"
)


print("\n📋 CLASSIFICATION REPORT:")

print(

    classification_report(

        y_test,

        y_pred,

        zero_division=0

    )

)


print("\n🔢 CONFUSION MATRIX:")

print(

    confusion_matrix(
        y_test,
        y_pred
    )

)


# ==========================================
# 16. SAMPLE PREDICTIONS
# ==========================================

print("\n" + "=" * 60)

print("🚦 SAMPLE TRAFFIC PREDICTIONS")

print("=" * 60)


sample_data = X_test.head(5)

sample_predictions = pipeline.predict(
    sample_data
)


for i, (actual, predicted) in enumerate(
    zip(
        y_test.head(5),
        sample_predictions
    ),
    start=1
):

    print(f"\n🚗 Traffic Record {i}")

    print(
        "Actual Traffic Level:   ",
        actual
    )

    print(
        "Predicted Traffic Level:",
        predicted
    )


# ==========================================
# 17. PROJECT COMPLETED
# ==========================================

print("\n" + "=" * 60)

print("🎉 SMART TRAFFIC MANAGEMENT PROJECT COMPLETED!")

print("=" * 60)

print("\n📌 The model predicts:")

print("🟢 Low Traffic")
print("🟡 Medium Traffic")
print("🔴 High Traffic")

print("\n🤖 Algorithm Used: Random Forest Classifier")

print("\n⚠️ Educational Machine Learning Project")