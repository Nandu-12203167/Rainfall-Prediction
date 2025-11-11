import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names: {df.columns.tolist()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")

    df.columns = df.columns.str.strip()

    numerical_cols = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
                      'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df['rainfall'] = df['rainfall'].map({'yes': 1, 'no': 0})

    feature_columns = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
                       'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

    X = df[feature_columns]
    y = df['rainfall']

    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]

    print(f"\nFinal dataset shape after cleaning: {X.shape}")
    print(f"Class distribution:\n{y.value_counts()}")

    return X, y, feature_columns

def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ''

    print("\n" + "="*60)
    print("MODEL TRAINING RESULTS")
    print("="*60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n{name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Rain', 'Rain']))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model_name} with accuracy: {best_accuracy:.4f}")
    print("="*60)

    return best_model, best_model_name, best_accuracy

def main():
    data_path = 'rainfall_cleaned.csv'

    print("Loading and preprocessing data...")
    X, y, feature_columns = load_and_preprocess_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining models...")
    best_model, best_model_name, best_accuracy = train_models(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    os.makedirs('models', exist_ok=True)

    joblib.dump(best_model, 'models/rainfall_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_columns, 'models/feature_columns.pkl')

    metadata = {
        'model_name': best_model_name,
        'accuracy': best_accuracy,
        'features': feature_columns
    }
    joblib.dump(metadata, 'models/metadata.pkl')

    print("\n" + "="*60)
    print("Model and preprocessing artifacts saved successfully!")
    print("="*60)
    print(f"Model: models/rainfall_model.pkl")
    print(f"Scaler: models/scaler.pkl")
    print(f"Features: models/feature_columns.pkl")
    print(f"Metadata: models/metadata.pkl")

if __name__ == "__main__":
    main()
