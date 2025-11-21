import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from packaging import version
import sklearn


def build_preprocessor(X):
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','bool','category']).columns.tolist()

    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    else:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', ohe)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ],
        remainder='drop'
    )

    return preprocessor


def train_models(df, target_col, mode, upsample=False):

    if mode == "binary":
        df['binary_target'] = (df[target_col] >= 8).astype(int)
        y = df['binary_target'].values
        X = df.drop(columns=[target_col, 'binary_target'])
    else:
        raw = df[target_col]
        if raw.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(raw.astype(str))
        else:
            y = raw.values
        X = df.drop(columns=[target_col])

    preprocessor = build_preprocessor(X)

    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    rf = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight='balanced' if mode == "binary" else None
    )

    lr = LogisticRegression(
        max_iter=3000,
        random_state=42,
        class_weight='balanced' if mode == "binary" else None
    )

    voting = VotingClassifier(
        estimators=[('rf', rf), ('lr', lr)],
        voting='soft'
    )

    pipeline_voting = Pipeline([('preprocessor', preprocessor), ('classifier', voting)])
    pipeline_rf = Pipeline([('preprocessor', preprocessor), ('classifier', rf)])
    pipeline_lr = Pipeline([('preprocessor', preprocessor), ('classifier', lr)])

    if upsample and mode == "binary":
        train_df = X_train.copy()
        train_df['y'] = y_train

        class0 = train_df[train_df['y'] == 0]
        class1 = train_df[train_df['y'] == 1]

        if len(class0) < len(class1):
            class0_up = resample(class0, replace=True, n_samples=len(class1), random_state=42)
            train_bal = pd.concat([class1, class0_up])
        else:
            class1_up = resample(class1, replace=True, n_samples=len(class0), random_state=42)
            train_bal = pd.concat([class0, class1_up])

        y_train_final = train_bal['y'].values
        X_train_final = train_bal.drop(columns=['y'])
    else:
        X_train_final, y_train_final = X_train, y_train

    pipeline_voting.fit(X_train_final, y_train_final)
    pipeline_rf.fit(X_train_final, y_train_final)
    pipeline_lr.fit(X_train_final, y_train_final)

    models = {
        "Voting": pipeline_voting,
        "RF": pipeline_rf,
        "LR": pipeline_lr
    }

    # evaluation
    y_pred = pipeline_voting.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\nVoting Accuracy:", acc * 100)
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["binary", "multi"], default="binary")
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--outdir", type=str, default="models")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    candidate_targets = [
        "Happiness_Index(1-10)", "target", "label", "class", "y"
    ]
    target_col = None
    for c in candidate_targets:
        if c in df.columns:
            target_col = c
            break

    if target_col is None:
        raise ValueError("Target column not found.")

    df = df[~df[target_col].isnull()].reset_index(drop=True)

    models = train_models(df, target_col, args.mode, args.upsample)

    out = Path(args.outdir)
    out.mkdir(exist_ok=True)

    for name, model in models.items():
        joblib.dump(model, out / f"{name.lower()}.pkl")

    print("\nSaved models to", out)


if __name__ == "__main__":
    main()
