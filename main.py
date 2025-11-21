Train models (Voting RF+LR + standalone RF + LR) and save pipelines.
Usage:
    python main.py --data /path/to/Mental_Health_and_Social_Media_Balance_Dataset.csv --mode binary --outdir models
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','bool','category']).columns.tolist()
    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_cols),
                                                   ('cat', categorical_transformer, cat_cols)], remainder='drop')
    return preprocessor, num_cols, cat_cols

def main(args):
    df = pd.read_csv(args.data)
    candidate_targets = ['target','label','class','diagnosis','mental_health','stress_level','outcome','y','sentiment','Happiness_Index(1-10)']
    target_col = None
    for c in candidate_targets:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        target_col = df.columns[-1]
    print("Target column:", target_col)
    df = df[~df[target_col].isnull()].reset_index(drop=True)

    if args.mode == 'binary':
        df['happy_binary'] = (df[target_col] >= 8).astype(int)
        y = df['happy_binary'].values
        X = df.drop(columns=[target_col,'happy_binary'])
    else:
        y_raw = df[target_col]
        if y_raw.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y_raw.astype(str))
        else:
            y = y_raw.values
        X = df.drop(columns=[target_col])

    preprocessor, num_cols, cat_cols = build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)

    rf = RandomForestClassifier(n_estimators=250, random_state=42, class_weight='balanced' if args.mode=='binary' else None)
    lr = LogisticRegression(max_iter=3000, random_state=42, class_weight='balanced' if args.mode=='binary' else None)
    voting = VotingClassifier(estimators=[('rf', rf), ('lr', lr)], voting='soft')

    pipeline_voting = Pipeline([('preprocessor', preprocessor), ('classifier', voting)])
    pipeline_rf = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=250, random_state=42, class_weight='balanced' if args.mode=='binary' else None))])
    pipeline_lr = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=3000, random_state=42, class_weight='balanced' if args.mode=='binary' else None))])

    if args.upsample and args.mode=='binary':
        train = X_train.copy(); train['y'] = y_train
        c0 = train[train['y']==0]; c1 = train[train['y']==1]
        if len(c0) < len(c1):
            c0_up = resample(c0, replace=True, n_samples=len(c1), random_state=42)
            train_bal = pd.concat([c1, c0_up])
        else:
            c1_up = resample(c1, replace=True, n_samples=len(c0), random_state=42)
            train_bal = pd.concat([c0, c1_up])
        y_train_bal = train_bal['y'].values
        X_train_bal = train_bal.drop(columns=['y'])
        pipeline_voting.fit(X_train_bal, y_train_bal)
        pipeline_rf.fit(X_train_bal, y_train_bal)
        pipeline_lr.fit(X_train_bal, y_train_bal)
    else:
        pipeline_voting.fit(X_train, y_train)
        pipeline_rf.fit(X_train, y_train)
        pipeline_lr.fit(X_train, y_train)

    # evaluate
    for name, p in [('Voting', pipeline_voting), ('RF', pipeline_rf), ('LR', pipeline_lr)]:
        y_pred = p.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} acc: {acc*100:.2f}%")
        print(classification_report(y_test, y_pred, zero_division=0))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline_voting, outdir / 'pipeline_voting.pkl')
    joblib.dump(pipeline_rf, outdir / 'pipeline_rf.pkl')
    joblib.dump(pipeline_lr, outdir / 'pipeline_lr.pkl')
    print("Saved models to", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="/mnt/data/Mental_Health_and_Social_Media_Balance_Dataset.csv")
    parser.add_argument('--mode', type=str, choices=['binary','multi'], default='binary')
    parser.add_argument('--upsample', action='store_true', help="Apply simple upsampling")
    parser.add_argument('--outdir', type=str, default='models')
    args = parser.parse_args()
    main(args)
