import os, json, joblib, argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_recall_fscore_support

DROP_COLS = ['EmployeeNumber','StandardHours','EmployeeCount','Over18']

def load_data(path):
    df = pd.read_csv(path)
    drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop)
    df['Attrition'] = df['Attrition'].map({'Yes':1,'No':0})
    return df

def build_pipeline(cat_cols, num_cols):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    scaler = StandardScaler()
    preprocess = ColumnTransformer([
        ('cat', ohe, cat_cols),
        ('num', scaler, num_cols)
    ])
    model = LogisticRegression(max_iter=2000, class_weight='balanced')
    return Pipeline([('preprocess', preprocess), ('clf', model)])

def save_metrics(y_true, y_pred, y_prob, out):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    metrics = {'accuracy':acc,'precision':prec,'recall':rec,'f1':f1,'roc_auc':auc}
    with open(out, 'w') as f: json.dump(metrics, f, indent=2)

def main(data_path, artifacts_dir):
    os.makedirs(artifacts_dir, exist_ok=True)
    df = load_data(data_path)
    y = df['Attrition']
    X = df.drop(columns=['Attrition'])
    cat = X.select_dtypes(include=['object']).columns.tolist()
    num = X.select_dtypes(exclude=['object']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    pipe = build_pipeline(cat, num)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:,1]

    print("\nModel Performance:\n")
    print(classification_report(y_test, y_pred, digits=3))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

    joblib.dump(pipe, f"{artifacts_dir}/model.joblib")

    with open(f"{artifacts_dir}/metadata.json", 'w') as f:
        json.dump({'cat_cols':cat,'num_cols':num}, f, indent=2)

    save_metrics(y_test,y_pred,y_prob,f"{artifacts_dir}/metrics.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/greendestination.csv')
    parser.add_argument('--artifacts', default='artifacts')
    args = parser.parse_args()
    main(args.data, args.artifacts)
