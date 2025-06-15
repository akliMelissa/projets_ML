

from pathlib import Path
import joblib, xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from data_preprocessing import load_data, preprocess

# evaluating the model : returns AUC
def evaluate_model(name, model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"{name:13s} | Accuracy={acc*100:6.2f}% | AUC={auc*100:6.2f}%")
    return auc


def main() -> None:

    # loading the data set and preprocessing it 
    df = load_data("data/GiveMeSomeCredit-training.csv")
    X, y, feature_names = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=12, stratify=y)

    # with three different models 
    models = {
        "LogisticReg": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, solver="liblinear")),
            ]),
            {"model__C": [0.001 ,0.01, 0.1, 1]},
        ),
        "RandomForest": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(random_state=12)),
            ]),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
            },
        ),
        "XGBoost": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", xgb.XGBClassifier(eval_metric="logloss", random_state=12)),
            ]),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.01, 0.1],
            },
        ),
    }


    # evaluating the models to pick the best one 
    best_pipelines = {}
    test_auc = {}

    for name, (pipe, grid) in models.items():
        search = GridSearchCV(pipe, grid, cv=3, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)

        best_pipelines[name] = search.best_estimator_
        print(f"{name:13s} | best cv-AUC = {search.best_score_:.3f}"
              f" | params = {search.best_params_}")

        auc = evaluate_model(name, search.best_estimator_, X_test, y_test)
        test_auc[name] = auc

    winner_name = max(test_auc, key=test_auc.get)
    best_model  = best_pipelines[winner_name]
    print(f"\n>>> Best model: {winner_name} with {test_auc[winner_name]*100:.2f}% test-AUC")

    # saving the result
    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")
    print("Saved: models/best_model.pkl  and  models/feature_names.pkl")


if __name__ == "__main__":
    main()
