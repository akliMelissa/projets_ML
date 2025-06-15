
import os, sys, joblib, pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit-Scoring", page_icon="")
st.title("Credit-Scoring App")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# getting the trained prediction model 
BEST_MODEL = joblib.load("models/best_model.pkl")      
FEATURES   = joblib.load("models/feature_names.pkl")   

if hasattr(BEST_MODEL, "named_steps"):                 
    clf_name = BEST_MODEL.named_steps["model"].__class__.__name__
else:                                                 
    clf_name = BEST_MODEL.__class__.__name__

st.sidebar.markdown(f"**Model in use:** {clf_name}")

# input features
inputs = {}
for feat in [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]:
    inputs[feat] = st.sidebar.number_input(feat, value=0.0, step=0.1)

#nprediction
df_in = pd.DataFrame([inputs])
df_in = pd.get_dummies(df_in, drop_first=True).reindex(columns=FEATURES, fill_value=0)
X = df_in.values                                         

prob = float(BEST_MODEL.predict_proba(X)[0, 1])
pred = int(BEST_MODEL.predict(X)[0])

# result
st.subheader("Results")
st.write(f"**Default risk:** {prob:.2%}")
st.write(f"**Prediction:** {'❌ High Risk' if pred else '✅ Low Risk'}")
